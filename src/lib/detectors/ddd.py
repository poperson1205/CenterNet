from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch


from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from .base_detector import BaseDetector

class DddDetector(BaseDetector):
  def __init__(self, opt):
    super(DddDetector, self).__init__(opt)
    self.calib = np.array([[707.0493, 0, 604.0814, 45.75831],
                           [0, 707.0493, 180.5066, -0.3454157],
                           [0, 0, 1., 0.004981016]], dtype=np.float32)


  def pre_process(self, image, scale, calib=None):
    height, width = image.shape[0:2]
    
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    c = np.array([width / 2, height / 2], dtype=np.float32)
    if self.opt.keep_res:
      s = np.array([inp_width, inp_height], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = image #cv2.resize(image, (width, height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = (inp_image.astype(np.float32) / 255.)
    inp_image = (inp_image - self.mean) / self.std
    images = inp_image.transpose(2, 0, 1)[np.newaxis, ...]
    calib = np.array(calib, dtype=np.float32) if calib is not None \
            else self.calib
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio,
            'calib': calib}
    return images, meta
  
  def process(self, images, masks=None, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if masks is not None:
        output['hm'] *= masks
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      wh = output['wh'] if self.opt.reg_bbox else None
      reg = output['reg'] if self.opt.reg_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=self.opt.K)
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    detections = ddd_post_process(
      dets.copy(), [meta['c']], [meta['s']], [meta['calib']], self.opt)
    self.this_calib = meta['calib']
    return detections[0]

  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1, masks=None):
    dets = dets.detach().cpu().numpy()
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = ((img * self.std + self.mean) * 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    if masks is not None:
      msk = (masks[0].detach().cpu().numpy() * 255).astype(np.uint8)
      debugger.add_img(msk, 'mask_img')
    debugger.add_blend_img(img, pred, 'pred_hm')
    debugger.add_ct_detection(
      img, dets[0], show_box=self.opt.reg_bbox, 
      center_thresh=self.opt.vis_thresh, img_id='det_pred')
  
  def show_results(self, debugger, image, results):
    debugger.add_3d_detection(
      image, results, self.this_calib,
      center_thresh=self.opt.vis_thresh, img_id='add_pred')
    debugger.add_bird_view(
      results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
    debugger.show_all_imgs(pause=self.pause)

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()

    mask = None
    if meta['mask'] is not None:
      mask = cv2.imread(meta['mask'], cv2.IMREAD_GRAYSCALE)

    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      masks = None
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta['calib'])
        if mask is not None:
          inp_mask = cv2.resize(mask, (self.opt.input_w // self.opt.down_ratio, self.opt.input_h // self.opt.down_ratio))
          inp_mask = 1.0 - inp_mask.astype(np.float32) / 255.
          inp_mask = inp_mask[np.newaxis, ...]
          masks = torch.from_numpy(inp_mask)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      if masks is not None:
        masks = masks.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, forward_time = self.process(images, masks, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale, masks)
      
      dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}