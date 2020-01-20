from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_pku_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_pku_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetPkuDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetPkuDetector, self).__init__(opt)
  
  def process(self, images, masks=None, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()

      if masks is not None:
        mask_inds = masks.lt(1).float()
        hm = hm * mask_inds

      wh = output['wh']
      pose = output['pose']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_pku_decode(hm, wh, pose, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_pku_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 12)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, masks, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))

      if masks is not None:
        msk = masks[i].detach().cpu().numpy()
        msk = cv2.resize(msk, (img.shape[0], img.shape[1]), cv2.INTER_LINEAR)
        _, msk = cv2.threshold(msk, 0, 255, cv2.THRESH_BINARY)
        msk = (msk * 255).astype(np.uint8)
        debugger.add_mask(msk, img, 'mask_debug_{:.1f}'.format(scale), trans=0.8)

      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, mask = None):
    calib = np.array([[2304.5479, 0, 1686.2379, 0.0],
                      [0, 2305.8757, 1354.9849, 0.0],
                      [0, 0, 1., 0.0]], dtype=np.float32)
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
      for pose in results[j]:
        if pose[4] > self.opt.vis_thresh:
          debugger.add_pku(pose[5:11], calib, img_id='ctdet')
    
    if mask is not None:
      mask = (mask.astype(np.float32) / 255.)
      debugger.add_mask(mask, image, imgId='mask')
      for j in range(1, self.num_classes + 1):
        for bbox in results[j]:
          if bbox[4] > self.opt.vis_thresh:
            debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='mask')

    debugger.show_all_imgs(pause=self.pause)

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
      if meta is not None and type(meta) is dict and 'mask_path' in meta.keys() and os.path.isfile(meta['mask_path']):
        mask = cv2.imread(meta['mask_path'], cv2.IMREAD_GRAYSCALE)
      else:
        mask = None
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      trans_output = get_affine_transform(meta['c'], meta['s'], 0, (meta['out_width'], meta['out_height']))
      
      if mask is not None:
        masks = cv2.warpAffine(mask, trans_output, (meta['out_width'], meta['out_height']))
        _, masks = cv2.threshold(masks, 0, 255, cv2.THRESH_BINARY)
        # masks = (masks.astype(np.float32) / 255.)
        masks = np.expand_dims(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.to(self.opt.device)
      else:
        masks = None

      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, forward_time = self.process(images, masks, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, masks, dets, output, scale)
      
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
      self.show_results(debugger, image, results, mask)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}
