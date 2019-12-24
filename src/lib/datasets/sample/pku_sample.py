from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco

class PkuSample(data.Dataset):
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = self.calib

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    if self.opt.keep_res:
      s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)
    
    aug = False
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
      aug = True
      sf = self.opt.scale
      cf = self.opt.shift
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    # if self.split == 'train' and not self.opt.no_color_aug:
    #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])

    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    xyz_rot = np.zeros((self.max_objs, 6), dtype=np.float32)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    for k in range(num_objs):
        ann = anns[k]

        model_type = ann['bbox']
        yaw = ann['yaw']
        pitch = ann['pitch']
        roll = ann['roll']
        x = ann['x']
        y = ann['y']
        z = ann['z']

        # if flipped:
        #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        x, y, z = affine_transform((x, y, z), trans_output)
        x = np.clip(x, 0, self.opt.output_w - 1)
        y = np.clip(y, 0, self.opt.output_h - 1)

        # TODO: select right size for gaussian
        radius = gaussian_radius((100, 100))
        radius = max(0, int(radius))
        ct = np.array([x, y], dtype=np.float32)
        draw_gaussian(hm[1], ct, radius)

        wh[k] = 1. * w, 1. * h
        gt_det.append([ct[0], ct[1], 1] + \
                        self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
                        [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
        if self.opt.reg_bbox:
            gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
        # if (not self.opt.car_only) or cls_id == 1: # Only estimate ADD for cars !!!

    ret = {'input': inp, 'hm': hm, 'xyz_rot': xyz_rot}

    return ret