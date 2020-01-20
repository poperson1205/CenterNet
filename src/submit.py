from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
import pandas as pd

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def submit(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  # opt.debug = max(opt.debug, 1)

  # Import network
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  # Specify data path
  DATA_PATH = '/workspace/code/pku-autonomous-driving/data/'
  # TEST_IMAGE_DIR = DATA_PATH + 'validation_images/'
  TEST_IMAGE_DIR = DATA_PATH + 'test_images/'
  TEST_MASK_DIR = DATA_PATH + 'test_masks/'

  if os.path.isdir(TEST_IMAGE_DIR) is False:
    return

  # Parse image paths from input directory
  image_names = []
  mask_names = []
  ls = os.listdir(TEST_IMAGE_DIR)
  for file_name in sorted(ls):
      ext = file_name[file_name.rfind('.') + 1:].lower()
      if ext in image_ext:
          image_names.append(os.path.join(TEST_IMAGE_DIR, file_name))
          mask_names.append(os.path.join(TEST_MASK_DIR, file_name))
  
  # Predict
  list_image_id = []
  list_prediction_string = []
  for (image_name, mask_name) in zip(image_names, mask_names):
      print('Progress: {} / {}'.format(len(list_image_id), len(image_names)))

      image_id = os.path.basename(image_name).rsplit('.', 1)[0]

      dets = detector.run(image_name, {'mask_path': mask_name})['results']

      prediction_string = ''
      for cat in dets:
          for i in range(len(dets[cat])):
              score = dets[cat][i, 4]
              pose = dets[cat][i, 5:11]
              if score > opt.vis_thresh:
                prediction_string += '{} {} {} {} {} {} {} '.format(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], score)
      prediction_string = prediction_string.strip()

      # # Blind guess
      # if prediction_string == '':
      #     prediction_string = '0.155064 -3.14 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 -3.14 3.14 -3.25322 8.459775 44.89555 0.8 0.155064 0 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 0 3.14 -3.25322 8.459775 44.89555 0.8 0.155064 3.14 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 3.14 3.14 -3.25322 8.459775 44.89555 0.8'
      # prediction_string = '0.155064 -3.14 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 -3.14 3.14 -3.25322 8.459775 44.89555 0.8 0.155064 0 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 0 3.14 -3.25322 8.459775 44.89555 0.8 0.155064 3.14 -3.14 -3.25322 8.459775 44.89555 0.8 0.155064 3.14 3.14 -3.25322 8.459775 44.89555 0.8'

      list_image_id.append(image_id)
      list_prediction_string.append(prediction_string)

  # Write submission file
  submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')[:len(image_names)]
  submission['ImageId'] = list_image_id
  submission['PredictionString'] = list_prediction_string
  submission.to_csv(DATA_PATH + 'submission.csv', index=False)

if __name__ == '__main__':
  opt = opts().parse()
  submit(opt)
