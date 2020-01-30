from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
import pandas as pd
import math

from opts import opts
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from tools.pp_keypoint_parser import get_keypoints

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

calib = np.array([[2304.5479, 0, 1686.2379, 0.0],
                  [0, 2305.8757, 1354.9849, 0.0],
                  [0, 0, 1., 0.0]], dtype=np.float32)

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
              score = np.array(dets[cat][i][4])
              points = np.array(dets[cat][i][5:31])
              points = np.reshape(points, (-1, 2))
              if score > opt.vis_thresh:
                # Compute 6D pose from points
                obj_points = get_keypoints('/workspace/code/pku-autonomous-driving/data/car_models_obj/mazida-6-2015.pp')
                obj_points = np.vstack((obj_points, np.array([[0.0, 0.0, 0.0]])))
                # _, rvec, tvec = cv2.solvePnP(obj_points, points[:12].astype(np.float64), calib[:,:3], np.zeros((4,1), dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
                _, rvec, tvec, _ = cv2.solvePnPRansac(obj_points, points.astype(np.float64), calib[:,:3], np.zeros((4,1), dtype=np.float64), flags=cv2.SOLVEPNP_EPNP)
                rmat, _ = cv2.Rodrigues(rvec)
                cam_pos = -np.matrix(rmat).T * np.matrix(tvec)
                P = np.hstack((rmat, tvec))
                euler_angles_radians = -cv2.decomposeProjectionMatrix(P)[6] / 180.0 * math.pi
                yaw = euler_angles_radians[0]
                pitch = euler_angles_radians[1]
                roll = euler_angles_radians[2]

                prediction_string += '{} {} {} {} {} {} {} '.format(-yaw, -pitch, -roll, tvec[0][0], tvec[1][0], tvec[2][0], score)
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
