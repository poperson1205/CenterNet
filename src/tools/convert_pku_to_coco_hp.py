from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np
import cv2
from math import sin, cos
import car_models
import json_mesh_parser
import pp_keypoint_parser

def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def project_vertices(vertices, yaw, pitch, roll, x, y, z):
  yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
  yaw, pitch, roll = -pitch, -yaw, -roll
  Rt = np.eye(4)
  t = np.array([x, y, z])
  Rt[:3, 3] = t
  Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
  Rt = Rt[:3, :]
  P = np.ones((vertices.shape[0],vertices.shape[1]+1))
  P[:, :-1] = vertices
  P = P.T
  img_cor_points = np.dot(calib[:, :3], np.dot(Rt, P))
  img_cor_points = img_cor_points.T
  img_cor_points[:, 0] /= img_cor_points[:, 2]
  img_cor_points[:, 1] /= img_cor_points[:, 2]
  return img_cor_points[:, :2]

def compute_bbox(vertices, yaw, pitch, roll, x, y, z):
  # Project vertices
  img_cor_points = project_vertices(vertices, yaw, pitch, roll, x, y, z)

  # Compute bounding box
  bbox = [float(img_cor_points[:, 0].min()), float(img_cor_points[:, 1].min()), float(img_cor_points[:, 0].max()), float(img_cor_points[:, 1].max())]
  return bbox

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def draw_obj(image, vertices, triangles):
  for t in triangles:
    coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
    cv2.polylines(image, np.int32([coord]), 1, (0,0,255))

def draw_bbox(image, bbox):
  bbox = [int(t) for t in bbox]
  cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 10)

def draw_points(image, points):
  for p in points:
    cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), thickness = -1)

# Make dictionary
def get_cat_info():
  cat_info = []
  
  cats = ['Car']
  cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
  for i, cat in enumerate(cats):
    cat_info.append({
      'supercategory': 'car',
      'id': i + 1,
      'name': cat,
      'keypoints': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
      'skeleton': [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [8, 9], [9, 10], [10, 11], [11, 8]]
      })

  return cat_info

def get_images_and_annotations(df):
  images = []
  annotations = []
  
  # Parse CSV line-by-line
  current_count = 0
  total_count = df.shape[0]
  for row in df.iterrows():
    print("progress: {0}".format(float(current_count) / float(total_count) * 100))
    current_count += 1

    idx = row[0]
    image_id = row[1]['ImageId']
    prediction_string = row[1]['PredictionString']

    image_info = {'file_name': '{}.jpg'.format(image_id),
                'id': int(idx),
                'calib': calib.tolist()
                }
    
    predictions = prediction_string[:-1].split(' ')
    for i in range(0, int(len(predictions)/7)):
      model_type = int(predictions[i*7])
      yaw = float(predictions[i*7+1])
      pitch = float(predictions[i*7+2])
      roll = float(predictions[i*7+3])
      x = float(predictions[i*7+4])
      y = float(predictions[i*7+5])
      z = float(predictions[i*7+6])

      # Load 3D Model
      car_name = car_models.car_id2name[model_type].name
      category_id = car_models.car_id2name[model_type].categoryId   # 0: 2x, 1: 3x, 2: SUV
      vertices, triangles = json_mesh_parser.get_mesh(PATH + 'car_models_json/{0}.json'.format(car_name))

      # Compute bounding box
      bbox = compute_bbox(vertices, yaw, pitch, roll, x, y, z)

      # Load keypoints
      keypoints_3d = pp_keypoint_parser.get_keypoints(PATH + 'car_models_obj/{0}.pp'.format(car_name))
      keypoints_3d = np.vstack([keypoints_3d, [x, y, z]])  # Add center as a keypoint
      keypoints_2d = project_vertices(keypoints_3d, yaw, pitch, roll, x, y, z)
      coco_keypoints = []
      for p in keypoints_2d:
        coco_keypoints.append(p[0])
        coco_keypoints.append(p[1])
        coco_keypoints.append(2)

      annotation = {
        'image_id': idx,
        'id': int(len(annotations) + 1),
        'bbox': _bbox_to_coco_bbox(bbox),
        'category_id': category_id,
        'model_type': model_type,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'x': x,
        'y': y,
        'z': z,
        'num_keypoints': 13,
        'keypoints': coco_keypoints
      }

      if DEBUG:
        image = cv2.imread(PATH + 'train_images/' + image_info['file_name'])

        overlay = np.zeros_like(image)
        draw_obj(overlay, vertices, triangles)
        draw_bbox(overlay, bbox)
        draw_points(overlay, keypoints_2d)

        alpha = .5
        image = np.array(image)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        image = cv2.resize(image, (1024, 1024))
        cv2.imshow('image', image)
        cv2.waitKey()

      images.append(image_info)
      annotations.append(annotation)

  return images, annotations

if __name__ == '__main__':
  DEBUG = False

  PATH = '/workspace/code/pku-autonomous-driving/data/'

  # Load data
  df = pd.read_csv(PATH + 'train.csv')
  calib = np.array([[2304.5479, 0, 1686.2379, 0.0],
                    [0, 2305.8757, 1354.9849, 0.0],
                    [0, 0, 1., 0.0]], dtype=np.float32)

  # Split train/val
  df_train, df_val = train_test_split(df, test_size=0.01, random_state=42)

  # Train set
  images_train, annotations_train = get_images_and_annotations(df_train)
  ret = {'images': images_train, 'annotations': annotations_train, "categories": get_cat_info()}
  json.dump(ret, open(PATH + 'train.json', 'w'))

  # Validation set
  images_val, annotations_val = get_images_and_annotations(df_val)
  ret = {'images': images_val, 'annotations': annotations_val, "categories": get_cat_info()}
  json.dump(ret, open(PATH + 'val.json', 'w'))

  # TrainVal set
  images_train_val = images_train + images_val
  annotations_train_val = annotations_train + annotations_val
  ret = {'images': images_train_val, 'annotations': annotations_train_val, "categories": get_cat_info()}
  json.dump(ret, open(PATH + 'trainval.json', 'w'))