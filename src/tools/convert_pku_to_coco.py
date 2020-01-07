from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np
import cv2
from math import sin, cos
import car_models

DEBUG = True

PATH = '/workspace/code/pku-autonomous-driving/data/'

# Load data
train = pd.read_csv(PATH + 'train.csv')
calib = np.array([[2304.5479, 0, 1686.2379, 0.0],
                  [0, 2305.8757, 1354.9849, 0.0],
                  [0, 0, 1., 0.0]], dtype=np.float32)

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

def draw_obj(image, vertices, triangles):
  for t in triangles:
    coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
    cv2.polylines(image, np.int32([coord]), 1, (0,0,255))

def draw_bbox(image, bbox):
  bbox = bbox.astype(np.int32)
  cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 10)

# Split train/val
df_train, df_val = train_test_split(train, test_size=0.01, random_state=42)

# Make dictionary
def get_cat_info():
  cat_info = []
  
  cats = ['Car']
  cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
  for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

  return cat_info

def get_images_and_annotations(df):
  images = []
  annotations = []
  
  for row in df.iterrows():
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
      annotation = {
        'image_id': idx,
        'id': int(len(annotations) + 1),
        'category_id': 1,
        'model_type': model_type,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'x': x,
        'y': y,
        'z': z
      }

      if DEBUG:
        image = cv2.imread(PATH + 'train_images/' + image_info['file_name'])

        # Load 3D Model
        car_name = car_models.car_id2name[model_type].name
        with open(PATH + 'car_models_json/{0}.json'.format(car_name)) as json_file:
          data = json.load(json_file)
        vertices = np.array(data['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(data['faces']) - 1

        overlay = np.zeros_like(image)
        yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
        # I think the pitch and yaw should be exchanged
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
        draw_obj(overlay, img_cor_points, triangles)

        # Compute bounding box (2d)
        bbox = np.zeros(4, dtype=np.float32)
        bbox[0] = img_cor_points[:, 0].min()
        bbox[1] = img_cor_points[:, 1].min()
        bbox[2] = img_cor_points[:, 0].max()
        bbox[3] = img_cor_points[:, 1].max()
        draw_bbox(overlay, bbox)

        alpha = .5
        image = np.array(image)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        image = cv2.resize(image, (600, 600))
        cv2.imshow('image', image)
        cv2.waitKey()

      images.append(image_info)
      annotations.append(annotation)

  return images, annotations
  
# Train set
images, annotations = get_images_and_annotations(df_train)
ret = {'images': images, 'annotations': annotations, "categories": get_cat_info()}
json.dump(ret, open(PATH + 'train.json', 'w'))

# Validation set
images, annotations = get_images_and_annotations(df_val)
ret = {'images': images, 'annotations': annotations, "categories": get_cat_info()}
json.dump(ret, open(PATH + 'val.json', 'w'))