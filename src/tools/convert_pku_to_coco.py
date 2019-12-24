from sklearn.model_selection import train_test_split
import pandas as pd
import json
import numpy as np
import cv2

DEBUG = False

PATH = '/workspace/code/pku-autonomous-driving/data/'

# Load data
train = pd.read_csv(PATH + 'train.csv')
calib = np.array([[2304.5479, 0, 1686.2379, 0.0],
                  [0, 2305.8757, 1354.9849, 0.0],
                  [0, 0, 1., 0.0]], dtype=np.float32)


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
    model_type = int(predictions[0])
    yaw = float(predictions[1])
    pitch = float(predictions[2])
    roll = float(predictions[3])
    x = float(predictions[4])
    y = float(predictions[5])
    z = float(predictions[6])
    annotation = {
      'image_id': image_id,
      'id': idx,
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