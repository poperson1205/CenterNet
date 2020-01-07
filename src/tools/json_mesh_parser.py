import json
import numpy as np

def get_mesh(path):
  with open(path) as json_file:
    data = json.load(json_file)
  vertices = np.array(data['vertices'])
  vertices[:, 1] = -vertices[:, 1]
  triangles = np.array(data['faces']) - 1
  return vertices, triangles