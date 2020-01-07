import json_mesh_parser
import os

def write_obj(vertices, triangles, path):
  with open(path, "w") as f:
    for i in range(len(vertices)):
      v = vertices[i]
      f.write("v {0} {1} {2} 1.0\n".format(v[0], v[1], v[2]))
    for i in range(len(triangles)):
      t = triangles[i] + 1
      f.write("f {0} {1} {2}\n".format(t[0], t[1], t[2]))

def main():
    input_dir = '/workspace/code/pku-autonomous-driving/data/car_models_json/'
    output_dir = '/workspace/code/pku-autonomous-driving/data/car_models_obj/'
    if os.path.dirname(input_dir) is False:
        return        

    for json_path in os.listdir(input_dir):
        json_path = os.path.join(input_dir, json_path)
        filename, ext = os.path.splitext(os.path.basename(json_path))
        if ext != '.json':
            continue

        vertices, triangles = json_mesh_parser.get_mesh(json_path)
        
        obj_path = os.path.join(output_dir, filename + '.obj')
        write_obj(vertices, triangles, obj_path)

    return

if __name__ == "__main__":
    main()