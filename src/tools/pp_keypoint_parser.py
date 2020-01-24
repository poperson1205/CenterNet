import xml.etree.ElementTree as ET
import numpy as np

def get_keypoints(path):
    points = []

    # Parse pp file
    tree = ET.parse(path)
    root = tree.getroot()
    for point_node in root.findall('point'):
        x = float(point_node.get('x'))
        y = float(point_node.get('y'))
        z = float(point_node.get('z'))
        points.append((x, y, z))

    return np.array(points)