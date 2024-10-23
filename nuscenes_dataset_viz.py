import pathlib
import open3d
import numpy as np
import math
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0]) + np.pi / 2

    return yaw


vis = open3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.zeros(3)

pred_labels = None

DATA_SET_PATH = pathlib.Path("v1.0-mini")

nusc = NuScenes(version='v1.0-mini', dataroot=DATA_SET_PATH, verbose=True)

frame_index = 0

sample_tokens = []

for scene in nusc.scene:
    first_sample_token = scene['first_sample_token']
    sample_tokens.append(first_sample_token)
    sample = nusc.get('sample', first_sample_token)
    while sample['next'] != '':
        sample_token = sample['next']
        sample_tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)

print(len(sample_tokens))

def load_next_frame(vis):
    global frame_index
    print("Loading frame", frame_index)

    vis.clear_geometries()
    view_control = vis.get_view_control()

    gt_o3d_boxes = []

    sample_token = sample_tokens[frame_index]
    my_sample = nusc.get('sample', sample_token)
    my_scene = nusc.get('scene', my_sample['scene_token'])
    print('Scene name: ', my_scene['name'], my_scene['description'])

    for anno_token in my_sample['anns']:
        anno = nusc.get('sample_annotation', anno_token)
        center = anno['translation']
        lwh = anno['size']

        quat = Quaternion(anno['rotation'])
        yaw = quaternion_yaw(quat)
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw])
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
        box3d.color = [0, 0, 1.0]
        gt_o3d_boxes.append(box3d)

    close_range = False
    max_dist = 0
    for gt_box in gt_o3d_boxes:
        points = gt_box.get_center()
        distance = math.sqrt(math.pow(points[0], 2) + math.pow(points[1], 2))
        if distance > 60:
            close_range = False
        if distance > max_dist:
            max_dist = distance

    # if close_range:
    #     frame_index += 1
    #     load_next_frame(vis)
    #     return

    print("Distance to GT Box:", max_dist)

    # Load point cloud
    
    sample_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
    sample_data_filename = DATA_SET_PATH / sample_data['filename']
    points = np.fromfile(str(sample_data_filename), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)

    for box in gt_o3d_boxes:
        vis.add_geometry(box)

    if len(gt_o3d_boxes) > 0:
        view_control.set_lookat(gt_o3d_boxes[0].get_center())
    else:
        view_control.set_lookat(np.array([0.0, 0.0, 0.0]))

    view_control.set_zoom(0.125)

    frame_index += 1


def load_prev_frame(vis):
    global frame_index
    if frame_index >= 2:
        frame_index -= 2
        return load_next_frame(vis)
    else:
        frame_index = 0
        return load_next_frame(vis)


def close_vis(vis):
    vis.close()
    vis.destroy_window()


key_to_callback = {}
key_to_callback[ord("N")] = load_next_frame
key_to_callback[ord("B")] = load_prev_frame
key_to_callback[ord("X")] = close_vis

load_next_frame(vis)

for key, val in key_to_callback.items():
    vis.register_key_callback(key, val)

vis.run()
