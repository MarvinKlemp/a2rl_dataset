from nuscenes.nuscenes import NuScenes
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt


if __name__ == "__main__":
   nusc = NuScenes(version='v1.0-mini', dataroot='./data/v1.0-mini', verbose=True)
   sample_token = nusc.sample[0]['token']
   sample_record = nusc.get('sample', sample_token)
   lidar_token = sample_record['data']['LIDAR_VEHICLE']
   pointcloud_path = nusc.get_sample_data_path(lidar_token)
   _, boxes, _ = nusc.get_sample_data(lidar_token)
   
   fig, ax = plt.subplots(figsize=(18, 9))
   LidarPointCloud.from_file(pointcloud_path).render_height(ax, x_lim=(-200, 200), y_lim=(-200, 200))
   for box in boxes:
      box.render(ax)
      corners = view_points(box.corners(), np.eye(3), False)[:2, :]
      # ax.set_xlim([np.min(corners[0, :] -2), np.max(corners[0, :]) + 2])
      # ax.set_ylim([np.min(corners[1, :] -2), np.max(corners[1, :]) + 2])
      ax.set_aspect('equal')
   plt.show()

