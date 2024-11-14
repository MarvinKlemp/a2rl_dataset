from nuscenes.nuscenes import NuScenes
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt


if __name__ == "__main__":
   nusc = NuScenes(version='v1.0-mini', dataroot='./data/v1.0-mini', verbose=True)
   sample_token = 'b63edf8e84f24e71bd9a252c5e86cb58'
   sample_record = nusc.get('sample', sample_token)
   lidar_token = sample_record['data']['LIDAR_TOP']
   pointcloud_path = nusc.get_sample_data_path(lidar_token)
   _, boxes, _ = nusc.get_sample_data(lidar_token)
   b = boxes[0]
   b.center = [195.3,-6.76, 3.16]
   
   # Create a figure with black background
   fig, ax = plt.subplots(figsize=(18, 9))
   ax.set_facecolor('black')  # Set background color to black
   
   # Render the point cloud
   LidarPointCloud.from_file(pointcloud_path).render_height(ax, x_lim=(-200, 200), y_lim=(-200, 200))
   
   # Render the boxes
   for box in boxes:
      box.render(ax, linewidth=1)
      corners = view_points(box.corners(), np.eye(3), False)[:2, :]
      ax.set_xlim([np.min(corners[0, :] - 2), np.max(corners[0, :]) + 2])
      ax.set_ylim([np.min(corners[1, :] - 2), np.max(corners[1, :]) + 2])
      ax.set_aspect('equal')
   ax.grid(False)
   ax.xaxis.set_visible(False)
   ax.yaxis.set_visible(False)

   plt.savefig('200m.pdf', format='pdf', bbox_inches='tight')
   plt.show()
