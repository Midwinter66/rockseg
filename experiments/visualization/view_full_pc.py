import laspy
import numpy as np
import open3d as o3d


def read_las(path):

    las = laspy.read(path)

    xyz = np.vstack((las.x, las.y, las.z)).T

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


pcd1 = read_las("D:\github_project\image_segment\DOM_Space_message_val\data\pointcloud3\Data\BlockB.laz")

pcd2 = read_las("D:\github_project\image_segment\DOM_Space_message_val\data\pointcloud3\Data\BlockY.laz")

# 降采样
pcd1 = pcd1.voxel_down_sample(0.05)

pcd2 = pcd2.voxel_down_sample(0.05)

# 上色
pcd1.paint_uniform_color([1, 0, 0])

pcd2.paint_uniform_color([0, 0, 1])

# 显示
o3d.visualization.draw_geometries(
    [pcd1, pcd2]
)