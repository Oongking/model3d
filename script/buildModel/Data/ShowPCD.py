import numpy as np
import open3d as o3d
import copy

pcd = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCoverModel.pcd")
pcdb = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/FanCoverBottom.pcd")
pcdt = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/FanCoverTop.pcd")
# pcd_down = pcd.voxel_down_sample(voxel_size)
mesh = o3d.io.read_triangle_mesh("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCover_poly_0_0006m.stl")



Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])

o3d.visualization.draw_geometries([Realcoor,pcd])
o3d.visualization.draw_geometries([Realcoor,mesh])
# o3d.visualization.draw_geometries([Realcoor,pcdt])

