
import open3d as o3d
import numpy as np

# Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
# pcd = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/MergeFan.pcd")
# o3d.visualization.draw_geometries([pcd,Realcoor])
# bufferPcd = pcd.voxel_down_sample(voxel_size=0.0003)
# o3d.visualization.draw_geometries([bufferPcd,Realcoor])

# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(bufferPcd, 0.0012)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh,Realcoor], mesh_show_back_face=True)


Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
pcd = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCoverModel.pcd")
o3d.visualization.draw_geometries([pcd,Realcoor])
bufferPcd = pcd.voxel_down_sample(voxel_size=0.0003)
o3d.visualization.draw_geometries([bufferPcd,Realcoor])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(bufferPcd, 0.0006)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh,Realcoor], mesh_show_back_face=True)

# o3d.io.write_triangle_mesh(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCover_poly_0_0006m.stl", mesh)

# bufferPcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=50))
# radii = [1, 5, 10, 20]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     bufferPcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([bufferPcd, rec_mesh],mesh_show_back_face=True)