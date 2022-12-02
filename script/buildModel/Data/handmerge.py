import numpy as np
import open3d as o3d


def fixbox(rot = [[1,0,0],[0,1,0],[0,0,1]],trans = [0,0,0], z_offset = 0) :
    # Before rotate to canon
    y = 0.3
    x = 0.065
    z = 0.3  
    x_offset = 0.017
    fix_box = np.array([
    [-x/2,-y/2,z_offset],
    [-x/2, y/2,z_offset],
    [x/2-x_offset, y/2,z_offset],
    [x/2-x_offset,-y/2,z_offset],

    [-x/2,-y/2,z+z_offset],
    [-x/2, y/2,z+z_offset],
    [x/2-x_offset,-y/2,z+z_offset],
    [x/2-x_offset, y/2,z+z_offset]
    ])

    fixbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(fix_box))
    fixbox.rotate(rot,(0,0,0))
    fixbox.translate(np.asarray(trans,dtype=np.float64),relative=True)
    fixbox.color = (0, 0, 0)

    return fixbox 

def Cluster(pcd,voxel):
    pcds = []
    labels = np.array(pcd.cluster_dbscan(eps= 2*voxel, min_points=5, print_progress=False))
    
    max_label = labels.max()
    for i in range(0,max_label+1):
        pcdcen = pcd.select_by_index(np.argwhere(labels==i))
        
        pcds.append(pcdcen)
    
    pcds = list(pcds)

    return pcds

pcdtop = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCoverTop_pre.pcd")
# pcdtop.paint_uniform_color([1, 0.706, 0])
pcdtop.translate((-0.0041,-0.001,0),relative=True)
Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
# o3d.visualization.draw_geometries([Realcoor,pcdtop])
pcdbot = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCoverBottom_pre.pcd")
o3d.visualization.draw_geometries([Realcoor,pcdbot,pcdtop])
pcdbot_cl, ind = pcdbot.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.8)
pcdtop_cl, ind = pcdtop.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.8)
o3d.visualization.draw_geometries([Realcoor,pcdbot_cl,pcdtop_cl])

pcd_combined = pcdbot_cl + pcdtop_cl
o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/FanCoverModel.pcd", pcd_combined)

# def Rz(degree):
#     rad = degree * np.pi/180
#     Rz = [  [np.cos(rad),-np.sin(rad),0],
#             [np.sin(rad),np.cos(rad),0],
#             [0,0,1]]
#     return Rz

# pcdbot.rotate(Rz(-45),(0,0,0))
# # o3d.visualization.draw_geometries([Realcoor,pcdbot,pcdtop])

# def Ry(degree):
#     rad = degree * np.pi/180
#     Ry = [  [np.cos(rad),0,np.sin(rad)],
#             [0,1,0],
#             [-np.sin(rad),0,np.cos(rad)]]
#     return Ry
# pcdbot.rotate(Ry(0.8),(0,0,0))
# # o3d.visualization.draw_geometries([Realcoor,pcdbot,pcdtop])

# def Rx(degree):
#     rad = degree * np.pi/180
#     Rx = [  [1,0,0],
#             [0,np.cos(rad),-np.sin(rad)],
#             [0,np.sin(rad),np.cos(rad)]]
#     return Rx
# pcdbot.rotate(Rx(0.5),(0,0,0))
# pcdbot.rotate(Rz(41+51.4285714*5),(0,0,0))
# pcdbot.translate([-0.002,0,0],relative=True)
# pcdbot.translate([0,0,-0.003],relative=True)
# o3d.visualization.draw_geometries([Realcoor,pcdbot,pcdtop])

# box = fixbox(z_offset = 0.024)
# o3d.visualization.draw_geometries([Realcoor,pcdbot])
# pcdbot.rotate([[1,0,0],
#                 [0,-1,0],
#                 [0,0,-1]],(0,0,0))
# pcdbot.translate(pcdbot.get_center(),relative=True)/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/pcdbot.pcd
#     rad = degree * np.pi/180
#     Rz = [  [np.cos(rad),-np.sin(rad),0],
#             [np.sin(rad),np.cos(rad),0],
#             [0,0,1]]
#     return Rz
# pcdbot.rotate(Rz(-4),(0,0,0))
# o3d.visualization.draw_geometries([Realcoor,pcdbot,pcdtop])

# o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/pcdbot.pcd", pcdbot)
# o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/pcdtop.pcd", pcdtop)
# o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/MergeFan.pcd", pcdtop+pcdbot)

# pcd_crop = pcd.crop(box)
# o3d.visualization.draw_geometries([Realcoor,pcd_crop,box])

# voxel = 0.0006
# pcd_down = pcd_crop.voxel_down_sample(voxel)
# o3d.visualization.draw_geometries([Realcoor,pcd_down,box])

# cl, ind = pcd_crop.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.8)
# o3d.visualization.draw_geometries([Realcoor,cl,box])

# pcds = Cluster(pcd_down,voxel)

# o3d.visualization.draw_geometries([Realcoor,pcds[0]])


