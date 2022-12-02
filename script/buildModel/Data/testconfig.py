import numpy as np
import open3d as o3d
import copy

def Cluster(pcd,voxel):
    pcds = []
    labels = np.array(pcd.cluster_dbscan(eps= 2*voxel, min_points=5, print_progress=True))
    
    max_label = labels.max()
    for i in range(0,max_label+1):
        pcdcen = pcd.select_by_index(np.argwhere(labels==i))
        
        pcds.append(pcdcen)
    
    pcds = list(pcds)

    return pcds

# voxel_size = 0.0004

pcd = o3d.io.read_point_cloud("/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/MergeFan.pcd")

# pcd_down = pcd.voxel_down_sample(voxel_size)


pcds = Cluster(pcd,0.0004)
cl = pcds[0]+pcds[1]+pcds[2]
# o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/predata/MergeFan.pcd", cl)
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=5)

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
cl.translate([-0.15,0,0],relative=True)
# show = copy.deepcopy()
o3d.visualization.draw_geometries([Realcoor,cl,pcd])

