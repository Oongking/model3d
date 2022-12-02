import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
import copy

class Data:
    def __init__(self, path):
        self.pcdsSource = []
        self.pcds = []
        self.Croppcds =[]
        self.PCD = path
        print(f"Create DATA with {path}")

    def LoadPCD(self,name,snum,fnum):
        print(f"Start Loading File....  \n'{name}'")

        for x in range(snum,fnum+1):
            filename_pcd = self.PCD+ name +str(x)+".pcd"
            pcd = o3d.io.read_point_cloud(filename_pcd)
            print("Complete load PCD file : "+str(x))
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=2000,
                                                    std_ratio=0.5)
            self.pcdsSource.append(cl)
            bufferPcd = cl.voxel_down_sample(voxel_size=voxel_size)
            bufferPcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=50))
            self.pcds.append(bufferPcd)
        print("----Complete Loading File----")
    
def displayPCD(pcd,name):
    coor_main = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])
    o3d.visualization.draw_geometries([pcd,coor_main])

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                    max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print(f"Round left : {source_id}/{n_pcds} ")
            print(f"Sub round left : {target_id}/({source_id + 1}->{n_pcds}) ")
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=True))
    return pose_graph

def filter_pcd(pcd):
        print("Start Filtering PCD....")
        pcd_filtered,ind = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=3.0)
        print("----Complete Filtering PCD----")
        return pcd_filtered

#keiba voxel = 1
voxel_size = 0.001
Data = Data("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\"+"fan\\")
Data.LoadPCD("fan_fragment_b", 0, 8)

print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15/10
max_correspondence_distance_fine = voxel_size * 1.5/10
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(Data.pcds, 
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

print("Transform points and display")
for point_id in range(len(Data.pcds)):
    print(pose_graph.nodes[point_id].pose)
    # Data.pcds[point_id].transform(pose_graph.nodes[point_id].pose)

# displayPCD(Data.pcds[0],"Data.pcds")

pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(Data.pcds)):
    Data.pcdsSource[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += Data.pcdsSource[point_id]

# pcd_combined = filter_pcd(pcd_combined)
displayPCD(pcd_combined,"pcd_combined")
# savename = ""
# o3d.io.write_point_cloud("D:/Oongking/iRAP/Robot arm/program/ZED2/buildModel/data/fan/fan_fragment_b_mergepart1.pcd", pcd_combined)
print("--------Complete Saving--------")
