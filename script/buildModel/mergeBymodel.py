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
            # bufferPcd = cl.voxel_down_sample(voxel_size=voxel_size)
            # bufferPcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=50))
            self.pcds.append(cl)
        print("----Complete Loading File----")

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(500,[0,0,0])
    o3d.visualization.draw_geometries([source_temp, target_temp,coor],
                                    zoom=0.4559,
                                    front=[0.6452, -0.3036, -0.7011],
                                    lookat=[1.9892, 2.0208, 1.8945],
                                    up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=500))
    # o3d.visualization.draw_geometries([pcd_down],window_name= "ssss")

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,pcd_model,pcd_input):
    print(":: Load two point clouds and disturb initial pose.")
    source = pcd_model
    target = pcd_input
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.8))
    return result

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])

if __name__ == "__main__":
    voxel_size = 0.001  
    Data = Data("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\"+"fan\\")
    Data.LoadPCD("fan_fragment_b", 0, 8)
    # pcd_model = o3d.io.read_point_cloud("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\fan\\fan_fragment_mergepart2.pcd")
    # pcd_input = o3d.io.read_point_cloud("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\fan\\fan_fragment_mergepart1.pcd")
    pcd_combined = o3d.geometry.PointCloud()
    for i in range(len(Data.pcds)-1):
        i+=1
        print(f"########## round {i} ##########")
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
            voxel_size,Data.pcds[i],Data.pcds[0])
        
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)

        print(result_ransac)
        print(result_ransac.transformation)
        
        Data.pcdsSource[i].transform(result_ransac.transformation)
        pcd_combined += Data.pcdsSource[i]

                        
    # pcd_model.transform(back_to_Origin)
    pcd_combined += Data.pcds[0]
    o3d.visualization.draw_geometries([pcd_combined,Realcoor])

    o3d.io.write_point_cloud("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\fan\\fan_fragment_b_mergepart.pcd", pcd_combined)
    # print("\n\n:: Back To Origin : Transformation Matrix ")
    # print(back_to_Origin)


