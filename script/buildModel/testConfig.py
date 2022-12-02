
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
import copy

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])

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

            o3d.visualization.draw_geometries([Realcoor,pcd])
            print("Complete load PCD file : "+str(x))
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=2000,
                                                    std_ratio=0.5)
            o3d.visualization.draw_geometries([Realcoor,cl])
            
            self.pcdsSource.append(cl)
            bufferPcd = cl.voxel_down_sample(voxel_size=voxel_size)
            bufferPcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=50))
            o3d.visualization.draw_geometries([Realcoor,bufferPcd])

            self.pcds.append(bufferPcd)
        print("----Complete Loading File----")

# voxel_size = 0.001
# Data = Data("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\"+"")
# Data.LoadPCD("data_fragment", 0, 0)

pcd = o3d.io.read_point_cloud("D:\Oongking\iRAP\Robot arm\program\ZED2\\buildModel\data\\fan\\fan_fragment_mergepart.pcd")

o3d.visualization.draw_geometries([Realcoor,pcd])