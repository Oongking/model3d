#!/usr/bin/env python

# Ros
import rospy
import rosnode
import tf
from tf.transformations import *

# msg & convert
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32

# Zivid
import dynamic_reconfigure.client
from zivid_camera.srv import *

# Image
import cv2
import cv2.aruco as aruco

# 3D
import open3d as o3d

# Utility
import getch
import numpy as np
import time
import copy


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

matrix_coefficients = [[ 1782.476318359375, 0.0, 965.43896484375], [ 0.0, 1784.1812744140625, 590.5164184570312], [ 0, 0, 1]]
distortion_coefficients = [[-0.08575305342674255, 0.1142171174287796, 0.00030625637737102807, -0.0007428471581079066, -0.048006460070610046]]
matrix_coefficients = np.array(matrix_coefficients)
distortion_coefficients = np.array(distortion_coefficients)

zivid_matrix_coefficients = np.array([[1775.45651052, 0.0, 971.44183783],
                                [0.0, 1776.00990344, 593.88802681],
                                [ 0.0, 0.0, 1.0 ]])
zivid_distortion_coefficients = np.array([[-0.08439001,  0.05578754,  0.00092648,  0.00037382,  0.11682607]])
# zivid_intrinsic = o3d.camera.PinholeCameraIntrinsic(1944, 1200, 1775.45651052, 1776.00990344, 971.44183783, 593.88802681)
# K: [1782.476318359375, 0.0, 965.43896484375, 0.0, 1784.1812744140625, 590.5164184570312, 0.0, 0.0, 1.0]

zivid_intrinsic = o3d.camera.PinholeCameraIntrinsic(1944, 1200, 1782.476318359375, 1784.1812744140625, 965.43896484375, 590.5164184570312)

def UndistortedImage(image):

    image_dist = cv2.undistort(image, matrix_coefficients, distortion_coefficients, None, matrix_coefficients)
    return image_dist

bridge=CvBridge()

class sphere:
    def __init__(self, center, color):
        self.pcd = o3d.geometry.TriangleMesh.create_sphere(radius = 0.003)
        self.pcd.compute_vertex_normals()
        self.pcd.translate((center[0], center[1], center[2]), relative=False)
        self.pcd.paint_uniform_color(color)

class zivid_cam:
    def __init__(self):
        rospy.init_node("Zivid_model", anonymous=True)

        rospy.loginfo(":: Starting Zivid_model ::")

        rospy.wait_for_service("/zivid_camera/capture", 30.0)

        rospy.Subscriber("/zivid_camera/points/xyzrgba", PointCloud2, self.points_callback)
        rospy.Subscriber("/zivid_camera/color/image_color", Image, self.rgb_callback)
        rospy.Subscriber("/zivid_camera/depth/image", Image, self.depth_callback)
        self.received_ros_cloud = None
        self.pcd = None
        self.rgb_image = None
        self.depth_image = None
        self.capture_service = rospy.ServiceProxy("/zivid_camera/capture", Capture)

        rospy.loginfo("Enabling the reflection filter")
        settings_client = dynamic_reconfigure.client.Client("/zivid_camera/settings/")
        settings_config = {"processing_filters_reflection_removal_enabled": True}
        settings_client.update_configuration(settings_config)

        rospy.loginfo("Enabling and configure the first acquisition")
        acquisition_0_client = dynamic_reconfigure.client.Client(
            "/zivid_camera/settings/acquisition_0"
        )
        acquisition_0_config = {
            "enabled": True,
            "aperture": 5.66,
            "brightness": 1.8,
            "exposure_time": 40000,
            "gain": 1,
        }
        acquisition_0_client.update_configuration(acquisition_0_config)

    def capture(self):
        rospy.loginfo("Calling capture service")
        self.capture_service()

        while 1:
            # print("wait for input")
            if (self.rgb_image is not None) and (self.pcd is not None):
                print("Break")
                break
        
        rgb_image = copy.deepcopy(self.rgb_image)
        pcd = copy.deepcopy(self.pcd)

        self.rgb_image = None
        self.pcd = None
        return rgb_image, pcd
    
    def testMatrix(self):
        rospy.loginfo("Calling capture service")
        self.rgb_image = None
        self.depth_image = None
        self.pcd = None
        self.capture_service()

        while 1:
            # print("wait for input")
            if (self.rgb_image is not None) and (self.depth_image is not None) and (self.pcd is not None):
                break
        
        un_rgb_image = UndistortedImage(self.rgb_image)
        un_depth_image = UndistortedImage(self.depth_image)
        depth = o3d.geometry.Image(un_depth_image)
        color = o3d.geometry.Image(cv2.cvtColor(un_rgb_image, cv2.COLOR_BGR2RGB))
        
        # depth = o3d.geometry.Image(self.depth_image)
        # color = o3d.geometry.Image(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)
        make_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, zivid_intrinsic)

        pcd = copy.deepcopy(self.pcd)
        rgb_image = copy.deepcopy(self.rgb_image)
        depth_image = copy.deepcopy(self.depth_image)
        self.pcd = None
        self.rgb_image = None
        self.depth_image = None
        return pcd,rgb_image, depth_image,make_pcd
        

    def points_callback(self, data):
        rospy.loginfo("PointCloud received")
        self.received_ros_cloud=data
        self.pcd = self.convertCloudFromRosToOpen3d()

    def rgb_callback(self, received_image):
        rospy.loginfo("Image received")
        try:
            self.rgb_image = bridge.imgmsg_to_cv2(received_image, "bgr8")
        except CvBridgeError as e:
                print(e)
    
    def depth_callback(self,data):
        try:
            T_depth_image = bridge.imgmsg_to_cv2(data, "32FC1")
            self.depth_image = T_depth_image.astype(np.float32)

        except CvBridgeError as e:
            print(e)

    def convertCloudFromRosToOpen3d(self):
        print(":: convertCloudFromRosToOpen3d ::")
        open3d_cloud = o3d.geometry.PointCloud()
        print("Data lenght : ",len(self.received_ros_cloud.data))
        if self.received_ros_cloud is not None:
            # Get cloud data from ros_cloud
            # print("FFFFFFFFFFTESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")

            field_names=[field.name for field in self.received_ros_cloud.fields]
            # print("field name : ",field_names)
            cloud_data = list(pc2.read_points(self.received_ros_cloud, skip_nans=False, field_names = field_names))

            
            # Check empty
            
            if len(cloud_data)==0:
                print("Converting an empty cloud")
                return None  

            # Set open3d_cloud

            if "rgba" in field_names:
                IDX_RGB_IN_FIELD=3 # x, y, z, rgb
                
                # Get xyz
                xyz = [(x,y,z) for x,y,z,rgba in cloud_data ] # (why cannot put this line below rgb?)
                print("lenght of xyz : ",len(xyz))
                # Get rgb
                # Check whether int or float
                if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                    rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
                else:
                    rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

                # combine
                open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
                open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
            else:
                xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
                open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

            print("Pointclound O3D : ", open3d_cloud)
            # return
            return open3d_cloud
        else:
            return None

def create_board(markerSize,dict):
    marker_dis=0.125
    marker_offset=marker_dis/2

    sub_b1_marker1=np.array([[-(marker_offset+markerSize),(marker_offset+markerSize),0],
                           [-marker_offset               ,(marker_offset+markerSize),0],
                           [-marker_offset               ,marker_offset,0],
                           [-(marker_offset+markerSize),marker_offset,0]],dtype=np.float32)
    
    sub_b1_marker2=np.array([[(marker_offset),(marker_offset+markerSize),0],
                           [marker_offset+markerSize   ,(marker_offset+markerSize),0],
                           [marker_offset+markerSize   ,marker_offset,0],
                           [(marker_offset)  ,marker_offset,0]],dtype=np.float32)
    
       
    sub_b1_marker3=np.array([[(marker_offset),-(marker_offset),0],
                           [marker_offset+markerSize   ,-(marker_offset),0],
                           [marker_offset+markerSize   ,-(marker_offset+markerSize),0],
                           [(marker_offset)  ,-(marker_offset+markerSize),0]],dtype=np.float32)

       
    sub_b1_marker4=np.array([[-(marker_offset+markerSize),-(marker_offset),0],
                           [-marker_offset               ,-(marker_offset),0],
                           [-marker_offset               ,-(marker_offset+markerSize),0],
                           [-(marker_offset+markerSize)  ,-(marker_offset+markerSize),0]],dtype=np.float32)
    board_corners=[]
    board_corners.append(sub_b1_marker1)
    board_corners.append(sub_b1_marker2)
    board_corners.append(sub_b1_marker3)
    board_corners.append(sub_b1_marker4)

    board_ids = np.array([[0],[1],[3],[2]], dtype=np.int32)

    board = cv2.aruco.Board_create(board_corners,dict, board_ids )

    return board

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])

if __name__ == '__main__':
    cam = zivid_cam()
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    arucoDictA3 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    
    arucoParams = cv2.aruco.DetectorParameters_create() 
    board2x2=create_board(markerSize=0.03075,dict=arucoDict)
    boardA4 = cv2.aruco.GridBoard_create(7, 5, 0.0315, 0.0081, arucoDict)
    boardA3 = cv2.aruco.GridBoard_create(14, 10, 0.02155, 0.0058, arucoDictA3)
    test_coordinate = []

    #A4
    # offset_x=0.5*((0.0315*7)+(0.0081*(7-1)))
    # offset_y=0.5*((0.0315*5)+(0.0081*(5-1)))
    #A3
    offset_x=0.5*((0.02155*14)+(0.0058*(14-1)))
    offset_y=0.5*((0.02155*10)+(0.0058*(10-1)))

    while True:
        print("=================================================================================")
        print("\n:: Key command ::\n\tc : Capturing\n\tp : Preview \n\te : Shutdown the Server")
        key = getch.getch().lower()
        print("key : ",key)
        if key == 't':
            pcd,rgb_image, depth_image,make_pcd = cam.testMatrix()

            
            # while not rospy.is_shutdown():
            #     waitkeyboard = cv2.waitKey(1)
            #     cv2.imshow('rgb_image', rgb_image)
            #     cv2.imshow('un_rgb_image', un_rgb_image)
            #     cv2.imshow('depth_image', depth_image)
            #     cv2.imshow('un_depth_image', un_depth_image)

            #     if waitkeyboard & 0xFF==ord('q'):
            #         print("===== End =====")
            #         break

            # make_pcd.translate([1,0,0])

            print(f"make_pcd : {make_pcd}")
            print(f"pcd : {pcd}")
            pcd.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries([Realcoor,pcd,make_pcd])
        if key == 'c':
            print("\n:: Capturing ::")
            rvec=None
            tvec=None
            
            rgb_image, pcd = cam.capture()
            # cv2.imshow('rgb_image', rgb_image)
            # o3d.visualization.draw_geometries([Realcoor,pcd])

            (corners, ids,rejected)= cv2.aruco.detectMarkers(rgb_image,arucoDictA3,parameters=arucoParams)

            if len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(rgb_image,corners,ids)
                _,rvec,tvec = cv2.aruco.estimatePoseBoard( corners, ids, boardA3, matrix_coefficients, distortion_coefficients,rvec,tvec)
            
            transformation_matrix = np.array([  [1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]],
                                                dtype=float)

            transformation_matrix[:3, :3], _ = cv2.Rodrigues(rvec)

            q= tf.transformations.quaternion_from_matrix(transformation_matrix)
            vec= [offset_x,offset_y,0,0]
            global_offset=quaternion_multiply(quaternion_multiply(q, vec),quaternion_conjugate(q))
            tvec[0]=tvec[0]+global_offset[0]
            tvec[1]=tvec[1]+global_offset[1]
            tvec[2]=tvec[2]+global_offset[2]

            transformation_matrix[ 3, :3] = np.asarray(tvec).transpose()

            # centercoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
            # centercoor.rotate(transformation_matrix[:3, :3],(0,0,0))
            # centercoor.translate(np.asarray(transformation_matrix[ 3, :3],dtype=np.float64),relative=False)
            # test_coordinate.append(centercoor)
            center_sphere = sphere(np.asarray(transformation_matrix[ 3, :3]),(1,0.2,0.2))
            test_coordinate.append(center_sphere.pcd)

            
            # cv2.aruco.drawAxis( rgb_image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.08 )
            # while 1:
            #         cv2.imshow('rgb_image', rgb_image)
            #         if cv2.waitKey(1) & 0xFF==ord('q'):
            #             break
            


        elif key == 'p':
            rospy.loginfo(" Preview ")
            o3d.visualization.draw_geometries([Realcoor,pcd]+test_coordinate)

        elif key == 'e':
            break

        else:
            print("Wrong Key")
    print("\n------ Shutdown the Server ------ ")