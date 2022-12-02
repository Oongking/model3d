#!/usr/bin/env python3

# Ros
import rospy
import tf
from tf.transformations import *

# msg & convert
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32

# 3D & Image
import cv2
import open3d as o3d

# Utility
import numpy as np
import time
import math
import copy

show_display = 1

FPS = 0

cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    # Create Trackbar to pick HSV value
def nothing(x):
    pass
cv2.createTrackbar("L - H1","Trackbars", 0,179,nothing)
cv2.createTrackbar("L - S1","Trackbars", 0,255,nothing)
cv2.createTrackbar("L - V1","Trackbars", 0,255,nothing)
cv2.createTrackbar("U - H1","Trackbars", 0,179,nothing)
cv2.createTrackbar("U - S1","Trackbars", 0,255,nothing)
cv2.createTrackbar("U - V1","Trackbars", 0,255,nothing)

bridge=CvBridge()
rgb_image=None
depth_image = None
updatetime = 0

# 608.7988891601562, 0.0, 637.463134765625, 0.0,
#  0.0, 608.79345703125, 364.19732666015625, 0.0, 
#  0.0, 0.0, 1.0, 0.0
""" Real Intrinsic"""
intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 610.89826648, 616.12187414, 647.16398913, 366.37689302)
Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])


    # Camera Matrix
matrix_coefficients = [[610.89826648,   0.,         647.16398913],
                        [  0.,         616.12187414, 366.37689302],
                        [  0.,           0.,           1.        ]]

# azure
distortion_coefficients = [[ 0.09865886, -0.11209954, -0.00087517,  0.00436045,  0.06666987]]

aruco_dict_type = cv2.aruco.DICT_6X6_250

matrix_coefficients = np.array(matrix_coefficients)
distortion_coefficients = np.array(distortion_coefficients)

def resize(img,scale_percent):
    # scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
        # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

# Image Function
def rgb_callback(data):
    global rgb_image
    global updatetime
    try:
       rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
       updatetime = time.time()
    except CvBridgeError as e:
        print(e)
    # print(rgb_image.shape)
    # cv2.imshow("t",rgb_image)
    # cv2.waitKey(2)
    # print("rgb_ok")

def depth_callback(data):
    global depth_image
    try:
       depth_image = bridge.imgmsg_to_cv2(data, "32FC1")
    except CvBridgeError as e:
        print(e)

def buildPCD(color_image,depth_image):
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=100.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # o3d.visualization.draw_geometries([downpcd,Realcoor])
    # return downpcd
    return pcd

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

def fixbox(rot,trans,z_offset) :
    # Before rotate to canon
    y = 0.3
    x = 0.3
    z = 0.3  
    
    fix_box = np.array([
    [-x/2,-y/2,z_offset],
    [-x/2, y/2,z_offset],
    [x/2, y/2,z_offset],
    [x/2,-y/2,z_offset],

    [-x/2,-y/2,z+z_offset],
    [-x/2, y/2,z+z_offset],
    [x/2,-y/2,z+z_offset],
    [x/2, y/2,z+z_offset]
    ])

    fixbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(fix_box))
    fixbox.rotate(rot,(0,0,0))
    fixbox.translate(np.asarray(trans,dtype=np.float64),relative=True)
    fixbox.color = (0, 0, 0)

    return fixbox 

class sphere:
    def __init__(self, center, color,r):
        self.pcd = o3d.geometry.TriangleMesh.create_sphere(radius = r)
        self.pcd.compute_vertex_normals()
        self.pcd.translate((center[0], center[1], center[2]), relative=False)
        self.pcd.paint_uniform_color(color)

def normalvector(base, cross1, cross2):
    vector1 = np.subtract(cross1,base)
    vector2 = np.subtract(cross2,base)
    normalVec = np.cross(vector1,vector2)
    uNormalVec = normalVec/np.linalg.norm(normalVec)
    print("Normal : ",UNormalVec)
    return UNormalVec

def depfromcolor(mask,depth_image):
    _,alpha = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    alpha[alpha == 255] = 1
    depth_image = depth_image*alpha
    return depth_image

if __name__ == '__main__':
    rospy.init_node('azure_model', anonymous=True)
    rospy.Subscriber("/rgb/image_raw", Image, rgb_callback)
    rospy.Subscriber("/depth_to_rgb/image_raw",Image,depth_callback)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    arucoDictA3 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    arucoParams = cv2.aruco.DetectorParameters_create() 
    board2x2=create_board(markerSize=0.03075,dict=arucoDict)
    pcd_merged = o3d.geometry.PointCloud()

    board75 = cv2.aruco.GridBoard_create(7, 5, 0.0315, 0.0081, arucoDict)

    # For Azure
    percentage_offset = 0.03
    ARsize = 0.02155 + (0.02155*percentage_offset)
    ARgabsize = 0.0058 + (0.0058*percentage_offset)

    boardA3 = cv2.aruco.GridBoard_create(14, 10, ARsize, ARgabsize, arucoDictA3)

    #A4
    # offset_x=0.5*((0.0315*7)+(0.0081*(7-1)))
    # offset_y=0.5*((0.0315*5)+(0.0081*(5-1)))
    #A3
    offset_x=0.5*((ARsize*14)+(ARgabsize*(14-1)))
    offset_y=0.5*((ARsize*10)+(ARgabsize*(10-1)))
    
    num = 0
    small_Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])

    print("=================================================================================")
    print("\n:: Key command ::\n\tc : Capturing & Merge Model\n\tr : Set color\n\tp : Preview pcd_merged\n\ts : Save\n\te : Shutdown the Server")

    lower_color1 = np.array([0,0,0])
    upper_color1 = np.array([179,255,255])
    
    while True:
        if (depth_image is not None) and (rgb_image is not None):
            start_time = time.time()
            color_image = copy.deepcopy(rgb_image)
            raw_depth_image = depth_image.astype(np.uint16)

            l_h1 = cv2.getTrackbarPos("L - H1","Trackbars")
            l_s1 = cv2.getTrackbarPos("L - S1","Trackbars")
            l_v1 = cv2.getTrackbarPos("L - V1","Trackbars")
            u_h1 = cv2.getTrackbarPos("U - H1","Trackbars")
            u_s1 = cv2.getTrackbarPos("U - S1","Trackbars")
            u_v1 = cv2.getTrackbarPos("U - V1","Trackbars")
            
            blurred_frame = cv2.GaussianBlur(color_image, (21,21),0)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_color1, upper_color1)

            maskC = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            im_thresh_color = cv2.bitwise_and(color_image, maskC)
            cv2.putText(color_image, f'Merge : {num}', (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),3)
            cv2.putText(color_image, f'Merge : {num}', (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,150),2)

            numpy_horizontal = np.hstack((resize(color_image,50), resize(im_thresh_color,50)))

            cv2.imshow('All', numpy_horizontal)

            T_depth_image = depfromcolor(mask,raw_depth_image)

            waitkeyboard = cv2.waitKey(1)

            if waitkeyboard & 0xFF==ord('c'):
                
                print("\n:: Capturing & Merge Model")
                rvec=None
                tvec=None

                pcd = buildPCD(rgb_image,T_depth_image)

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
                
                if show_display:
                        if rvec is not None and tvec is not None:
                            cv2.aruco.drawAxis( rgb_image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.08 )
                            cv2.imshow('rgb_image', rgb_image)

                transformation_matrix[ :3, 3] = np.asarray(tvec).transpose()

                z_offset = 0.003
                box = fixbox(transformation_matrix[:3, :3],transformation_matrix[ :3, 3],z_offset)

                centercoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01,[0,0,0])
                centercoor.rotate(transformation_matrix[:3, :3],(0,0,0))
                centercoor.translate(np.asarray(transformation_matrix[ :3, 3],dtype=np.float64),relative=False)

                if show_display:
                    sph = sphere(np.asarray(transformation_matrix[ :3, 3]),(0.8,0,0),0.001)
                    o3d.visualization.draw_geometries([Realcoor,centercoor,pcd,box,sph.pcd])

                pcd_crop = pcd.crop(box)
                if not pcd_crop.is_empty():
                    pcd_crop.translate(np.asarray(-transformation_matrix[ :3, 3],dtype=np.float64),relative=True)
                    pcd_crop.rotate(transformation_matrix[:3, :3].transpose(),(0,0,0))
                    pcd_crop.translate([0,0,-z_offset],relative=True)

                    # if show_display:
                    #     o3d.visualization.draw_geometries([Realcoor,centercoor,pcd,box,pcd_crop])
                    pcd_crop_filted, ind = pcd_crop.remove_statistical_outlier(nb_neighbors=100,std_ratio=0.5)
                    pcd_merged += pcd_crop_filted
                    num +=1
                    print(" :: Complete Capturing :: ")
                else:
                    rospy.loginfo(" No Point Cloud In Workspace ")

            if waitkeyboard & 0xFF==ord('r'):
                lower_color1 = np.array([l_h1,l_s1,l_v1])
                upper_color1 = np.array([u_h1,u_s1,u_v1])            

            if waitkeyboard & 0xFF==ord('s'):
                o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/Model.pcd", pcd_merged)
                # Raw Data
                # o3d.io.write_point_cloud(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/Model.pcd", pcd)
                # cv2.imwrite(f"/home/oongking/RobotArm_ws/src/model3d/script/buildModel/Data/Model.png",rgb_image)
                o3d.visualization.draw_geometries([small_Realcoor,pcd_merged])
                rospy.loginfo("complete")

            if waitkeyboard & 0xFF==ord('p'):
                rospy.loginfo(" Preview pcd_merged ")
                o3d.visualization.draw_geometries([small_Realcoor,pcd_merged])

            if waitkeyboard & 0xFF==ord('e'):
                break


    # device.close()
    cv2.destroyAllWindows()

