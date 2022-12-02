#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import open3d as o3d
import numpy as np
import rospy
import time
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32
import copy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

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
aruco_dict_type = cv2.aruco.DICT_5X5_50

matrix_coefficients = np.array(matrix_coefficients)
distortion_coefficients = np.array(distortion_coefficients)

bridge=CvBridge()
rgb_image=None

Realcoor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1,[0,0,0])

class sphere:
    def __init__(self, center, color):
        self.pcd = o3d.geometry.TriangleMesh.create_sphere(radius = 0.003)
        self.pcd.compute_vertex_normals()
        self.pcd.translate((center[0], center[1], center[2]), relative=False)
        self.pcd.paint_uniform_color(color)

def fixbox() :
    # Before rotate to canon
    y = 0.3
    x = 0.3
    z = 0.3  
    
    fix_box = np.array([
    [-x/2,-y/2,0],
    [-x/2, y/2,0],
    [x/2, y/2,0],
    [x/2,-y/2,0],

    [-x/2,-y/2,z],
    [-x/2, y/2,z],
    [x/2,-y/2,z],
    [x/2, y/2,z]
    ])

    # fixbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(fix_box))
    fixbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(fix_box))
    # theta = -(90-24.)*D2R
    # Rotation = [[1,0,0],
    #             [0,np.cos(theta),-np.sin(theta)],
    #             [0,np.sin(theta),np.cos(theta)]]
    
    # fixbox.transform(np.asarray([[1, 0, 0, 0], 
    #                     [0, np.cos(24.*D2R), -np.sin(24.*D2R), 0],
    #                     [0, np.sin(24.*D2R), np.cos(24.*D2R), 0.459], 
    #                     [0, 0, 0, 1]],dtype=np.float64))
    
    # fixbox.rotate(Rotation,(0,0,0))
    # fixbox.translate(np.asarray([0,0,0.92],dtype=np.float64),relative=False)
    fixbox.color = (0, 0, 0)

    return fixbox 

def convertCloudFromRosToOpen3d(ros_cloud):
    
    if ros_cloud is not None:
        # Get cloud data from ros_cloud
        field_names=[field.name for field in ros_cloud.fields]
        cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()
        if len(cloud_data)==0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        print("field_names : ",field_names)
        if "rgba" in field_names:
            IDX_RGB_IN_FIELD=3 # x, y, z, rgb
            
            # Get xyz
            xyz = [(x,y,z) for x,y,z,rgba in cloud_data ] # (why cannot put this line below rgb?)

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

        # return
        return open3d_cloud
    else:
        open3d_cloud = o3d.geometry.PointCloud()
        return open3d_cloud

def resize(img,scale_percent):
    # scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
        # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):                            
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray1,(15,15),0)
    cv2.imshow('blur',blur)
    ret,Thres = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    cv2.imshow('Thres',Thres)
    # gray = Thres
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.dilate(Thres,kernel,iterations = 1)
    cv2.imshow('gray',gray)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray1, arucoDict, parameters = arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
        cv2.imshow('img',resize(img,50))

    return [bboxs, ids]

def pose_esitmation(frame, aruco_dict_type , matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
    tvecs = []
    rvecs= []
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03075, matrix_coefficients,
                                                                       distortion_coefficients)
            print("tvec : ",tvec)
            print("rvec : ",rvec)
            tvecs.append(tvec)
            rvecs.append(rvec)
            # print("Point : ", corners[i])
            M = cv2.moments(corners[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX)
            print(cY)
            
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
            cv2.putText(frame, " X : "+str(tvec[0][0][0]*1000) , (cX+50, cY+50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.putText(frame, " Y : "+str(tvec[0][0][1]*1000) , (cX+50, cY+70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.putText(frame, " Z : "+str(tvec[0][0][2]*1000) , (cX+50, cY+90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
            cv2.putText(frame, " ids : "+str(ids[i]) , (cX+50, cY+110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    return frame,tvecs,rvecs

def create_board(markerSize,dict):
    marker_dis=0.125
    marker_offset=marker_dis/2
    #board1
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

global received_ros_cloud
received_ros_cloud = None
def callback(ros_cloud):
    global received_ros_cloud
    received_ros_cloud=ros_cloud
    rospy.loginfo("-- Received ROS PointCloud2 message.")

rospy.init_node('Zivid_Axis', anonymous=True)
rospy.Subscriber("/zivid_camera/color/image_color", Image, rgb_callback)
rospy.Subscriber("/zivid_camera/points/xyzrgba", PointCloud2, callback)  

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
arucoParams = cv2.aruco.DetectorParameters_create() 
board4x4=create_board(markerSize=0.03075,dict=arucoDict)


while True:
    if (rgb_image is not None) and (received_ros_cloud is not None) and not rospy.is_shutdown():
        rvec=None
        tvec=None

        (corners, ids,rejected)= cv2.aruco.detectMarkers(rgb_image,arucoDict,parameters=arucoParams)
        print(len(corners))
        print(corners)
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(rgb_image,corners,ids)
            _,rvec,tvec = cv2.aruco.estimatePoseBoard( corners, ids, board4x4, matrix_coefficients, distortion_coefficients,rvec,tvec)
            if rvec is not None and tvec is not None:
                cv2.aruco.drawAxis( rgb_image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.08 )
                cv2.imshow('rgb_image', rgb_image)

        transformation_matrix = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]],
                          dtype=float)

        transformation_matrix[:3, :3], _ = cv2.Rodrigues(rvec)
        transformation_matrix[ 3, :3] = np.asarray(tvec).transpose()
        box = fixbox()
        box.rotate(transformation_matrix[:3, :3],(0,0,0))
        box.translate(np.asarray(transformation_matrix[ 3, :3],dtype=np.float64),relative=True)

        centercoor = copy.deepcopy(Realcoor)
        centercoor.rotate(transformation_matrix[:3, :3],(0,0,0))
        centercoor.translate(np.asarray(transformation_matrix[ 3, :3],dtype=np.float64),relative=False)
        pcd = convertCloudFromRosToOpen3d(received_ros_cloud)
        o3d.visualization.draw_geometries([Realcoor,centercoor,pcd,box])

        received_ros_cloud = None
        rgb_image = None
        if cv2.waitKey(1) & 0xFF==ord('q'):
            
            break

cv2.destroyAllWindows()