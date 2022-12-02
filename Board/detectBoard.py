#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('abu_india')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import tf
from tf.transformations import *
import numpy as np



def create_board(markerSize,dict):
    corner_dis=markerSize/2
    pre_dis = 0
    dis_shift = 0.1
    board_corners=[]

    first_marker=np.array([[pre_dis,-corner_dis+dis_shift,corner_dis],
                           [pre_dis,corner_dis+dis_shift,corner_dis],
                           [pre_dis,corner_dis+dis_shift,-corner_dis],
                           [pre_dis,-corner_dis+dis_shift,-corner_dis,]],dtype=np.float32)
    board_corners.append(first_marker)

    second_marker=np.array([[pre_dis,-corner_dis-dis_shift,corner_dis],
                           [pre_dis,corner_dis-dis_shift,corner_dis],
                           [pre_dis,corner_dis-dis_shift,-corner_dis],
                           [pre_dis,-corner_dis-dis_shift,-corner_dis,]],dtype=np.float32)
    
    board_corners.append(second_marker)
    # rad=(np.pi/4)
    # Z rot
    # rot=np.array([[np.cos(rad),-np.sin(rad),0],
    #             [np.sin(rad),np.cos(rad),0],
    #             [0,0,1]])
    # rot=np.array([[1,0,0],
    #             [0,np.cos(rad),-np.sin(rad)],
    #             [0,np.sin(rad),np.cos(rad)]])

    # next_marker=np.matmul(rot,first_marker.transpose(),dtype=np.float32).transpose()
    # board_corners.append(next_marker)

    # for i in range(1,6):
    #     rad=i*(np.pi/2)
    #     if(i<4):
    #         rot=np.array([[np.cos(rad),-np.sin(rad),0],
    #                     [np.sin(rad),np.cos(rad),0],
    #                     [0,0,1]])
    #     else:
    #         if i==4:
    #             rad=-np.pi/2
                
    #         elif i==5:
    #             rad=np.pi/2
    #         rot=np.array([[np.cos(rad),0,np.sin(rad)],
    #                         [0,1,0],
    #                         [-np.sin(rad),0,np.cos(rad)]])
    #     next_marker=np.matmul(rot,first_marker.transpose(),dtype=np.float32).transpose()
    #     board_corners.append(next_marker)

    board_ids = np.array([[11],[9]], dtype=np.int32)
    board = cv2.aruco.Board_create(board_corners,dict, board_ids )
    return board



class estimate_pos:

  def __init__(self):


    # usb camera param
    self.camera_matrix=np.array([[ 814.31300186, 0, 255.40960568], 
                                [ 0, 814.10027848, 224.87643229], 
                                [ 0, 0, 1]])
    self.distortion=np.array([[-2.15297149e-01, 1.59199358e+00, 1.85334845e-03, -2.66133599e-02, -4.39119721e+00]])

    self.projection=np.array([[634.477600,0.000000,334.238549,0.000000],
                        [0.000000,637.022705,250.051943,0.000000],
                        [0.000000,0.000000,1.000000,0.000000]])
    #

    self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    self.arucoParams = cv2.aruco.DetectorParameters_create()
    self.board=create_board(markerSize=0.055,dict=self.arucoDict)
    self.image_pub = rospy.Publisher("/image_pose",Image,queue_size=5)

    self.bridge = CvBridge()
    #usb_cam
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
    #azure
    # self.image_sub = rospy.Subscriber("/rgb/image_raw/compressed",CompressedImage,self.callback)    
    # self.cam_info_sub = rospy.Subscriber("/rgb/camera_info",CameraInfo,self.cam_info_callback)
    
    self.rvec=None
    self.tvec=None
    self.br = tf.TransformBroadcaster()
  def cam_info_callback(self,data):
      self.K=np.asarray(data.K)
      self.distortion=np.asarray(data.D)
      self.camera_matrix=self.K.reshape(3,3)      
      
  def callback(self,data):
    
    try:
      #usb_cam
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    # convert compressedImage
      # np_arr = np.fromstring(data.data,np.uint8)
      # cv_image=cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
    except CvBridgeError as e:
      print(e)
    
    (corners, ids,rejected)= cv2.aruco.detectMarkers(cv_image,self.board.dictionary,parameters=self.arucoParams)
    self.rvec=None
    self.tvec=None
    if len(corners) > 0:
      cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)

      _,self.rvec,self.tvec = cv2.aruco.estimatePoseBoard( corners, ids, self.board, self.camera_matrix, self.distortion,self.rvec,self.tvec)

      if self.rvec is not None and self.tvec is not None:

        rotation_matrix = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]],
                          dtype=float)

        rotation_matrix[:3, :3], _ = cv2.Rodrigues(self.rvec)

        self.q= tf.transformations.quaternion_from_matrix(rotation_matrix)

        cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.distortion, self.rvec, self.tvec, 0.08 )

      
        cv_image=cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.distortion, self.rvec, self.tvec, 0.05)
      


        self.br.sendTransform((self.tvec[0], self.tvec[1], self.tvec[2]),
                            (self.q[0], self.q[1], self.q[2], self.q[3]),
                            rospy.Time.now(),
                            "target_pose",
                            "map") 

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
    rospy.init_node('estimatePoseBoard', anonymous=True)
 
    estimatePose = estimate_pos()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)