#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

matrix_coefficients = [[ 814.31300186, 0, 255.40960568], [ 0, 814.10027848, 224.87643229], [ 0, 0, 1]]
distortion_coefficients = [[-2.15297149e-01, 1.59199358e+00, 1.85334845e-03, -2.66133599e-02, -4.39119721e+00]]
aruco_dict_type = cv2.aruco.DICT_6X6_250

matrix_coefficients = np.array(matrix_coefficients)
distortion_coefficients = np.array(distortion_coefficients)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    rospy.init_node('raw_input_cam', anonymous=True)
    image_pub = rospy.Publisher("/usb_cam/image_raw",Image,queue_size=5)
    bridge = CvBridge()
    while True:
        success, img = cap.read()




        image_pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))        
        cv2.imshow('Estimated Pose', img)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()