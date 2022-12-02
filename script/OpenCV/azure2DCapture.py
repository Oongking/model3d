#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import time

from ctypes import * # convert float to uint32

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


bridge=CvBridge()
rgb_image=None
depth_image = None

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


rospy.init_node('Azure_2DCapture', anonymous=True)
rospy.Subscriber("/rgb/image_raw", Image, rgb_callback)

i = 1

while True:
    if (rgb_image is not None) and not rospy.is_shutdown():
                
        cv2.imshow('rgb_image', rgb_image)
        

        key = cv2.waitKey(1)
        if key & 0xFF==ord('s'):
            cv2.imwrite(f"/home/oongking/RobotArm_ws/src/model3d/script/OpenCV/data/Azure/chessboard{i}.png",rgb_image)
            i+=1
        if key & 0xFF==ord('q'):
            break

        rgb_image = None


cv2.destroyAllWindows()


    