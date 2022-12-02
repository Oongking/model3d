# from tabnanny import check
import cv2
import numpy as np
import os

# ARUCO_DICT = {
# 	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
# 	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
# 	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
# 	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
# 	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
# 	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
# 	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
# 	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
# 	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
# 	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
# 	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
# 	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
# 	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
# 	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
# 	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
# 	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
# 	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# 	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
# 	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
# 	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
# 	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
# }

def main():

    # camera_matrix=np.array([[625.604450,0.000000,336.271017],
    #                        [0.000000,625.572464,249.551023],
    #                        [0.000000,0.000000,1.000000]])
    # distortion=np.array([[-0.197848,0.593473,0.004297,0.000780,0.000000]])

    # projection=np.array([[634.477600,0.000000,334.238549,0.000000],
    #                 [0.000000,637.022705,250.051943,0.000000],
    #                 [0.000000,0.000000,1.000000,0.000000]])

    	# cv.aruco.getPredefinedDictionary
    # arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    arucoParams = cv2.aruco.DetectorParameters_create() 

    #Creating board
    board = cv2.aruco.GridBoard_create(14,10,0.0315,0.0081,arucoDict)
    # charucoBoard = cv2.aruco.CharucoBoard_create(7, 5, 0.1, 0.075, arucoDict)
    # img=cv2.aruco_CharucoBoard.draw(board,(1754,1240),5,50)
    
    img=cv2.aruco.drawPlanarBoard(board,(2480,1754),5,50)
    cv2.imwrite("/home/oongking/RobotArm_ws/src/model3d/Board/pic/poseboardA3.png",img)
   


if __name__ == '__main__':
  main()