import cv2
import cv2.aruco as aruco
import numpy as np
import os

width=8 
height=6
square_size = 0.025

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24.856, 0.001)

objp = np.zeros((height*width, 3), np.float32)
objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

objp = objp * square_size

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
while True:
    success, img = cap.read()
    if success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("ret : ", ret)
        print("mtx : ", mtx)
        print("dist : ", dist)
        print("rvecs : ", rvecs)
        print("tvecs : ", tvecs)


        cv2.imshow("Original Image",img)   
        if cv2.waitKey(1) & 0xFF==ord('q'):
            f = open("/home/oongking/RobotArm_ws/src/model3d/script/OpenCV/calbrationText/Cam1.txt","w+")
            f.write("ret : ")
            f.write(str(ret))
            f.write('\n')
            f.write("mtx : ")
            f.write(str(mtx))
            f.write('\n')
            f.write("dist : ")
            f.write(str(dist))
            f.write('\n')
            f.write("rvecs : ")
            f.write(str(rvecs))
            f.write('\n')
            f.write("tvecs : ")
            f.write(str(tvecs))
            break

cv2.destroyAllWindows()