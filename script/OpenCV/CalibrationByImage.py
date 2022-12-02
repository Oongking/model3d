# https://machinelearningknowledge.ai/augmented-reality-using-aruco-marker-detection-with-python-opencv/
import numpy as np
import cv2
import cv2.aruco as aruco

def calibrate(path, square_size, width=8, height=6, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    if visualize:
        cv2.imshow('img',img)
        cv2.waitKey(0)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    print("arucoDict : ",arucoDict)
    arucoParam = aruco.DetectorParameters_create()
    print("arucoParam : ",arucoParam)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    print(ids)
    if draw:
        print("Box : ",bboxs)
        aruco.drawDetectedMarkers(img, bboxs) 

# path = r'D:\Oongking\iRAP\Robot arm\program\OpenCV\data\ARcode.png'
path = r'D:\Oongking\iRAP\Robot arm\program\OpenCV\data\Chessboard.jpg'
img = cv2.imread(path)
blurred_img = cv2.GaussianBlur(img, (21,21),0)
hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
# findArucoMarkers(img)
square_size = 0.025
ret, mtx, dist, rvecs, tvecs = calibrate(path, square_size, visualize= True)

h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

print("ret : ", ret)
print("mtx : ", mtx)
print("dist : ", dist)
print("rvecs : ", rvecs)
print("tvecs : ", tvecs)
while True:
    cv2.imshow("Original Image",img)
    cv2.imshow("Undistortion Image",dst)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()