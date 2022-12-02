import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):                            
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray1,(5,5),0)
    cv2.imshow('blur',blur)
    ret,Thres = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    cv2.imshow('Thres',Thres)
    kernel = np.ones((3,3),np.uint8)
    gray = cv2.dilate(Thres,kernel,iterations = 1)
    cv2.imshow('gray',gray)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def arucoAug(bbox, id, img, imgAug, drawId = True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgout = img + imgout
    return imgout
cap = cv2.VideoCapture(0)
imgAug = cv2.imread(r"D:\Oongking\iRAP\Robot arm\program\OpenCV\data\Way1.jpg")
while True:
    success, img = cap.read()
    arucofound = findArucoMarkers(img)
    # loop through all the markers and augment each one
    if len(arucofound[0])!=0:
        for bbox, id in zip(arucofound[0], arucofound[1]):
            img = arucoAug(bbox, id, img, imgAug)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()