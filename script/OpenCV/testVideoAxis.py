import cv2
import cv2.aruco as aruco
import numpy as np
import os

matrix_coefficients = [[ 814.31300186, 0, 255.40960568], [ 0, 814.10027848, 224.87643229], [ 0, 0, 1]]
distortion_coefficients = [[-2.15297149e-01, 1.59199358e+00, 1.85334845e-03, -2.66133599e-02, -4.39119721e+00]]
aruco_dict_type = cv2.aruco.DICT_6X6_250

matrix_coefficients = np.array(matrix_coefficients)
distortion_coefficients = np.array(distortion_coefficients)

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

def pose_esitmation(frame, aruco_dict_type , matrix_coefficients, distortion_coefficients):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.055, matrix_coefficients,
                                                                       distortion_coefficients)
            print("tvec : ",tvec)
            print("rvec : ",rvec)

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
            cv2.putText(frame, " ID : "+str(ids[i]) , (cX+50, cY+110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return frame

cap = cv2.VideoCapture(0)
imgAug = cv2.imread(r"D:\Oongking\iRAP\Robot arm\program\OpenCV\data\Way1.jpg")
while True:
    success, img = cap.read()
    # arucofound = findArucoMarkers(img)

    output = pose_esitmation(img, aruco_dict_type, matrix_coefficients, distortion_coefficients)

    cv2.imshow('Estimated Pose', output)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()