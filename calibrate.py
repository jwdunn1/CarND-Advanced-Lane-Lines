#_______________________________________________________________________________
# calibrate.py                                                             80->|

import cv2
import glob
import pickle
import numpy as np


#_______________________________________________________________________________
# Extract object points and image points for camera calibration

corners= []
# Prepare object points
objp= np.zeros((6*9,3), np.float32)
objp[:,:2]= np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints= [] # 3d points in real world space
imgpoints= [] # 2d points in image plane.

# Make a list of calibration images
images= glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img= cv2.imread(fname)
    imgName= fname.split('\\')[1]
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    print(idx)
    ret, corners= cv2.findChessboardCorners(gray, (9,6), None, flags=5 ) 
    
    # If found, add object points, image points
    if ret == True:
        print('  Append')
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imwrite('output_images/cal/'+imgName,img)


#_______________________________________________________________________________
# Calibrate, calculate distortion coefficients, and test undistortion

# Test undistortion on an image
img= cv2.imread('camera_cal/calibration1.jpg')
img_size= (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

undist= cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/caltest/calibration1-undist.jpg',undist)

# Save the camera calibration result
dist_pickle= {}
dist_pickle["mtx"]= mtx
dist_pickle["dist"]= dist
pickle.dump( dist_pickle, open( "dist_pickle.p", "wb" ) )
