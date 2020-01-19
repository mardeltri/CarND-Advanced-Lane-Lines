import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os.path

from param import CHESSBOARD_SHAPE

class CalibrateCamera():
    def __init__(self, path, chessboard_shape):
        self._nx = chessboard_shape[0]
        self._ny = chessboard_shape[1]
        
        self._path = path
        
        data_path = path + 'wide_dist_pickle.p'
        if os.path.exists(data_path):
            print('Loading previous data')
            f = open(data_path,"rb")
            calibration_data = pickle.load(f)
            f.close()
        else:
            print('Computing calibration')
            calibration_data = self._calibrate()
            f = open(data_path, "wb")
            pickle.dump(calibration_data, f)
            f.close()
            print('Calibration done')
            
        self._mtx = calibration_data["mtx"]
        self._dist = calibration_data["dist"]
        
    def undistort(self, img):
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
       
    def _calibrate(self):
        nx = self._nx
        ny = self._ny
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny * nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        img_list = self._path + 'calibration*.jpg'
        images = glob.glob(img_list)
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
        calibration_data = {}
        calibration_data["mtx"] = mtx
        calibration_data["dist"] = dist
        return calibration_data
    
if __name__ == '__main__':
    cam_calibration = CalibrateCamera('..\\camera_cal\\', chessboard_shape)
    img = cv2.imread('..\\camera_cal\\calibration1.jpg')
    undist = cam_calibration.undistort(img)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.savefig('output_images/undistort_output_c'+str(idx+1)+'.png')
    plt.show()