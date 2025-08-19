import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = glob.glob('data/*.jpg')

if __name__ == '__main__':
  chessboard_fp = '../lsd-slam/media/car_pov.mp4'
  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  frames = []

  for fname in images:
    frame = cv2.imread(fname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    if ret:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners2)

  # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  ret = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  for m in ret:
    print(m)
