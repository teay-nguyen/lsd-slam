import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
images = glob.glob('data/*.jpg')

if __name__ == '__main__':
  chessboard_fp = '../lsd-slam/media/car_pov.mp4'
  objp = np.zeros((6*7,3), np.float32)
  objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
  objpoints, imgpoints, frames = [],[],[] # 3d/2d point in real world space, frames

  for fname in images:
    frame = cv2.imread(fname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    if ret:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
      imgpoints.append(corners2)

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  # ret = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

  img = cv2.imread('data/left12.jpg')
  h,w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  x, y, w, h = roi
  dst = dst[y:y+h, x:x+w]
  cv2.imwrite('calibresult.png', dst)
