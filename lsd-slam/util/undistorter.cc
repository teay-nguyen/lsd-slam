#include "undistorter.h"
#include <fstream>
#include <sstream>

UndistorterOpenCV::UndistorterOpenCV(const std::string& config_fn) {
  m_valid = true;
  std::ifstream params_file(config_fn.c_str());
  assert(params_file.good());

  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  int img_width, img_height;

  cv::FileStorage fs(config_fn, cv::FileStorage::READ);
  fs["Camera_Matrix"] >> camera_matrix;
  fs["Distortion_Coefficients"] >> distortion_coefficients;
  fs["image_Width"] >> img_width;
  fs["image_Height"] >> img_height;
  m_out_width = img_width;
  m_out_height = img_height;
  fs.release();
  cv::Size new_img_size(m_out_width, m_out_height);

  m_output_calibration = 1;
  bool center_principal_pnt = false;
  m_K = cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients,
                                      cv::Size(img_width, img_height),
                                      m_output_calibration, new_img_size, nullptr,
                                      center_principal_pnt);

  cv::initUndistortRectifyMap(camera_matrix, distortion_coefficients, cv::Mat(),
                              m_K, new_img_size, CV_16SC2, m_map1, m_map2);

  m_originalK = camera_matrix.t();
  m_K = m_K.t();
}

void UndistorterOpenCV::undistort(cv::InputArray img, cv::OutputArray res) {
  cv::remap(img, res, m_map1, m_map2, cv::INTER_LINEAR);
}