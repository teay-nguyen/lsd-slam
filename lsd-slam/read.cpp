#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <sophus/sim3.hpp>
#include <g2o/core/base_vertex.h>
#include <unordered_set>
#include <array>
#include <memory>

static constexpr int PYRAMID_LEVELS = 5;

/*

what is the intrinsic matrix K? - captures how 3d points in the camera coordinate frame
get projected into pixel coordinates

K_inv is what you apply to a pixel homogeneous coordinate to recover the normalized image (camera) coordinates

tracking:
- tracking reference
- keyframe classes (e.g. frame)
- se3 tracker

fx, fy - focal lengths (?)
cx, cy - coordinates of the principle point (?)

gradients are used for computing the photometric error


TODO: write the se3 tracker and tracking reference 

*/

class keyframe_pose_obj;
class keyframe_obj;
class tracking_reference;
class se3_tracker;



// TODO: figure out what this does
class vertex_sim3 : public g2o::BaseVertex<7, Sophus::Sim3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() {
    _estimate = Sophus::Sim3d();
  }

  virtual void oplusImpl(const double* update_) {
    Eigen::Map<Eigen::Matrix<double, 7, 1>> update(const_cast<double*>(update_));
    if(m_fix_scale) { update[6] = 0; }
    setEstimate(Sophus::Sim3d::exp(update) * estimate());
  }

  bool m_fix_scale;
};


class keyframe_pose_obj {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  keyframe_pose_obj(keyframe_obj* t_frame);
  keyframe_pose_obj(const keyframe_pose_obj&) = delete;
  keyframe_pose_obj& operator=(const keyframe_pose_obj&) = delete;

  keyframe_pose_obj* m_tracking_parent;
  Sophus::Sim3d m_tracking_result_to_parent;

  int m_frame_id; // why?
  keyframe_obj* m_keyframe;

  //    node of the pose graph that gets optimized
  vertex_sim3* m_graph_vertex;

private:
  Sophus::Sim3d m_absolute_pos_cam_to_world;
  Sophus::Sim3d m_absolute_pos_cam_to_world_new; // added when merging optimization
};

keyframe_pose_obj::keyframe_pose_obj(keyframe_obj* t_frame) : m_tracking_parent(nullptr), m_keyframe(t_frame), m_graph_vertex(nullptr) {
  m_absolute_pos_cam_to_world = m_absolute_pos_cam_to_world_new = m_tracking_result_to_parent = Sophus::Sim3d();
}

class keyframe_obj {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  keyframe_obj(int t_id, int t_width, int t_height, const Eigen::Matrix3f& t_K);
  keyframe_obj(const keyframe_obj&) = delete;
  keyframe_obj& operator=(const keyframe_obj&) = delete;

  std::unordered_set<keyframe_obj*,
                     std::hash<keyframe_obj*>,
                     std::equal_to<keyframe_obj*>,
                     Eigen::aligned_allocator<keyframe_obj*>> m_neighbors;

  int m_id;
  std::array<int, PYRAMID_LEVELS> m_width, m_height;

  std::unique_ptr<keyframe_pose_obj> m_pose;
  Sophus::Sim3d m_last_constraint_tracked_cam_to_world;

  std::array<Eigen::Matrix3f, PYRAMID_LEVELS> m_K, m_K_inv;
  std::array<float, PYRAMID_LEVELS> m_fx, m_fy, m_cx, m_cy;
  std::array<float, PYRAMID_LEVELS> m_fix, m_fiy, m_cix, m_ciy; // inverse

  std::array<Eigen::Vector4f, PYRAMID_LEVELS> m_gradients;
};

keyframe_obj::keyframe_obj(int t_id, int t_width, int t_height, const Eigen::Matrix3f& t_K) : m_id(t_id) {
  m_pose = std::make_unique<keyframe_pose_obj>(this);

  m_K[0] = t_K;
  m_fx[0] = t_K(0,0);
	m_fy[0] = t_K(1,1);
	m_cx[0] = t_K(0,2);
	m_cy[0] = t_K(1,2);

  m_K_inv[0] = t_K.inverse();
  m_fix[0] = m_K_inv[0](0,0);
	m_fiy[0] = m_K_inv[0](1,1);
	m_cix[0] = m_K_inv[0](0,2);
	m_ciy[0] = m_K_inv[0](1,2);
}




// reference frame(?) to update map
class tracking_reference {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  tracking_reference();

  int m_frame_id;
  keyframe_obj* m_keyframe;

  std::array<Eigen::Vector3f*, PYRAMID_LEVELS> m_pos; // (x,y,z)
  std::array<Eigen::Vector2f*, PYRAMID_LEVELS> m_grad; // (dx, dy)
};

tracking_reference::tracking_reference() : m_frame_id(-1), m_keyframe(nullptr) {
  m_pos.fill(nullptr);
  m_grad.fill(nullptr);
}



class se3_tracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  se3_tracker(int t_w, int t_h, Eigen::Matrix3f t_K);

  int m_width, m_height;

  Eigen::Matrix3f m_K, m_K_inv;
  float m_fx, m_fy, m_cx, m_cy;
  float m_fix, m_fiy, m_cix, m_ciy;
};

se3_tracker::se3_tracker(int t_w, int t_h, Eigen::Matrix3f t_K) : m_width(t_w), m_height(t_h), m_K(t_K), m_K_inv(t_K.inverse()) {
  m_fx = t_K(0,0);
  m_fy = t_K(1,1);
	m_cx = t_K(0,2);
	m_cy = t_K(1,2);

  m_K_inv = t_K.inverse();
  m_fix = m_K_inv(0,0);
  m_fiy = m_K_inv(1,1);
	m_cix = m_K_inv(0,2);
	m_ciy = m_K_inv(1,2);
}





class slam_context {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  slam_context(int t_w, int t_h, Eigen::Matrix3f t_K);

  //    disable copying
  slam_context(const slam_context&) = delete;
  slam_context& operator=(const slam_context&) = delete;

  int m_width, m_height;
  Eigen::Matrix3f m_K;

  std::shared_ptr<keyframe_obj> m_current_keyframe;

private:
  std::unique_ptr<tracking_reference> m_reference_tracker;
  std::unique_ptr<se3_tracker> m_tracker;
};

slam_context::slam_context(int t_w, int t_h, Eigen::Matrix3f t_K) : m_width(t_w), m_height(t_h), m_K(t_K), m_current_keyframe(nullptr) {
  std::cout << "slam_context instantiated" << '\n';
}






int main(int argc, char** argv) {
  int w=0, h=0;
  Eigen::Matrix3f K;
  K << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  std::unique_ptr<slam_context> ctx = std::make_unique<slam_context>(w,h,K);

    // opencv
  const std::string inputPath = "./car_pov.mp4";

  cv::VideoCapture cap(inputPath);
  if (!cap.isOpened()) {
    std::cerr << "ERROR: could not open video source: " << inputPath << std::endl;
    return 1;
  }

  const std::string windowName = "SLAM";
  cv::namedWindow(windowName, cv::WINDOW_NORMAL);

  cv::Mat frame;
  while (true) {
    bool readSuccess = cap.read(frame);
    if (!readSuccess) {
      std::cout << "Finished reading or error occurred." << std::endl;
      break;
    }

    cv::Mat img_gray;
    cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
    cv::imshow(windowName, img_gray);

    if (cv::waitKey(30) >= 0) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
}
