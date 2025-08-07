#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <sophus/sim3.hpp>
#include <g2o/core/base_vertex.h>
#include <unordered_set>

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

  bool m_fix_scale;

  virtual void setToOriginImpl() {
    _estimate = Sophus::Sim3d();
  }

  virtual void oplusImpl(const double* update_) {
    Eigen::Map<Eigen::Matrix<double, 7, 1>> update(const_cast<double*>(update_));
    if(m_fix_scale) { update[6] = 0; }
    setEstimate(Sophus::Sim3d::exp(update) * estimate());
  }
};


class keyframe_pose_obj {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  keyframe_pose_obj(keyframe_obj* t_frame);
  ~keyframe_pose_obj();

  keyframe_pose_obj* m_tracking_parent;
  Sophus::Sim3d m_tracking_result_to_parent;

  int frame_id; // why?
  keyframe_obj* m_keyframe;

  //    node of the pose graph that gets optimized
  vertex_sim3* m_graph_vertex;

private:
  Sophus::Sim3d m_absolute_pos_cam_to_world;
  Sophus::Sim3d m_absolute_pos_cam_to_world_new; // added when merging optimization
};

class keyframe_obj {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  std::unordered_set<keyframe_obj*,
                     std::hash<keyframe_obj*>,
                     std::equal_to<keyframe_obj*>,
                     Eigen::aligned_allocator<keyframe_obj*>> m_neighbors;

  int m_id;
  int m_width[PYRAMID_LEVELS], m_height[PYRAMID_LEVELS];

  keyframe_pose_obj* m_pose;
  Sophus::Sim3d m_last_constraint_tracked_cam_to_world;

  Eigen::Matrix3f m_K[PYRAMID_LEVELS], m_K_inv[PYRAMID_LEVELS];

  Eigen::Vector4f m_gradients[PYRAMID_LEVELS];
};




// reference frame(?) to update map
class tracking_reference {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

class se3_tracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  int m_width, m_height;

  Eigen::Matrix3f m_K, m_K_inv;
};

class slam_context {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  int m_width, m_height;
  Eigen::Matrix3f m_K;
};






int main(int argc, char** argv) {
  slam_context ctx;

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
