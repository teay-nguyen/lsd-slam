#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <sophus/sim3.hpp>
#include <sophus/se3.hpp>
#include <g2o/core/base_vertex.h>
#include <unordered_set>
#include <array>
#include <memory>

static constexpr int PYRAMID_LEVELS = 5;

using FAligned = Eigen::aligned_allocator<float>;


struct dense_depth_tracker_settings {

};


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
NOTE: there's a noticable lack of vectors (references), owning pointers are used in lieu
NOTE: what is the inverse depth variance?
NOTE: document the use of the inverse depth variance (1/z)

*/

class keyframe_pose_obj;
class keyframe_obj;
class tracking_reference;
class se3_tracker;



// TODO: figure out what this does
class vertex_sim3 : public g2o::BaseVertex<7, Sophus::Sim3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  vertex_sim3() : m_fix_scale(false) {}

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
// tracking_reference precedes the se3_tracker
class tracking_reference {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  tracking_reference();

  int m_frame_id;
  keyframe_obj* m_keyframe;

  // NOTE: not sure why original source code used pointers for this
  std::array<std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>, PYRAMID_LEVELS> m_pos;   // (x,y,z)
  std::array<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>, PYRAMID_LEVELS> m_grad;  // (dx,dy)
  std::array<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>, PYRAMID_LEVELS> m_Ivar;  // (I, var) color and variance
};

tracking_reference::tracking_reference() : m_frame_id(-1), m_keyframe(nullptr) {

}



class se3_tracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  se3_tracker(int t_w, int t_h, Eigen::Matrix3f t_K);
  se3_tracker(const se3_tracker&) = delete;
  se3_tracker& operator=(const se3_tracker&) = delete;
  ~se3_tracker();

  int m_width, m_height;

  Eigen::Matrix3f m_K, m_K_inv;
  float m_fx, m_fy, m_cx, m_cy;
  float m_fix, m_fiy, m_cix, m_ciy;

  Sophus::SE3d track_frame(tracking_reference* t_ref, keyframe_obj* t_frame, const Sophus::SE3d& t_frame_to_ref_initial_estimate);

private:
  /*

  ref_pnt - 3d points in reference keyframe
  ref_col_var - per point photometric data (stored grayscale intensity, intensity variance)
  idx_buf - valid point indicecs
  ref_num - number of points in reference set
  frame - current frame being aligned to reference
  ref_to_frame - current pose estimate from reference frame to current frame (rot + translation), this is what the optimizer is trying to refine to minimize photometric error
  int lvl - img pyramid level

  */

  float calculate_residual_and_bufs(const Eigen::Vector3f* ref_pnt,
                                    const Eigen::Vector2f* ref_col_var,
                                    int* idx_buf,
                                    int ref_num,
                                    const keyframe_obj& frame,
                                    const Sophus::SE3d& ref_to_frame,
                                    int lvl);

  /*

  per point working memory the se3 tracker keeps for photometric alignment

  */

  std::vector<float, FAligned> m_buf_warped_residual, m_buf_warped_dx, m_buf_warped_dy, m_buf_warped_x,
	                             m_buf_warped_y, m_buf_warped_z, m_buf_d, m_buf_idepthVar, m_buf_weight_p;

	int m_buf_warped_size;
};

se3_tracker::se3_tracker(int t_w, int t_h, Eigen::Matrix3f t_K) : m_width(t_w), m_height(t_h), m_K(t_K), m_K_inv(t_K.inverse()) {
  m_fx = t_K(0,0);
  m_fy = t_K(1,1);
	m_cx = t_K(0,2);
	m_cy = t_K(1,2);

  m_fix = m_K_inv(0,0);
  m_fiy = m_K_inv(1,1);
	m_cix = m_K_inv(0,2);
	m_ciy = m_K_inv(1,2);

  m_buf_warped_size = 0;
}

se3_tracker::~se3_tracker() {
  // free (owning) buffers

}

Sophus::SE3d se3_tracker::track_frame(tracking_reference* t_ref, keyframe_obj* t_frame,
                                      const Sophus::SE3d& t_frame_to_ref_initial_estimate) {
  return Sophus::SE3d();
}


float se3_tracker::calculate_residual_and_bufs(const Eigen::Vector3f* ref_pnt,
                                  const Eigen::Vector2f* ref_col_var,
                                  int* idx_buf,
                                  int ref_num,
                                  const keyframe_obj& frame,
                                  const Sophus::SE3d& ref_to_frame,
                                  int lvl) {
  return 0.f;
}





class slam_context {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  slam_context(int t_w, int t_h, Eigen::Matrix3f t_K);
  ~slam_context() = default;
  //    disable copying
  slam_context(const slam_context&) = delete;
  slam_context& operator=(const slam_context&) = delete;

  void random_init(int frame_id);

  int m_width, m_height;
  Eigen::Matrix3f m_K;

  std::shared_ptr<keyframe_obj> m_current_keyframe;

private:
  //    tracking thread
  std::unique_ptr<tracking_reference> m_reference_tracker;
  std::unique_ptr<se3_tracker> m_pose_tracker;

  // mapping thread
  std::unique_ptr<tracking_reference> m_mapping_reference_tracker;
};

slam_context::slam_context(int t_w, int t_h, Eigen::Matrix3f t_K) : m_width(t_w), m_height(t_h),
                                                                    m_K(t_K), m_current_keyframe(nullptr) {
  std::cout << "slam_context instantiated" << '\n';
  m_reference_tracker = std::make_unique<tracking_reference>();
  m_mapping_reference_tracker = std::make_unique<tracking_reference>();
  m_pose_tracker = std::make_unique<se3_tracker>(t_w, t_h, t_K);
}

void slam_context::random_init(int frame_id) {
  m_current_keyframe.reset(new keyframe_obj(frame_id, m_width, m_height, m_K));
}





int main(int argc, char** argv) {
  int w=1280, h=720;
  Eigen::Matrix3f K;
  K << 700.f, 0.f, 640.f,
      0.f, 700.f, 360.f,
      0.f,   0.f,   1.f;

  int running_idx = 0;
  std::unique_ptr<slam_context> ctx = std::make_unique<slam_context>(w,h,K);
  ctx->random_init(running_idx);

    // opencv
  const std::string inputPath = "./car_pov.mp4";

  cv::VideoCapture cap(inputPath);
  if (!cap.isOpened()) {
    std::cerr << "ERROR: could not open video source: " << inputPath << std::endl;
    return 1;
  }

  const std::string windowName = "SLAM";
  cv::namedWindow(windowName, cv::WINDOW_NORMAL);

  cv::Mat frame = cv::Mat(w,h,CV_8U);
  while (true) {
    bool readSuccess = cap.read(frame);
    if (!readSuccess) {
      std::cout << "Finished reading or error occurred." << std::endl;
      break;
    }

    // all the warps/Jacobians assume a pinhole model and stable brightness.
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
