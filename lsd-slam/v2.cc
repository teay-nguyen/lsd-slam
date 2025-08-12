struct Sim3 {
  // scale * R * x + t
  double s;                        // >0
  Eigen::Quaterniond R;            // unit
  Eigen::Vector3d t;               // translation

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Sim3Measurement {
  Sim3 S_ji;                       // measured transform: node j relative to i
  Eigen::Matrix<double,7,7> info;  // information (inverse covariance)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RobustKernel {
  enum Type { NONE, HUBER, CAUCHY, TUKEY } type;
  double delta;                    // e.g., Huber threshold
};

struct Keyframe {
  int id;                          // unique graph id
  int64_t timestamp_ns;

  // Pose estimate in GLOBAL frame (to be optimized)
  Sim3 pose;

  // Mapping payload
  int width, height;
  std::vector<cv::Mat> pyr;        // image pyramid (grayscale, CV_32F)
  cv::Mat depth_inv;               // inverse depth (CV_32F) for semi-dense pixels
  cv::Mat depth_var;               // variance (CV_32F)
  cv::Mat mask;                    // uint8 semi-dense mask

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseGraphEdge {
  int i;                           // from node id
  int j;                           // to node id

  Sim3Measurement meas;            // \hat{S}_{ji}
  RobustKernel robust;

  // Optional: enable/disable or switchable weight (for outlier handling)
  bool enabled;
  double switch_weight;            // if using switchable constraints

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseGraph {
  std::vector<Keyframe, Eigen::aligned_allocator<Keyframe>> nodes;
  std::vector<PoseGraphEdge, Eigen::aligned_allocator<PoseGraphEdge>> edges;

  // Fast lookup from id -> index in nodes
  std::unordered_map<int, size_t> id_to_index;
};

/*


// Add/find nodes
bool add_keyframe(PoseGraph& g, const Keyframe& kf);
Keyframe* find_keyframe(PoseGraph& g, int id);

// Add an edge (Sim3 constraint)
bool add_edge(PoseGraph& g, const PoseGraphEdge& e);

// Build linear system (GN/LM) from edges
struct LinearSystem {
  // block-sparse Hessian and gradient (7x7 blocks per node)
  // If using a solver library, you may not need to roll your own.
};

void assemble_normal_equations(const PoseGraph& g, LinearSystem& ls);

// Solve and update poses on manifold
void solve_and_update(PoseGraph& g, LinearSystem& ls);

// Utility: left/right comp, log/exp on Sim3 (for residuals & Jacobians)
Eigen::Matrix<double,7,1> sim3_log(const Sim3& S);
Sim3 sim3_exp(const Eigen::Matrix<double,7,1>& xi);
Sim3 compose(const Sim3& A, const Sim3& B);    // A * B
Sim3 inverse(const Sim3& S);


*/
