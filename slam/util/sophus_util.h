#pragma once

#include "../../thirdparty/sophus/sim3.hpp"
#include "../../thirdparty/sophus/se3.hpp"

// Typedef and conversion macro for Eigen matrices to currently used type.
// NOTE: a "no-op conversion" is free in terms of performance, as it should be compiled out.
#ifdef SOPHUS_USE_FLOAT
typedef Sophus::SE3f SE3;
typedef Sophus::Sim3f Sim3;
typedef Sophus::SO3f SO3;
#define toSophus(x) ((x).cast<float>())
#define sophusType float
#else
typedef Sophus::SE3d SE3;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;
#define toSophus(x) ((x).cast<double>())
#define sophusType double
#endif


namespace lsd_slam {

inline Sim3 sim3FromSE3(const SE3& se3, sophusType scale) {
  Sim3 result(se3.unit_quaternion(), se3.translation());
  result.setScale(scale);
  return result;
}

inline SE3 se3FromSim3(const Sim3& sim3) {
  return SE3(sim3.quaternion(), sim3.translation());
}


}

extern template class Eigen::Quaternion<float>;
extern template class Eigen::Quaternion<double>;