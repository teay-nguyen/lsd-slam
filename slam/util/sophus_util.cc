#include "../../thirdparty/sophus/se3.hpp"
#include "../../thirdparty/sophus/sim3.hpp"

// Compile the templates here once so they don't need to be compiled in every
// other file using them.
//
// Other files then include SophusUtil.h which contains extern template
// declarations to prevent compiling them there again. (For this reason,
// this header must not be included here).
//
// Eigen::Matrix seemingly cannot be instantiated this way, as it tries to
// compile a constructor variant for 4-component vectors, resulting in a
// static assertion failure.


template class Eigen::Quaternion<float>;
template class Eigen::Quaternion<double>;