#pragma once
#if defined(ANDROID)
#include <cmath>
namespace std {
using ::log1p;
}
#endif

#include <Eigen/Core>
