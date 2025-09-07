
#pragma once
#include <opencv2/core/core.hpp>
#include <algorithm>

#include "../settings.h"
#include "sophus_util.h"


namespace lsd_slam {

template<typename T>class NotifyBuffer;
class Frame;

// reads interpolated element from a uchar* array
// SSE2 optimization possible
[[nodiscard]] inline float getInterpolatedElement(const float* const mat, const float x, const float y, const int width) noexcept {
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    float dx = x - static_cast<float>(ix);
    float dy = y - static_cast<float>(iy);
    float dxdy = dx*dy;
    const float* bp = mat+ix+iy*width;
    float res = dxdy * bp[1+width] + (dy-dxdy) * bp[width] + (dx-dxdy) * bp[1] + (1-dx-dy+dxdy) * bp[0];
    return res;
}

template<int N>
[[nodiscard]] inline Eigen::Matrix<float,N,1> getInterpolatedElement4N(const Eigen::Vector4f* const mat, float x, float y, int width) noexcept {
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    float dx = x - static_cast<float>(ix);
    float dy = y - static_cast<float>(iy);
    float dxdy = dx*dy;
    
    float w11 = dxdy;
    float w01 = dy - dxdy;
    float w10 = dx - dxdy;
    float w00 = 1.0f - dx - dy + dxdy;
    
    const Eigen::Vector4f* bp = mat + ix + iy + width;
    Eigen::Vector4f p00 = bp[0];
    Eigen::Vector4f p10 = bp[1];
    Eigen::Vector4f p01 = bp[width];
    Eigen::Vector4f p11 = bp[1 + width];
    
    Eigen::Vector4f v = w11 * p11 + w01 * p01 + w10 * p10 + w00 * p00;
    return v.template head<N>();
}

[[nodiscard]] inline Eigen::Vector3f getInterpolatedElement43(const Eigen::Vector4f* const mat, float x, float y, int width) noexcept {
    return getInterpolatedElement4N<3>(mat,x,y,width);
}

[[nodiscard]] inline Eigen::Vector4f getInterpolatedElement44(const Eigen::Vector4f* const mat, float x, float y, int width) noexcept {
    return getInterpolatedElement4N<4>(mat,x,y,width);
}

[[nodiscard]] inline Eigen::Vector2f getInterpolatedElement42(const Eigen::Vector4f* const mat, float x, float y, int width) noexcept {
    return getInterpolatedElement4N<2>(mat,x,y,width);
}

inline void fillCvMat(cv::Mat* mat, cv::Vec3b color) {
    if(mat == nullptr || mat->empty()) return;
    mat->setTo(color);
}

inline void setPixelInCvMat(cv::Mat* mat, cv::Vec3b color, int xx, int yy, int lvlFac) {
    int x0 = xx * lvlFac;
    int y0 = yy * lvlFac;
    int x1 = std::min(x0 + lvlFac, mat->cols);
    int y1 = std::min(y0 + lvlFac, mat->rows);
    if(x0 >= x1 || y0 >= y1) return;
    cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    mat->operator()(roi).setTo(color);
}

[[nodiscard]] inline cv::Vec3b getGrayCvPixel(float val) noexcept {
    unsigned char g = cv::saturate_cast<unsigned char>(val);
    return cv::Vec3b(g,g,g);
}

SE3 SE3CV2Sophus(const cv::Mat& R, const cv::Mat& t);
void printMessageOnCVImage(cv::Mat &image, std::string line1, std::string line2);

cv::Mat getDepthRainbowPlot(Frame* kf, int lvl=0);
cv::Mat getDepthRainbowPlot(const float* idepth, const float* idepthVar, const float* gray, int width, int height);
cv::Mat getVarRedGreenPlot(const float* idepthVar, const float* gray, int width, int height);

}