#include "fxn.h"
#include "sophus_util.h"
#include <opencv2/opencv.hpp>
#include "../model/frame.h"

namespace lsd_slam {

SE3 SE3CV2Sophus(const cv::Mat &R, const cv::Mat &t)
{
    CV_Assert(R.rows == 3 && R.cols == 3);
    CV_Assert(t.total() == 3 && (t.cols == 1 || t.rows == 1));

    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);

    Sophus::Matrix3f sR;
    Sophus::Vector3f st;
    for (int c = 0; c < 3; ++c) {
        sR(0, c) = static_cast<float>(R64.at<double>(0, c));
        sR(1, c) = static_cast<float>(R64.at<double>(1, c));
        sR(2, c) = static_cast<float>(R64.at<double>(2, c));
        st[c] = static_cast<float>(t64.at<double>(c));
    }

    return SE3(toSophus(sR.inverse()), toSophus(st));
}

void printMessageOnCVImage(cv::Mat &image, std::string line1,std::string line2)
{
  if (image.empty()) return;

    // Shade a bottom band
    const int bandHeight = std::min(30, image.rows);
    cv::Rect band(0, image.rows - bandHeight, image.cols, bandHeight);
    cv::Mat roi = image(band);

    // Multiply by 0.5 to darken (works for 8U and 32F images; convertScaleAbs will clamp)
    if (roi.type() == CV_8UC3 || roi.type() == CV_8UC1) cv::convertScaleAbs(roi, roi, 0.5, 0.0);
    else roi *= 0.5;

    const double fontScale = 0.5;
    const int thickness = 1;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Scalar color(200, 200, 250);

    cv::putText(image, line1, cv::Point(10, image.rows - 18), font, fontScale, color, thickness, cv::LINE_AA);
    cv::putText(image, line2, cv::Point(10, image.rows - 5),  font, fontScale, color, thickness, cv::LINE_AA);
}


cv::Mat getDepthRainbowPlot(Frame* kf, int lvl)
{
    if(!kf) return {};
    return getDepthRainbowPlot(kf->idepth(lvl), kf->idepthVar(lvl), kf->image(lvl),
                               kf->width(lvl), kf->height(lvl));
}

cv::Mat getDepthRainbowPlot(const float* idepth, const float* idepthVar,
                            const float* gray, int width, int height)
{
    cv::Mat res = cv::Mat(height,width,CV_8UC3);
    if(gray != nullptr)
    {
        cv::Mat keyFrameImage(height, width, CV_32F, const_cast<float*>(gray));
        cv::Mat keyFrameImage8u;
        keyFrameImage.convertTo(keyFrameImage8u, CV_8UC1);
        cv::cvtColor(keyFrameImage8u, res, cv::COLOR_GRAY2RGB);
    }
    else
        res.setTo(cv::Vec3b(255,170,168));

    for (int y = 0; y < height; ++y) {
        auto* row = res.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const int idx = x + y * width;
            const float id = idepth[idx];
            if (id >= 0.0f && idepthVar[idx] >= 0.0f) {
                float r = std::fabs((0.0f - id) * 255.0f);
                float g = std::fabs((1.0f - id) * 255.0f);
                float b = std::fabs((2.0f - id) * 255.0f);

                const auto rc = static_cast<std::uint8_t>(std::clamp(r, 0.0f, 255.0f));
                const auto gc = static_cast<std::uint8_t>(std::clamp(g, 0.0f, 255.0f));
                const auto bc = static_cast<std::uint8_t>(std::clamp(b, 0.0f, 255.0f));

                // Original used (255-rc, 255-gc, 255-bc)
                row[x] = cv::Vec3b(255 - rc, 255 - gc, 255 - bc);
            }
        }
    }
    return res;
}
cv::Mat getVarRedGreenPlot(const float* idepthVar, const float* gray,
                           int width, int height)
{
    std::vector<float> idepthVarExt(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    std::copy(idepthVar, idepthVar + static_cast<std::size_t>(width) * static_cast<std::size_t>(height), idepthVarExt.begin());

    const auto idxAt = [width](int x, int y) { return y * width + x; };

    for (int x = 2; x < width - 2; ++x) {
        for (int y = 2; y < height - 2; ++y) {
            const int idx = idxAt(x, y);
            if (idepthVar[idx] <= 0.0f) {
                idepthVarExt[idx] = -1.0f;
                continue;
            }

            float sumIvar = 0.0f;
            float numIvar = 0.0f;
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int nidx = idxAt(nx, ny);
                    const float var = idepthVar[nidx];
                    if (var > 0.0f) {
                        // distance factor like original
                        const float distFac = static_cast<float>(dx * dx + dy * dy) * (0.075f * 0.075f) * 0.02f;
                        const float ivar = 1.0f / (var + distFac);
                        sumIvar += ivar;
                        numIvar += 1.0f;
                    }
                }
            }
            idepthVarExt[idx] = (numIvar > 0.0f) ? (numIvar / sumIvar) : -1.0f;
        }
    }

    cv::Mat res = cv::Mat(height,width,CV_8UC3);
    if(gray != nullptr)
    {
        cv::Mat keyFrameImage(height, width, CV_32F, const_cast<float*>(gray));
        cv::Mat keyFrameImage8u;
        keyFrameImage.convertTo(keyFrameImage8u, CV_8UC1);
        cv::cvtColor(keyFrameImage8u, res, cv::COLOR_GRAY2RGB);
    }
    else
        res.setTo(cv::Vec3b(255,170,168));

    for (int y = 0; y < height; ++y) {
        auto* row = res.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const float idv = idepthVarExt[static_cast<std::size_t>(x+y*width)];
            if (idv > 0.0f) {
                float var = std::sqrt(idv);
                float value = var * 60.0f * 255.0f * 0.5f - 20.0f;
                value = std::clamp(value, 0.0f, 255.0f);
                const std::uint8_t v = static_cast<std::uint8_t>(value);
                // BGR: (0, 255 - v, v) → green to red ramp
                row[x] = cv::Vec3b(0, static_cast<std::uint8_t>(255 - v), v);
            }
        }
    }

    return res;
}
}