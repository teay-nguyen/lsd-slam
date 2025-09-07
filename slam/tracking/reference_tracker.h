
#pragma once
#include "../settings.h"
#include "../util/eigen_core_include.h"
#include <array>
#include <mutex>
#include <shared_mutex>

namespace lsd_slam {

class Frame;
class DepthMapPixelHypothesis;
class KeyFrameGraph;

/**
 * Point cloud used to track frame poses.
 *
 * Basically this stores a point cloud generated from known frames. It is used to
 * track a new frame by finding a projection of the point cloud which makes it
 * look as much like the new frame as possible.
 *
 * It is intended to use more than one old frame as source for the point cloud.
 * Also other data like Kinect depth data could be imported.
 *
 * ATTENTION: as the level zero point cloud is not used for tracking, it is not
 * fully calculated. Only the weights are valid on this level!
 */
class TrackingReference {
public:
    /** Creates an empty TrackingReference with optional preallocation per level. */
    TrackingReference() noexcept;
    ~TrackingReference();

    TrackingReference(const TrackingReference&) = delete;
    TrackingReference& operator=(const TrackingReference&) = delete;
    TrackingReference(TrackingReference&&) = delete;
    TrackingReference& operator=(TrackingReference&&) = delete;

    void importFrame(Frame* source);

    Frame* keyframe = nullptr;
    std::shared_lock<std::shared_mutex> keyframeLock;
    int frameID = -1;

    void makePointCloud(int level);
    void clearAll();
    void invalidate() noexcept;

    // World-space positions (x,y,z)
    std::array<Eigen::Vector3f*, PYRAMID_LEVELS> posData{{nullptr}};
    // Image-space gradients (dx, dy)
    std::array<Eigen::Vector2f*, PYRAMID_LEVELS> gradData{{nullptr}};
    // Intensity and variance (I, Var)
    std::array<Eigen::Vector2f*, PYRAMID_LEVELS> colorAndVarData{{nullptr}};
    // Linearized image index (x + y*width)
    std::array<int*, PYRAMID_LEVELS> pointPosInXYGrid{{nullptr}};
    // Number of valid points per level
    std::array<int, PYRAMID_LEVELS> numData{{0}};

private:
    int wh_allocated=0;
    std::mutex accessMutex;
    void releaseAll() noexcept;
};
}