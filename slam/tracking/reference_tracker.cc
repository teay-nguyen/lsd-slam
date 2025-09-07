#include "reference_tracker.h"
#include "../model/frame.h"
#include "../depth_estimation/depth_map_pixel_hypothesis.h"
#include "../global_mapping/key_frame_graph.h"
#include "../util/fxn.h"
#include "../io_wrapper/image_display.h"


/*

builds a sparse, per-pyramid level reference point cloud
from a keyframe and keeps it around so the tracker can efficiently
align a new frame photometrically.

* 3d positions in camera coords (posData)
* image gradients (gradData = dx,dy)
* intensity and inverse-depth variance (colorAndVarData)
* source pixel indices (pointPosInXYGrid)
* Count of valid points(numData)

- lazily builds the levels point cloud used for photometric alignment (SE3/Sim3 estimation)
- guard state transitions


*/
namespace lsd_slam {

TrackingReference::TrackingReference() noexcept : keyframe(nullptr), frameID(-1), wh_allocated(0) {
    // zero-init per-level arrays
    for (int level = 0; level < PYRAMID_LEVELS; ++level) {
        posData[level]           = nullptr;
        gradData[level]          = nullptr;
        colorAndVarData[level]   = nullptr;
        pointPosInXYGrid[level]  = nullptr;
        numData[level]           = 0;
    }
}

TrackingReference::~TrackingReference() {
    std::unique_lock<std::mutex> lock(accessMutex);
    invalidate();
    releaseAll();
}

void TrackingReference::releaseAll() noexcept {
    for (int level = 0; level < PYRAMID_LEVELS; ++level) {
        // Arrays were allocated with new[], so delete[] them correspondingly
        if (posData[level] != nullptr)           { delete[] posData[level];          posData[level] = nullptr; }
        if (gradData[level] != nullptr)          { delete[] gradData[level];         gradData[level] = nullptr; }
        if (colorAndVarData[level] != nullptr)   { delete[] colorAndVarData[level];  colorAndVarData[level] = nullptr; }
        if (pointPosInXYGrid[level] != nullptr)  { delete[] pointPosInXYGrid[level]; pointPosInXYGrid[level] = nullptr; }
        numData[level] = 0;
    }
    wh_allocated = 0;
}

void TrackingReference::clearAll() {
    std::unique_lock<std::mutex> lock(accessMutex);
    for (int level = 0; level < PYRAMID_LEVELS; ++level)
        numData[level] = 0;
}

void TrackingReference::invalidate() noexcept {
    // Release the keyframe’s active lock if we still hold it
    if (keyframe != nullptr && keyframeLock.owns_lock())
        keyframeLock.unlock();

    keyframe = nullptr;
    frameID  = -1;
}


void TrackingReference::importFrame(Frame* sourceKF) {
    std::unique_lock<std::mutex> lock(accessMutex);

    // Acquire an “active” shared lock on the frame (prevents minimization)
    keyframeLock = sourceKF->getActiveLock();
    keyframe = sourceKF;
    frameID = keyframe->id();

    // reset allocation if dimensions changed (rare)
    const int new_wh = keyframe->width(0) * keyframe->height(0);
    if (new_wh != wh_allocated) {
        releaseAll();
        wh_allocated = new_wh;
    }

    // reset counts; buffers will be (re)filled on demand
    for (int level = 0; level < PYRAMID_LEVELS; ++level)
        numData[level] = 0;
}

void TrackingReference::makePointCloud(int level) {
    assert(keyframe != nullptr);
    std::unique_lock<std::mutex> lock(accessMutex);
    if (numData[level] > 0) return; // already built

    const int w = keyframe->width(level);
    const int h = keyframe->height(level);

    const float fxInvLevel = keyframe->fxInv(level);
    const float fyInvLevel = keyframe->fyInv(level);
    const float cxInvLevel = keyframe->cxInv(level);
    const float cyInvLevel = keyframe->cyInv(level);

    const float*               pyrIdepthSource    = keyframe->idepth(level);
    const float*               pyrIdepthVarSource = keyframe->idepthVar(level);
    const float*               pyrColorSource     = keyframe->image(level);
    const Eigen::Vector4f*     pyrGradSource      = keyframe->gradients(level);

    // Lazy allocate per-level arrays
    const int wh = w * h;
    if (posData[level]          == nullptr) posData[level]          = new Eigen::Vector3f[wh];
    if (pointPosInXYGrid[level] == nullptr) pointPosInXYGrid[level] = new int[wh];
    if (gradData[level]         == nullptr) gradData[level]         = new Eigen::Vector2f[wh];
    if (colorAndVarData[level]  == nullptr) colorAndVarData[level]  = new Eigen::Vector2f[wh];

    Eigen::Vector3f* posDataPT         = posData[level];
    int*             idxPT             = pointPosInXYGrid[level];
    Eigen::Vector2f* gradDataPT        = gradData[level];
    Eigen::Vector2f* colorAndVarDataPT = colorAndVarData[level];

    // Skip 1-pixel border (consistent with original)
    for (int x = 1; x < w - 1; ++x) {
        for (int y = 1; y < h - 1; ++y) {
            const int idx = x + y * w;

            // validity: variance > 0 and non-zero inverse depth
            if (pyrIdepthVarSource[idx] <= 0.0f || pyrIdepthSource[idx] == 0.0f)
                continue;

            // Back-project using intrinsics (inverse form): (X/Z, Y/Z, 1) scaled by Z
            // Here Z = 1 / inverseDepth
            const float Z = 1.0f / pyrIdepthSource[idx];
            *posDataPT = Z * Eigen::Vector3f(fxInvLevel * x + cxInvLevel,
                                             fyInvLevel * y + cyInvLevel,
                                             1.0f);

            // Store gradient (dx, dy)
            *gradDataPT = pyrGradSource[idx].head<2>();

            // Store intensity and variance
            *colorAndVarDataPT = Eigen::Vector2f(pyrColorSource[idx],
                                                 pyrIdepthVarSource[idx]);

            // Store linear index on the source image/grid
            *idxPT = idx;

            // advance outputs
            ++posDataPT;
            ++gradDataPT;
            ++colorAndVarDataPT;
            ++idxPT;
        }
    }

    numData[level] = static_cast<int>(posDataPT - posData[level]);
}

};