#pragma once

#include "../util/sophus_util.h"
#include "../settings.h"

#include <Eigen/Core>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <cstddef>   // std::size_t
#include <cstring>   // std::memset

#include "frame_pose_struct.h"
#include "frame_memory.h"

namespace lsd_slam {

class DepthMapPixelHypothesis;
class TrackingReference;

class Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    friend class FrameMemory;

    Frame(int id, int width, int height, const Eigen::Matrix3f& K,
          double timestamp, const unsigned char* image);

    Frame(int id, int width, int height, const Eigen::Matrix3f& K,
          double timestamp, const float* image);

    ~Frame();

    /** Sets or updates idepth and idepthVar on level zero. Invalidates higher levels. */
    void setDepth(const DepthMapPixelHypothesis* newDepth);

    /** Calculates mean information for statistical purposes. */
    void calculateMeanInformation();

    /** Sets ground truth depth (real, not inverse!) on level zero. Invalidates higher levels. */
    void setDepthFromGroundTruth(const float* depth, float cov_scale = 1.0f);

    /** Prepare this frame for stereo with another frame. */
    void prepareForStereoWith(Frame* other, const Sim3& thisToOther,
                              const Eigen::Matrix3f& K, int level);

    // Accessors
    inline int id() const;

    inline int  width(int level = 0) const;
    inline int  height(int level = 0) const;

    inline const Eigen::Matrix3f& K(int level = 0) const;
    inline const Eigen::Matrix3f& KInv(int level = 0) const;
    inline float fx(int level = 0) const;
    inline float fy(int level = 0) const;
    inline float cx(int level = 0) const;
    inline float cy(int level = 0) const;
    inline float fxInv(int level = 0) const;
    inline float fyInv(int level = 0) const;
    inline float cxInv(int level = 0) const;
    inline float cyInv(int level = 0) const;

    inline double timestamp() const;

    inline float*                  image(int level = 0);
    inline const Eigen::Vector4f*  gradients(int level = 0);
    inline const float*            maxGradients(int level = 0);
    inline bool                    hasIDepthBeenSet() const;
    inline const float*            idepth(int level = 0);
    inline const float*            idepthVar(int level = 0);
    inline const unsigned char*    validity_reAct();
    inline const float*            idepth_reAct();
    inline const float*            idepthVar_reAct();

    inline bool* refPixelWasGood();
    inline bool* refPixelWasGoodNoCreate();
    inline void  clear_refPixelWasGood();

    /** Flags for require()/requirePyramid(). */
    enum DataFlags {
        IMAGE         = 1<<0,
        GRADIENTS     = 1<<1,
        MAX_GRADIENTS = 1<<2,
        IDEPTH        = 1<<3,
        IDEPTH_VAR    = 1<<4,
        REF_ID        = 1<<5,

        ALL = IMAGE | GRADIENTS | MAX_GRADIENTS | IDEPTH | IDEPTH_VAR | REF_ID
    };

    void setPermaRef(TrackingReference* reference);
    void takeReActivationData(DepthMapPixelHypothesis* depthMap);

    // Hold this shared_lock while any minimizable arrays are in use.
    inline std::shared_lock<std::shared_mutex> getActiveLock()
    {
        return FrameMemory::getInstance().activateFrame(this);
    }

    /*
     * ==================================================================================
     * Pose & scale info (relative to frame)
     */
    FramePoseStruct* pose;
    Sim3 getScaledCamToWorld(int /*num*/ = 0) { return pose->getCamToWorld(); }
    bool hasTrackingParent() { return pose->trackingParent != nullptr; }
    Frame* getTrackingParent() { return pose->trackingParent->frame; }

    Sim3 lastConstraintTrackedCamToWorld;

    /** Adjacent frames in graph (empty for non-keyframes). */
    std::unordered_set<Frame*> neighbors;

    /** Keyframes for which tracking failed with an initialization transform. */
    std::unordered_multimap<Frame*, Sim3> trackingFailed;

    // Flag set when depth is updated.
    bool depthHasBeenUpdatedFlag = false;

    // Relocalization / re-keyframe positioning reference (kept in memory).
    std::mutex permaRef_mutex;
    Eigen::Vector3f*  permaRef_posData   = nullptr; // (x,y,z)
    Eigen::Vector2f*  permaRef_colorAndVarData = nullptr; // (I, Var)
    int               permaRefNumPts = 0;

    // Temporary values
    int   referenceID        = -1;
    int   referenceLevel     = 0;
    float distSquared        = 0.f;
    Eigen::Matrix3f K_otherToThis_R;
    Eigen::Vector3f K_otherToThis_t;
    Eigen::Vector3f otherToThis_t;
    Eigen::Vector3f K_thisToOther_t;
    Eigen::Matrix3f thisToOther_R;
    Eigen::Vector3f otherToThis_R_row0;
    Eigen::Vector3f otherToThis_R_row1;
    Eigen::Vector3f otherToThis_R_row2;
    Eigen::Vector3f thisToOther_t;

    // statistics
    float initialTrackedResidual = 0.f;
    int   numFramesTrackedOnThis = 0;
    int   numMappedOnThis        = 0;
    int   numMappedOnThisTotal   = 0;
    float meanIdepth             = 0.f;
    int   numPoints              = 0;
    int   idxInKeyframes         = -1;
    float edgeErrorSum           = 0.f;
    float edgesNum               = 0.f;
    int   numMappablePixels      = 0;
    float meanInformation        = 0.f;

private:
    void require(int dataFlags, int level = 0);
    void release(int dataFlags, bool pyramidsOnly, bool invalidateOnly);

    void initialize(int id, int width, int height, const Eigen::Matrix3f& K,
                    double timestamp);
    void setDepth_Allocate();

    void buildImage(int level);
    void releaseImage(int level);

    void buildGradients(int level);
    void releaseGradients(int level);

    void buildMaxGradients(int level);
    void releaseMaxGradients(int level);

    void buildIDepthAndIDepthVar(int level);
    void releaseIDepth(int level);
    void releaseIDepthVar(int level);

    void printfAssert(const char* message) const;

    struct Data {
        int id = -1;

        int width[PYRAMID_LEVELS]{}, height[PYRAMID_LEVELS]{};

        Eigen::Matrix3f K[PYRAMID_LEVELS], KInv[PYRAMID_LEVELS];
        float fx[PYRAMID_LEVELS]{}, fy[PYRAMID_LEVELS]{}, cx[PYRAMID_LEVELS]{},
              cy[PYRAMID_LEVELS]{};
        float fxInv[PYRAMID_LEVELS]{}, fyInv[PYRAMID_LEVELS]{}, cxInv[PYRAMID_LEVELS]{},
              cyInv[PYRAMID_LEVELS]{};

        double timestamp = 0.0;

        float*          image[PYRAMID_LEVELS]{};
        bool            imageValid[PYRAMID_LEVELS]{};

        Eigen::Vector4f* gradients[PYRAMID_LEVELS]{};
        bool             gradientsValid[PYRAMID_LEVELS]{};

        float* maxGradients[PYRAMID_LEVELS]{};
        bool   maxGradientsValid[PYRAMID_LEVELS]{};

        bool hasIDepthBeenSet = false;

        // Negative idepth allowed; validity if idepthVar[i] > 0.
        float* idepth[PYRAMID_LEVELS]{};
        bool   idepthValid[PYRAMID_LEVELS]{};

        // MUST contain -1 for invalid pixels!
        float* idepthVar[PYRAMID_LEVELS]{};
        bool   idepthVarValid[PYRAMID_LEVELS]{};

        // Re-activation data (minimal representation).
        unsigned char* validity_reAct = nullptr;
        float*         idepth_reAct   = nullptr;
        float*         idepthVar_reAct= nullptr;
        bool           reActivationDataValid = false;

        // From initial tracking; deleted once mapped.
        bool* refPixelWasGood = nullptr;
    };
    Data data;

    // Prevent concurrent builders for the same frame data.
    std::mutex buildMutex;

    // Shared/exclusive locking for minimizing frame memory.
    std::shared_mutex activeMutex;
    bool isActive = false;

    /** Releases everything that can be recalculated while keeping a minimal
      * representation in memory. ONLY CALL if an exclusive lock on activeMutex is owned! */
    bool minimizeInMemory();
};

/* -------------------- Inline definitions -------------------- */

inline int Frame::id() const { return data.id; }
inline int Frame::width(int level) const { return data.width[level]; }
inline int Frame::height(int level) const { return data.height[level]; }

inline const Eigen::Matrix3f& Frame::K(int level) const    { return data.K[level]; }
inline const Eigen::Matrix3f& Frame::KInv(int level) const { return data.KInv[level]; }
inline float Frame::fx(int level) const    { return data.fx[level]; }
inline float Frame::fy(int level) const    { return data.fy[level]; }
inline float Frame::cx(int level) const    { return data.cx[level]; }
inline float Frame::cy(int level) const    { return data.cy[level]; }
inline float Frame::fxInv(int level) const { return data.fxInv[level]; }
inline float Frame::fyInv(int level) const { return data.fyInv[level]; }
inline float Frame::cxInv(int level) const { return data.cxInv[level]; }
inline float Frame::cyInv(int level) const { return data.cyInv[level]; }

inline double Frame::timestamp() const { return data.timestamp; }

inline float* Frame::image(int level)
{
    if (!data.imageValid[level]) require(IMAGE, level);
    return data.image[level];
}
inline const Eigen::Vector4f* Frame::gradients(int level)
{
    if (!data.gradientsValid[level]) require(GRADIENTS, level);
    return data.gradients[level];
}
inline const float* Frame::maxGradients(int level)
{
    if (!data.maxGradientsValid[level]) require(MAX_GRADIENTS, level);
    return data.maxGradients[level];
}
inline bool Frame::hasIDepthBeenSet() const { return data.hasIDepthBeenSet; }

inline const float* Frame::idepth(int level)
{
    if (!data.hasIDepthBeenSet)
    {
        printfAssert("Frame::idepth(): idepth has not been set yet!");
        return nullptr;
    }
    if (!data.idepthValid[level]) require(IDEPTH, level);
    return data.idepth[level];
}
inline const unsigned char* Frame::validity_reAct()
{
    if (!data.reActivationDataValid) return nullptr;
    return data.validity_reAct;
}
inline const float* Frame::idepth_reAct()
{
    if (!data.reActivationDataValid) return nullptr;
    return data.idepth_reAct;
}
inline const float* Frame::idepthVar_reAct()
{
    if (!data.reActivationDataValid) return nullptr;
    return data.idepthVar_reAct;
}

inline const float* Frame::idepthVar(int level) {
    if (!data.hasIDepthBeenSet) {
        printfAssert("Frame::idepthVar(): idepth has not been set yet!");
        return nullptr;
    }
    if (!data.idepthVarValid[level]) require(IDEPTH_VAR, level);
    return data.idepthVar[level];
}

inline bool* Frame::refPixelWasGood() {
    if (data.refPixelWasGood == nullptr) {
        std::unique_lock<std::mutex> lock2(buildMutex);
        if (data.refPixelWasGood == nullptr) {
            const int w = data.width[SE3TRACKING_MIN_LEVEL];
            const int h = data.height[SE3TRACKING_MIN_LEVEL];
            data.refPixelWasGood = static_cast<bool*>(
                FrameMemory::getInstance().getBuffer(sizeof(bool) * static_cast<std::size_t>(w) * static_cast<std::size_t>(h))
            );
            // Keep original behavior: set all bytes to 0xFF (true)
            std::memset(data.refPixelWasGood, 0xFF, sizeof(bool) * static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        }
    }
    return data.refPixelWasGood;
}

inline bool* Frame::refPixelWasGoodNoCreate() { return data.refPixelWasGood; }

inline void Frame::clear_refPixelWasGood() {
    if (data.refPixelWasGood) {
        FrameMemory::getInstance().returnBuffer(static_cast<void*>(data.refPixelWasGood));
        data.refPixelWasGood = nullptr;
    }
}

} // namespace lsd_slam
