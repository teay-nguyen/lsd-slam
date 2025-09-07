#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <shared_mutex>

#include "../util/eigen_core_include.h"
#include "../util/sophus_util.h"

#include <g2o/core/sparse_optimizer.h>   // g2o::SparseOptimizer
#include <g2o/core/robust_kernel.h>      // g2o::RobustKernel

/*

owns the global sim(3) pose graph (g2o) over keyframes and constraints:
* adds frames/keyframes
* adds sim(3) constraints (odometry and loop closures)
* runs batch optim
* provides graph-distance queries for scheduling/relocalization

*/


namespace lsd_slam {

class Frame;
class KeyFrameGraph;
class VertexSim3;
class EdgeSim3;
class FramePoseStruct;

/** Pairwise keyframe constraint stored alongside the g2o edge. */
struct KFConstraintStruct
{
    KFConstraintStruct() noexcept
        : firstFrame(nullptr),
          secondFrame(nullptr),
          robustKernel(nullptr),
          edge(nullptr),
          usage(0.f),
          meanResidualD(0.f),
          meanResidualP(0.f),
          meanResidual(0.f),
          reciprocalConsistency(0.f),
          idxInAllEdges(-1)
    {
        information.setZero();
    }

    ~KFConstraintStruct();

    Frame* firstFrame;                     // not owning
    Frame* secondFrame;                    // not owning
    Sophus::Sim3d secondToFirst;           // relative pose (j -> i)
    Eigen::Matrix<double, 7, 7> information;

    g2o::RobustKernel* robustKernel;       // owned by g2o edge/optimizer
    EdgeSim3* edge;                        // owned by g2o optimizer

    float usage;
    float meanResidualD;
    float meanResidualP;
    float meanResidual;

    float reciprocalConsistency;

    int idxInAllEdges;                     // position in edgesAll
};

/**
 * Graph consisting of KeyFrames and constraints, performing optimization.
 */
class KeyFrameGraph
{
    friend class IntegrationTest;

public:
    /** Constructs an empty pose graph. */
    KeyFrameGraph();

    /** Deletes the g2o graph. */
    ~KeyFrameGraph();

    /** Adds a new KeyFrame to the graph. */
    void addKeyFrame(Frame* frame);

    /** Adds a new Frame to the graph (stores only its pose struct). */
    void addFrame(Frame* frame);

    void dumpMap(std::string folder);

    /**
     * Adds a new constraint to the graph.
     *
     * The transformation must map world points such that they move as if
     * attached to a frame which moves from firstFrame to secondFrame:
     *   second->camToWorld * first->worldToCam * point
     *
     * If isOdometryConstraint is set, scaleInformation is ignored.
     */
    void insertConstraint(KFConstraintStruct* constraint);

    /**
     * Optimizes the graph. Does not update the keyframe poses,
     * only the vertex poses. Call updateKeyFramePoses() afterwards.
     */
    int  optimize(int num_iterations);
    bool addElementsFromBuffer();

    /** Creates a hash map of keyframe -> distance (hops) to the given frame. */
    void calculateGraphDistancesToFrame(
        Frame* frame,
        std::unordered_map<Frame*, int>* distanceMap);

    // Stats
    int totalPoints  = 0;
    int totalEdges   = 0;
    int totalVertices= 0;

    // =========================== Keyframe & Pose Lists/Maps ============================
    // Always lock with the corresponding mutex before touching these.

    // All finished keyframes (does not include the one currently being built).
    std::shared_mutex keyframesAllMutex;
    std::vector<Frame*> keyframesAll;                  // non-owning

    /** Maps frame id -> keyframe. Contains ALL keyframes ever allocated
     * (including the one currently being built). Holds shared_ptrs to keep them alive. */
    std::shared_mutex idToKeyFrameMutex;
    std::unordered_map<int, std::shared_ptr<Frame>> idToKeyFrame; // BFS over keyframes

    // All constraints/edges once created.
    std::shared_mutex edgesListsMutex;
    std::vector<KFConstraintStruct*> edgesAll;         // non-owning; lifetime managed externally/g2o

    // All frame poses, chronological, as they are tracked (frame may be removed later).
    // These are referenced by the owning Frame/Keyframe as well.
    std::shared_mutex allFramePosesMutex;
    std::vector<FramePoseStruct*> allFramePoses;       // non-owning

    // Keyframes considered for re-tracking (arbitrary order).
    // Re-tracked frames get pushed to the back; sampling often takes from the front third.
    std::mutex keyframesForRetrackMutex;
    std::deque<Frame*> keyframesForRetrack;            // non-owning

private:
    /** Pose graph (g2o) */
    g2o::SparseOptimizer graph;

    // Staging buffers for concurrent producers
    std::vector<Frame*>            newKeyframesBuffer; // non-owning
    std::vector<KFConstraintStruct*> newEdgeBuffer;    // non-owning

    int nextEdgeId = 0;

    // Non-copyable / non-movable (optimizer holds internal pointers)
    KeyFrameGraph(const KeyFrameGraph&) = delete;
    KeyFrameGraph& operator=(const KeyFrameGraph&) = delete;
    KeyFrameGraph(KeyFrameGraph&&) = delete;
    KeyFrameGraph& operator=(KeyFrameGraph&&) = delete;
};

} // namespace lsd_slam
