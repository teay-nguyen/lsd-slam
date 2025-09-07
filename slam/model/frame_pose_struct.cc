

#include "frame_pose_struct.h"
#include "frame.h"

/*

FramePoseStruct is the pose container for every Frame in lsd slam.
it provides representation, update, caching schemes for the absolute camera pose
sim(3) transform of each fraem

* tracking/ local odometry: when a frame is created, its pose is defined relative
  to a tracking parent frame (usually the last tracked keyframe).
* global optim: when keyframes are inserted into the sim(3) pose graph (via g2o)
  the optimized absolute pose is later written back into the FramePoseStruct


each frame owns a frame pose struct


*/

namespace lsd_slam {

int FramePoseStruct::cacheValidCounter = 0;

int privateFramePoseStructAllocCount = 0;

FramePoseStruct::FramePoseStruct(Frame* f)
    : trackingParent(nullptr),
      thisToParent_raw(Sim3()),
      frameID(f ? f->id() : -1),
      frame(f),
      isRegisteredToGraph(false),
      isOptimized(false),
      isInGraph(false),
      graphVertex(nullptr),
      cacheValidFor(-1),
      camToWorld(Sim3()),
      camToWorld_new(Sim3()),
      hasUnmergedPose(false)
{
    ++privateFramePoseStructAllocCount;
    if (enablePrintDebugInfo && printMemoryDebugInfo) {
        std::printf("ALLOCATED pose %d, now there are %d\n",
                    frameID, privateFramePoseStructAllocCount);
    }
}


FramePoseStruct::~FramePoseStruct()
{
    --privateFramePoseStructAllocCount;
    if (enablePrintDebugInfo && printMemoryDebugInfo) {
        std::printf("DELETED pose %d, now there are %d\n",
                    frameID, privateFramePoseStructAllocCount);
    }
}


void FramePoseStruct::setPoseGraphOptResult(const Sim3& p_camToWorld)
{
    // Only keyframes that are actually inserted into the graph can receive optimized poses.
    if(!isInGraph) return;

    camToWorld_new = p_camToWorld;
    hasUnmergedPose = true;
}

void FramePoseStruct::applyPoseGraphOptResult()
{
    if(!hasUnmergedPose) return;

    camToWorld = camToWorld_new;
    isOptimized = true;
    hasUnmergedPose = false;
    cacheValidCounter++;
}
void FramePoseStruct::invalidateCache()
{
    cacheValidFor = -1;
}
Sim3 FramePoseStruct::getCamToWorld(int recursionDepth)
{
    // prevent stack overflow
    assert(recursionDepth < 5000);

    // if the node is in the graph, it's absolute pose is only changed by optimization.
    // If this node has already been optimized, its absolute pose is authoritative.
    if(isOptimized) return camToWorld;


    // return cached pose, if still valid.
    // If our cached absolute pose is valid for the current epoch, reuse it.
    if(cacheValidFor == cacheValidCounter) return camToWorld;

    // return id if there is no parent (very first frame)
    // If there is no parent, this is the (tracking) root: identity in world.
    if(trackingParent == nullptr) {
      camToWorld = Sim3();
      cacheValidFor = cacheValidCounter;
      return camToWorld;
    }


    // Otherwise, compose from parent's absolute pose and our relative transform,
    // then cache for the current epoch.
    cacheValidFor = cacheValidCounter;
    return camToWorld = trackingParent->getCamToWorld(recursionDepth+1) * thisToParent_raw;
}

}