#include "frame_memory.h"
#include "frame.h"
#include <memory>
#include <Eigen/Core>
#include <mutex>
#include <shared_mutex>
#include <cstddef>


/*


singleton pool of aligned buffers
- hand out eigen-aligned chunks
active frame tracking and locking

* reuses large float/byte buffers(images, depth, var, grad) to avoid malloc/free churn
* tracks which frames are active so memory minimiziation doesn't race with readers


*/

namespace lsd_slam {

FrameMemory& FrameMemory::getInstance() {
    static FrameMemory instance;
    return instance;
}

void FrameMemory::releaseBuffers()
{
    std::unique_lock<std::mutex> lock(accessMutex);
    std::size_t total = 0;

    for(auto &kv : availableBuffers) {
      auto size = kv.first;
      auto &vec = kv.second;
      if(printMemoryDebugInfo) std::printf("deleting %ld buffers of size %d!\n", vec.size(), size);

      total += vec.size() * size;

      for(void *ptr : vec) {
          Eigen::internal::aligned_free(ptr);
          bufferSizes.erase(ptr);
      }

      vec.clear();
    }

    availableBuffers.clear();

    if(printMemoryDebugInfo)
      std::printf("released %.1f MB!\n", static_cast<double>(total) / 1000000.0f);
}


void* FrameMemory::getBuffer(std::size_t sizeInByte) {
    std::unique_lock<std::mutex> lock(accessMutex);
    auto it = availableBuffers.find(sizeInByte);

    if (it != availableBuffers.end()) {
        auto& availableOfSize = it->second;
        if (!availableOfSize.empty()) {
            void* buffer = availableOfSize.back();
            availableOfSize.pop_back();
            return buffer;
        }
        // fallthrough -> allocate
    }

    void *buffer = allocateBuffer(sizeInByte);
    return buffer;
}

float* FrameMemory::getFloatBuffer(std::size_t size) {
    return static_cast<float*>(getBuffer(sizeof(float) * size));
}

void FrameMemory::returnBuffer(void* buffer) {
    if(buffer == nullptr) return;
    std::unique_lock<std::mutex> lock(accessMutex);

    const auto it = bufferSizes.find(buffer);
    if(it == bufferSizes.end()) {
      Eigen::internal::aligned_free(buffer);
      return;
    }

    const std::size_t size = it->second;
    availableBuffers[size].push_back(buffer);
}

void* FrameMemory::allocateBuffer(std::size_t size) {
  void* buffer = Eigen::internal::aligned_malloc(size);
  if (!buffer) throw std::bad_alloc();
  bufferSizes.emplace(buffer, size);
  return buffer;
}

std::shared_lock<std::shared_mutex> FrameMemory::activateFrame(Frame* frame) {
    std::unique_lock<std::mutex> lock(activeFramesMutex);
    if(frame->isActive) activeFrames.remove(frame);
    activeFrames.push_front(frame);
    frame->isActive = true;
    return std::shared_lock<std::shared_mutex>(frame->activeMutex);
}
void FrameMemory::deactivateFrame(Frame* frame) {
    std::unique_lock<std::mutex> lock(activeFramesMutex);
    if(!frame->isActive) return;
    activeFrames.remove(frame);
    while(!frame->minimizeInMemory()) std::printf("cannot deactivateFrame frame %d, as some acvite-lock is lingering. May cause deadlock!\n", frame->id());	// do it in a loop, to make shure it is really, really deactivated.
    frame->isActive = false;
}

void FrameMemory::pruneActiveFrames() {
  std::unique_lock<std::mutex> lock(activeFramesMutex);
  while(static_cast<int>(activeFrames.size())> maxLoopClosureCandidates + 20) {
      Frame* tail = activeFrames.back();

      for(int attempt=0; attempt<2; ++attempt) {
        if(tail->minimizeInMemory()) break;
        if(attempt==1) { std::printf("failed to minimize frame %d twice. maybe some active-lock is lingering?\n", tail->id()); return; }
      }

      tail->isActive = false;
      activeFrames.pop_back();
  }
}

}