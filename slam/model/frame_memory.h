#pragma once
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <cstddef> 

namespace lsd_slam {

/** Singleton class for re-using buffers in the Frame class. */
class Frame;
class FrameMemory {
public:
  FrameMemory(const FrameMemory&) = delete;
  FrameMemory& operator=(const FrameMemory&) = delete;
  FrameMemory(FrameMemory&&) = delete;
  FrameMemory& operator=(FrameMemory&&) = delete;

  /** Returns the global instance. Creates it when the method is first called. */
  static FrameMemory& getInstance();

  /** Allocates or fetches a buffer with length: size * sizeof(float).
    * Corresponds to "buffer = new float[size]". */
  float* getFloatBuffer(std::size_t size);

  /** Allocates or fetches a buffer with length: size * sizeof(float).
    * Corresponds to "buffer = new float[size]". */
  void* getBuffer(std::size_t sizeInByte);

  /** Returns an allocated buffer back to the global storage for re-use.
    * Corresponds to "delete[] buffer". */
  void returnBuffer(void* buffer);

  std::shared_lock<std::shared_mutex> activateFrame(Frame* frame);
  void deactivateFrame(Frame* frame);
  void pruneActiveFrames();

  void releaseBuffers();
private:
    FrameMemory() = default;
    void* allocateBuffer(std::size_t sizeInByte);

    std::mutex accessMutex;
    std::unordered_map< void*, unsigned int > bufferSizes;
    std::unordered_map< unsigned int, std::vector< void* > > availableBuffers;

    std::mutex activeFramesMutex;
    std::list<Frame*> activeFrames;
};

}