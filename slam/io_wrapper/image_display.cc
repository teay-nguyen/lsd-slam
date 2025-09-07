#include "image_display.h"

#include <opencv2/highgui.hpp>   // modern include path
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace lsd_slam {
namespace Util {

// Toggle to use the background display thread or show immediately on the caller's thread.
static constexpr bool kUseImageDisplayThread = false;

// State guarded by g_displayMutex
static std::unordered_set<std::string> g_openWindows;
static std::deque<DisplayImageObject>  g_displayQueue;

static std::mutex              g_displayMutex;
static std::condition_variable g_displayCv;
static std::unique_ptr<std::thread> g_displayThread;
static std::atomic<bool>       g_keepRunning{false};

static void displayThreadLoop()
{
    std::printf("started image display thread!\n");

    std::unique_lock<std::mutex> lock(g_displayMutex);
    while (g_keepRunning.load(std::memory_order_acquire)) {
        // Wait until there's work or we are asked to stop
        g_displayCv.wait(lock, [] {
            return !g_keepRunning.load(std::memory_order_acquire) || !g_displayQueue.empty();
        });
        if (!g_keepRunning.load(std::memory_order_acquire))
            break;

        // Process all queued display requests
        while (!g_displayQueue.empty()) {
            DisplayImageObject obj = std::move(g_displayQueue.front());
            g_displayQueue.pop_front();

            // We keep the lock while creating/showing windows to mirror original behavior.
            if (!obj.autoSize) {
                if (g_openWindows.find(obj.name) == g_openWindows.end()) {
                    cv::namedWindow(obj.name, cv::WINDOW_NORMAL);
                    cv::resizeWindow(obj.name, obj.img.cols, obj.img.rows);
                    g_openWindows.insert(obj.name);
                }
            }
            cv::imshow(obj.name, obj.img);
        }
        // loop back to wait for more
    }

    // Clean up windows on thread exit
    cv::destroyAllWindows();
    g_openWindows.clear();

    std::printf("ended image display thread!\n");
}

static void ensureDisplayThread()
{
    if (!g_displayThread) {
        g_keepRunning.store(true, std::memory_order_release);
        g_displayThread = std::make_unique<std::thread>(&displayThreadLoop);
    }
}

void displayImage(const char* windowName, const cv::Mat& image, bool autoSize)
{
    if (kUseImageDisplayThread) {
        ensureDisplayThread();

        // Enqueue a copy (clone image to keep lifetime safe)
        {
            std::lock_guard<std::mutex> lk(g_displayMutex);
            DisplayImageObject obj;
            obj.autoSize = autoSize;
            obj.img      = image.clone();
            obj.name     = windowName ? windowName : "window";
            g_displayQueue.emplace_back(std::move(obj));
        }
        g_displayCv.notify_one();
    } else {
        // Immediate display on caller's thread; track window sizing once
        if (!autoSize) {
            if (g_openWindows.find(windowName) == g_openWindows.end()) {
                cv::namedWindow(windowName, cv::WINDOW_NORMAL);
                cv::resizeWindow(windowName, image.cols, image.rows);
                g_openWindows.insert(windowName);
            }
        }
        cv::imshow(windowName, image);
    }
    // Intentionally not calling waitKey(1) here to match original behavior.
}

int waitKey(int milliseconds)
{
    return cv::waitKey(milliseconds);
}

int waitKeyNoConsume(int milliseconds)
{
    // Cannot implement this with OpenCV functions (same as original).
    return cv::waitKey(milliseconds);
}

void closeAllWindows()
{
    if (kUseImageDisplayThread) {
        std::unique_ptr<std::thread> toJoin;
        {
            std::lock_guard<std::mutex> lk(g_displayMutex);
            if (g_displayThread) {
                g_keepRunning.store(false, std::memory_order_release);
                g_displayCv.notify_all();
                toJoin = std::move(g_displayThread);
            }
        }
        if (toJoin) {
            std::printf("waiting for image display thread to end!\n");
            toJoin->join();
            std::printf("done waiting for image display thread to end!\n");
        }
    } else {
        cv::destroyAllWindows();
        g_openWindows.clear();
    }
}

} // namespace Util
} // namespace lsd_slam
