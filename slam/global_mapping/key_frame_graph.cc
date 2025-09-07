#include "key_frame_graph.h"
#include "../model/frame.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

#include <g2o/types/sim3/sim3.h>
#include "../global_mapping/g2o_type_sim3_sophus.h"

#include "../io_wrapper/image_display.h"
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <memory>

#include "../util/fxn.h"
#include "../util/snprintf.h"

namespace fs = std::filesystem;


/*

unique lock is a RAII wrapper around a mutex
* gives exclusive ownership

shared mutex
* shared(reader) lock: multiple threads can hold a shared lock at the same time
* unique(writer) lock: only one thread can hold it, and it excludes all readers
* many threads can read, but only one thread can write


global sim(3) pose-graph assesmbly and optimization
* this moduel owns the global sim(3) pose graph built over keyframes.
* is the bridge between local, frame-level estimation (images, inverse-depth maps) and
  global consistency (loop closures, scale drift correction)

* stages new vertices/edges, injects them into g2o, runs batch optimization, and
  exports diagnostics.


batch optimization takes all the vertices and edges in the global pose grpah,
minimizes the total error across the graph simultaneously so this distributes
loop-closure corrections globally.

*/

namespace lsd_slam {


KFConstraintStruct::~KFConstraintStruct() {
    if(edge != nullptr) delete edge;
}

KeyFrameGraph::KeyFrameGraph() : nextEdgeId(0) {
    using BlockSolver = g2o::BlockSolver_7_3;
    using LinearSolver = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;

    auto solver = std::make_unique<LinearSolver>();
    auto blockSolver = std::make_unique<BlockSolver>(std::move(solver));
    auto algorithm = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(std::move(blockSolver));

    graph.setAlgorithm(algorithm.release());
    graph.setVerbose(false); // printOptimizationInfo

    // solver->setWriteDebug(true);
    // blockSolver->setWriteDebug(true);
    // algorithm->setWriteDebug(true);

    totalPoints = totalEdges = totalVertices = 0;
}

KeyFrameGraph::~KeyFrameGraph() {
    // deletes edges
    for (KFConstraintStruct* edge : newEdgeBuffer)
        delete edge;	// deletes the g2oedge, which deletes the kernel.

    // deletes keyframes (by deleting respective shared pointers).

    idToKeyFrame.clear();

    // deletes pose structs (their memory is managed by graph)
    // WARNING: at this point, all Frames have to be deleted, otherwise it night cause segfaults!
    
    {
      std::unique_lock<std::shared_mutex> lk(allFramePosesMutex);
      for(FramePoseStruct* p : allFramePoses)
          delete p;
      allFramePoses.clear();
    }
}


void KeyFrameGraph::addFrame(Frame* frame)
{

    frame->pose->isRegisteredToGraph = true;
    FramePoseStruct* pose = frame->pose;

    {
      std::unique_lock<std::shared_mutex> lk(allFramePosesMutex);
      allFramePoses.push_back(pose);
    }
}

void KeyFrameGraph::dumpMap(std::string folder)
{
    std::printf("DUMP MAP: dumping to %s\n", folder.c_str());

    std::vector<Frame*> kfSnapshot;
    {
        std::shared_lock<std::shared_mutex> lk(keyframesAllMutex);
        kfSnapshot = keyframesAll;
    }

    std::vector<KFConstraintStruct*> edgesSnapshot;
    {
        std::shared_lock<std::shared_mutex> lk(edgesListsMutex);
        edgesSnapshot = edgesAll;
    }

    try {
        fs::path out(folder);
        if (fs::exists(out)) {
            fs::remove_all(out);
        }
        fs::create_directories(out);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DUMP MAP: filesystem error for '%s': %s\n",
                     folder.c_str(), e.what());
        return;
    }


     char buf[256];
    for (std::size_t i = 0; i < kfSnapshot.size(); ++i)
    {
        Frame* f = kfSnapshot[i];
        std::snprintf(buf, sizeof(buf), "%s/depth-%zu.png", folder.c_str(), i);
        cv::imwrite(buf, getDepthRainbowPlot(f, 0));

        std::snprintf(buf, sizeof(buf), "%s/frame-%zu.png", folder.c_str(), i);
        cv::imwrite(buf, cv::Mat(f->height(), f->width(), CV_32F, f->image()));

        std::snprintf(buf, sizeof(buf), "%s/var-%zu.png", folder.c_str(), i);
        cv::imwrite(buf, getVarRedGreenPlot(f->idepthVar(), f->image(), f->width(), f->height()));
    }


    


    if (!kfSnapshot.empty())
    {
        std::size_t i = kfSnapshot.size() - 1;
        Util::displayImage("VAR PREVIEW",
                           getVarRedGreenPlot(kfSnapshot[i]->idepthVar(),
                                              kfSnapshot[i]->image(),
                                              kfSnapshot[i]->width(),
                                              kfSnapshot[i]->height()));
    }

    std::printf("DUMP MAP: dumped %zu depthmaps\n", kfSnapshot.size());

    Eigen::MatrixXf res, resD, resP, usage, consistency, distance, error;
    Eigen::VectorXf meanRootInformation, usedPixels;

    const int N = static_cast<int>(kfSnapshot.size());
    res.resize(N, N);          res.setZero();
    resD.resize(N, N);         resD.setZero();
    resP.resize(N, N);         resP.setZero();
    usage.resize(N, N);        usage.setZero();
    consistency.resize(N, N);  consistency.setZero();
    distance.resize(N, N);     distance.setZero();
    error.resize(N, N);        error.setZero();

    meanRootInformation.resize(N); meanRootInformation.setZero();
    usedPixels.resize(N);          usedPixels.setZero();

    for (int i = 0; i < N; ++i)
    {
        meanRootInformation[i] = kfSnapshot[i]->meanInformation;
        usedPixels[i] = (kfSnapshot[i]->numMappablePixels > 0)
                        ? (kfSnapshot[i]->numPoints / static_cast<float>(kfSnapshot[i]->numMappablePixels))
                        : 0.0f;
    }


    for (KFConstraintStruct* e : edgesSnapshot)
    {
        const int i = e->firstFrame->idxInKeyframes;
        const int j = e->secondFrame->idxInKeyframes;
        if (i < 0 || j < 0 || i >= N || j >= N) continue;

        res(i, j)        = e->meanResidual;
        resD(i, j)       = e->meanResidualD;
        resP(i, j)       = e->meanResidualP;
        usage(i, j)      = e->usage;
        consistency(i, j)= e->reciprocalConsistency;
        distance(i, j)   = static_cast<float>(e->secondToFirst.translation().norm());
        error(i, j)      = static_cast<float>(e->edge ? e->edge->chi2() : 0.0);
    }

    auto write_matrix = [&](const std::string& name, const auto& M)
    {
        std::ofstream ofs(fs::path(folder) / name);
        if (!ofs) {
            std::fprintf(stderr, "DUMP MAP: failed to open %s\n", name.c_str());
            return;
        }
        ofs << M;
    };

    write_matrix("residual.txt",            res);
    write_matrix("residualD.txt",           resD);
    write_matrix("residualP.txt",           resP);
    write_matrix("usage.txt",               usage);
    write_matrix("consistency.txt",         consistency);
    write_matrix("distance.txt",            distance);
    write_matrix("error.txt",               error);
    write_matrix("meanRootInformation.txt", meanRootInformation);
    write_matrix("usedPixels.txt",          usedPixels);

    std::printf("DUMP MAP: dumped %zu edges\n", edgesSnapshot.size());
}



void KeyFrameGraph::addKeyFrame(Frame* frame)
{
    if(frame->pose->graphVertex != nullptr)
        return;

    // Insert vertex into g2o graph
    VertexSim3* vertex = new VertexSim3();
    vertex->setId(frame->id());

    Sophus::Sim3d camToWorld_estimate = frame->getScaledCamToWorld();

    if(!frame->hasTrackingParent())
        vertex->setFixed(true);

    vertex->setEstimate(camToWorld_estimate);
    vertex->setMarginalized(false);

    frame->pose->graphVertex = vertex;

    newKeyframesBuffer.push_back(frame);

}

void KeyFrameGraph::insertConstraint(KFConstraintStruct* constraint)
{
    EdgeSim3* edge = new EdgeSim3();
    edge->setId(nextEdgeId++);
    totalEdges++;

    edge->setMeasurement(constraint->secondToFirst);
    edge->setInformation(constraint->information);
    edge->setRobustKernel(constraint->robustKernel);

    edge->resize(2);
    assert(constraint->firstFrame->pose->graphVertex != nullptr);
    assert(constraint->secondFrame->pose->graphVertex != nullptr);
    edge->setVertex(0, constraint->firstFrame->pose->graphVertex);
    edge->setVertex(1, constraint->secondFrame->pose->graphVertex);

    constraint->edge = edge;
    newEdgeBuffer.push_back(constraint);


    constraint->firstFrame->neighbors.insert(constraint->secondFrame);
    constraint->secondFrame->neighbors.insert(constraint->firstFrame);

    {
        std::unique_lock<std::shared_mutex> lk(edgesListsMutex);
        constraint->idxInAllEdges = static_cast<int>(edgesAll.size());
        edgesAll.push_back(constraint);
    }
}


bool KeyFrameGraph::addElementsFromBuffer()
{
    bool added = false;

    std::vector<Frame*> localNewKFs;
    localNewKFs.swap(newKeyframesBuffer);

    {
        std::lock_guard<std::mutex> lk(keyframesForRetrackMutex);
        for (Frame* newKF : localNewKFs)
        {
            graph.addVertex(newKF->pose->graphVertex);
            if (!newKF->pose->isInGraph)
            {
                newKF->pose->isInGraph = true;
                ++totalVertices;
            }
            keyframesForRetrack.push_back(newKF);
            added = true;
        }
    }

    std::vector<KFConstraintStruct*> localNewEdges;
    localNewEdges.swap(newEdgeBuffer);
    for (auto* e : localNewEdges)
    {
        graph.addEdge(e->edge);
        added = true;
    }

    return added;
}

int KeyFrameGraph::optimize(int num_iterations) {
    // Abort if graph is empty, g2o shows an error otherwise
    if (graph.edges().empty()) return 0;

    graph.setVerbose(false); // printOptimizationInfo
    graph.initializeOptimization();
    return graph.optimize(num_iterations, /*online*/ false);

}



void KeyFrameGraph::calculateGraphDistancesToFrame(Frame* startFrame, std::unordered_map<Frame*, int>* distanceMap)
{
    distanceMap->insert(std::make_pair(startFrame, 0));

    std::multimap< int, Frame* > priorityQueue;
    priorityQueue.insert(std::make_pair(0, startFrame));

    while (! priorityQueue.empty())
    {
        auto it = priorityQueue.begin();
        const int length = it->first;
        Frame* frame = it->second;
        priorityQueue.erase(it);

        auto mapEntry = distanceMap->find(frame);

        if (mapEntry != distanceMap->end() && length > mapEntry->second)
            continue;

        for (Frame* neighbor : frame->neighbors)
        {
            auto neighborMapEntry = distanceMap->find(neighbor);
            const int cand = length + 1;

            if (neighborMapEntry != distanceMap->end() && cand >= neighborMapEntry->second)
                continue;

            if (neighborMapEntry != distanceMap->end())
                neighborMapEntry->second = cand;
            else
                distanceMap->insert(std::make_pair(neighbor, cand));
            priorityQueue.insert(std::make_pair(cand, neighbor));
        }
    }
}

}
