
#pragma once
#include "../util/sophus_util.h"

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_six_dof_expmap.h"


namespace lsd_slam {

class VertexSim3 final : public g2o::BaseVertex<7, Sophus::Sim3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSim3() noexcept : _fix_scale(false) {}
    bool read(std::istream&) override { return false; };
    bool write(std::ostream&) const override { return false; };

    void setToOriginImpl() override {
        _estimate = Sophus::Sim3d();
    }

    void oplusImpl(const double* update_raw) override {
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> update(update_raw);
        Eigen::Matrix<double, 7, 1> update_mod = update;
        if (_fix_scale) update_mod[6] = 0.0f;
        setEstimate(Sophus::Sim3d::exp(update_mod) * estimate());
    }

    bool _fix_scale;
};

/**
* \brief 7D edge between two Vertex7
*/
class EdgeSim3 final : public g2o::BaseBinaryEdge<7, Sophus::Sim3d, VertexSim3, VertexSim3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3() noexcept : _inverseMeasurement(Sophus::Sim3d()) {};

    bool read(std::istream&) override { return false; };
    bool write(std::ostream&) const override { return false; };

    void computeError() override {
        const VertexSim3* v_from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* v_to = static_cast<const VertexSim3*>(_vertices[1]);
        Sophus::Sim3d error_group = v_from->estimate().inverse() * v_to->estimate() * _inverseMeasurement;
        _error = error_group.log();
    }

    void linearizeOplus() override {
        const VertexSim3* v_from = static_cast<const VertexSim3*>(_vertices[0]);
        _jacobianOplusXj = v_from->estimate().inverse().Adj();
        _jacobianOplusXi = -_jacobianOplusXj;
    }


    void setMeasurement(const Sophus::Sim3d& m) override {
        _measurement = m;
        _inverseMeasurement = m.inverse();
    }

    bool setMeasurementData(const double* m) override {
        Eigen::Map<const g2o::Vector7> v(m);
        setMeasurement(Sophus::Sim3d::exp(v));
        return true;
    }

    bool setMeasurementFromState() override {
        const VertexSim3* v_from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* v_to   = static_cast<const VertexSim3*>(_vertices[1]);
        Sophus::Sim3d delta = v_from->estimate().inverse() * v_to->estimate();
        setMeasurement(delta);
        return true;
    }

    double initialEstimatePossible(const g2o::OptimizableGraph::VertexSet&, g2o::OptimizableGraph::Vertex*) override {
        return 1.;
    }

    void initialEstimate(const g2o::OptimizableGraph::VertexSet& from, g2o::OptimizableGraph::Vertex* /*to*/) override {
        VertexSim3 *v_from = static_cast<VertexSim3*>(_vertices[0]);
        VertexSim3 *v_to   = static_cast<VertexSim3*>(_vertices[1]);
        if (from.count(v_from) > 0) v_to->setEstimate(v_from->estimate() * _measurement);
        else v_from->setEstimate(v_to->estimate() * _inverseMeasurement);
    }

protected:
    Sophus::Sim3d _inverseMeasurement;
};

}
