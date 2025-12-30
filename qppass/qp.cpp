#include "qp.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <optional>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

void Qp_Workspace::reset() {}

class NotImplementedError : public std::exception {
  std::string message_;

public:
  explicit NotImplementedError(const std::string &msg) : message_(msg) {}

  const char *what() const noexcept override { return message_.c_str(); }
};

void Qp_Workspace::allocate(size_t batch_size, size_t cost_dim, size_t eq_dim,
                            size_t n_threads, size_t strategy,
                            size_t ineq_dim) {
  eq_dim = 0;
  if (strategy_ != strategy || batch_size != batch_size_ ||
      cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
    strategy_ = strategy;
    batch_size_ = batch_size;
    cost_dim_ = cost_dim;
    eq_dim_ = eq_dim;
    n_threads_ = n_threads;

    double eps_abs = 1e-10;
    double eps_rel = 1e-10;
    qp.resize(batch_size);

    grad_rhs_mem_.clear();
    grad_rhs_mem_.reserve(batch_size);
    grad_KKT_mem_.clear();
    grad_KKT_mem_.reserve(batch_size);
    grad_G_mem_.clear();
    grad_G_mem_.reserve(batch_size);
    grad_ub_mem_.clear();
    grad_ub_mem_.reserve(batch_size);
    grad_lb_mem_.clear();
    grad_lb_mem_.reserve(batch_size);
    warm_start_x.resize(batch_size, std::nullopt);
    warm_start_eq.resize(batch_size, std::nullopt);
    warm_start_neq.resize(batch_size, std::nullopt);
    for (size_t i = 0; i < batch_size; ++i) {
      if (strategy == 2) {
        qp[i].emplace(cost_dim, 0, 0);

      } else if (strategy == 3) {
        qp[i].emplace(cost_dim, eq_dim, ineq_dim);
      }
      qp[i]->settings.eps_abs = eps_abs;
      qp[i]->settings.eps_rel = eps_rel;
      qp[i]->settings.primal_infeasibility_solving = false;
      qp[i]->settings.verbose = false;

      grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
      grad_rhs_mem_[i].setZero();
      grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
      grad_KKT_mem_[i].setZero();
      grad_G_mem_.emplace_back(ineq_dim, cost_dim);
      grad_G_mem_[i].setZero();
      grad_lb_mem_.emplace_back(ineq_dim);
      grad_lb_mem_[i].setZero();
      grad_ub_mem_.emplace_back(ineq_dim);
      grad_ub_mem_[i].setZero();
    }
    identity.clear();
    lb.clear();
    ub.clear();
    output.clear();

    identity.reserve(n_threads);
    lb.reserve(n_threads);
    ub.reserve(n_threads);
    output.reserve(n_threads);

    for (size_t i = 0; i < n_threads; ++i) {
      identity.emplace_back(Eigen::MatrixXd::Identity(cost_dim, cost_dim));
      lb.emplace_back(Eigen::VectorXd::Constant(cost_dim, bound));
      ub.emplace_back(Eigen::VectorXd::Constant(cost_dim, -bound));
      output.emplace_back(2 * cost_dim + eq_dim);
    }
  }
}

Eigen::Ref<Eigen::Vector<double, Eigen::Dynamic>>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   [[maybe_unused]] Eigen::Ref<const Eigen::MatrixXd> A,
   [[maybe_unused]] Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, size_t thread_id, size_t batch_position,
   std::optional<Eigen::Ref<const Eigen::MatrixXd>> G_,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> lb_,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> ub_) {
  if (workspace.strategy_ == 2) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    Eigen::VectorXd &output = workspace.output[thread_id];

    workspace.qp[batch_position]->init(Q, p, proxsuite::nullopt,
                                       proxsuite::nullopt, proxsuite::nullopt,
                                       proxsuite::nullopt, proxsuite::nullopt);

    workspace.qp[batch_position]->solve(workspace.warm_start_x[batch_position],
                                        proxsuite::nullopt, proxsuite::nullopt);
    output = workspace.qp[batch_position]->results.x;
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    return output;

  } else if (workspace.strategy_ == 3 && G_.has_value()) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    Eigen::VectorXd &output = workspace.output[thread_id];
    workspace.qp[batch_position]->init(Q, p, proxsuite::nullopt,
                                       proxsuite::nullopt, G_, lb_, ub_);
    workspace.qp[batch_position]->solve(
        workspace.warm_start_x[batch_position], proxsuite::nullopt,
        workspace.warm_start_neq[batch_position]);
    output = workspace.qp[batch_position]->results.x;
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    return output;
  } else if (workspace.strategy_ == 3 && !G_.has_value()) {
    throw "G has no values";
  }
  throw NotImplementedError("Strategy " + std::to_string(workspace.strategy_) +
                            " not implemented");
}

void QP_backward(Qp_Workspace &workspace,
                 Eigen::Ref<Eigen::VectorXd> grad_output,
                 size_t batch_position) {
  size_t cost_dim = workspace.cost_dim_;
  if (workspace.strategy_ == 2) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    proxsuite::proxqp::dense::compute_backward<double>(
        *workspace.qp[batch_position], grad_output, 1e-10, 1e-10, 1e-10);
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    workspace.grad_KKT_mem_[batch_position].setZero();
    workspace.grad_rhs_mem_[batch_position].setZero();
    workspace.grad_KKT_mem_[batch_position].topLeftCorner(cost_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dH;
    workspace.grad_rhs_mem_[batch_position].head(cost_dim) =
        -workspace.qp[batch_position]->model.backward_data.dL_dg;
  } else if (workspace.strategy_ == 3) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    proxsuite::proxqp::dense::compute_backward<double>(
        *workspace.qp[batch_position], grad_output, 1e-10, 1e-10, 1e-10);
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    workspace.grad_KKT_mem_[batch_position].setZero();
    workspace.grad_rhs_mem_[batch_position].setZero();
    workspace.grad_KKT_mem_[batch_position].topLeftCorner(cost_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dH;
    workspace.grad_rhs_mem_[batch_position].head(cost_dim) =
        -workspace.qp[batch_position]->model.backward_data.dL_dg;
    workspace.grad_G_mem_[batch_position] =
        workspace.qp[batch_position]->model.backward_data.dL_dC;
    workspace.grad_ub_mem_[batch_position] =
        workspace.qp[batch_position]->model.backward_data.dL_du;
    workspace.grad_lb_mem_[batch_position] =
        workspace.qp[batch_position]->model.backward_data.dL_dl;
  }
}
