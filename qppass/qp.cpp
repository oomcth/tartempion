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
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace pinocchio;
using namespace proxsuite;
using namespace proxsuite::proxqp;

void printConditionNumber(const Eigen::MatrixXd &A) {
  if (A.rows() == 0 || A.cols() == 0) {
    std::cerr << "Matrix is empty." << std::endl;
    return;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
  double sigma_max = svd.singularValues()(0);
  double sigma_min = svd.singularValues()(svd.singularValues().size() - 1);

  if (sigma_min == 0) {
    std::cout << "Matrix is singular or nearly singular. Condition number is "
                 "infinite."
              << std::endl;
  } else {
    double condition_number = sigma_max / sigma_min;
    std::cout << "Condition number: " << condition_number << std::endl;
  }
}

void Qp_Workspace::reset() {}

class NotImplementedError : public std::exception {
  std::string message_;

public:
  explicit NotImplementedError(const std::string &msg) : message_(msg) {}

  const char *what() const noexcept override { return message_.c_str(); }
};

void Qp_Workspace::allocate(int batch_size, int cost_dim, int eq_dim,
                            int n_threads, int strategy) {
  if (false && strategy == 0) {
    if (strategy_ != strategy || batch_size != batch_size_ ||
        cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
      strategy_ = strategy;
      batch_size_ = batch_size;
      cost_dim_ = cost_dim;
      eq_dim_ = eq_dim;
      n_threads_ = n_threads;

      kkt_mem_.clear();
      kkt_mem_.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        kkt_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
      }

      grad_KKT_mem_.clear();
      grad_KKT_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
      }

      grad_KKT_.clear();
      grad_KKT_.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        grad_KKT_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
      }

      grad_rhs_mem_.clear();
      grad_rhs_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
      }

      kkt_.clear();
      kkt_.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        kkt_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
      }

      rhs_.clear();
      rhs_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        rhs_.emplace_back(cost_dim + eq_dim);
      }

      sol_.clear();
      sol_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        sol_.emplace_back(cost_dim + eq_dim);
      }

      lu_.clear();
      lu_.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        lu_.emplace_back();
      }

      temp_vec_.clear();
      temp_vec_.resize(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        temp_vec_.emplace_back(cost_dim + eq_dim);
      }
    }
  }
  if (false && strategy == 1) {
    if (true || strategy_ != strategy || batch_size != batch_size_ ||
        cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
      strategy_ = strategy;
      batch_size_ = batch_size;
      cost_dim_ = cost_dim;
      eq_dim_ = eq_dim;
      n_threads_ = n_threads;

      double eps_abs = 1e-10;
      double eps_rel = 1e-10;
      qp.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        qp[i].emplace(cost_dim, eq_dim, cost_dim);
        qp[i]->settings.eps_abs = eps_abs;
        qp[i]->settings.eps_rel = eps_rel;
        qp[i]->settings.primal_infeasibility_solving = false;
        qp[i]->settings.eps_primal_inf = 1e-5;
        qp[i]->settings.eps_dual_inf = 1e-5;
        qp[i]->settings.verbose = false;
      }
      identity.clear();
      std::cout << n_threads << std::endl;
      identity.resize(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        identity[i] = Eigen::MatrixXd(cost_dim, cost_dim);
        identity[i].setZero();
        identity[i].diagonal().array() = 1;
      }

      grad_rhs_mem_.clear();
      grad_rhs_mem_.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
        grad_rhs_mem_[i].setZero();
      }

      grad_KKT_mem_.clear();
      grad_KKT_mem_.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
        grad_KKT_mem_[i].setZero();
      }
    }
  }
  if (strategy == 2 || strategy == 3) {
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
      for (int i = 0; i < batch_size; ++i) {
        if (strategy == 2) {
          qp[i].emplace(cost_dim, 0, 0);

        } else if (strategy == 3) {
          qp[i].emplace(cost_dim, 0, 1);
        }
        qp[i]->settings.eps_abs = eps_abs;
        qp[i]->settings.eps_rel = eps_rel;
        qp[i]->settings.primal_infeasibility_solving = false;
        qp[i]->settings.eps_primal_inf = 1e-4;
        qp[i]->settings.eps_dual_inf = 1e-4;
        qp[i]->settings.verbose = false;

        grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
        grad_rhs_mem_[i].setZero();
        grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
        grad_KKT_mem_[i].setZero();
        grad_G_mem_.emplace_back(cost_dim + 3, cost_dim);
        grad_G_mem_[i].setZero();
        grad_lb_mem_.emplace_back(cost_dim);
        grad_lb_mem_[i].setZero();
        grad_ub_mem_.emplace_back(cost_dim);
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

      for (int i = 0; i < n_threads; ++i) {
        identity.emplace_back(Eigen::MatrixXd::Identity(cost_dim, cost_dim));
        lb.emplace_back(Eigen::VectorXd::Constant(cost_dim, bound));
        ub.emplace_back(Eigen::VectorXd::Constant(cost_dim, -bound));
        output.emplace_back(2 * cost_dim + eq_dim);
      }
    }
  }
}

void Qp_Workspace::change_bound(double bound_) { bound = bound_; }

Eigen::Vector<double, Eigen::Dynamic>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, double bias, double mu, int n_iters, int thread_id,
   int batch_position, std::optional<Eigen::Ref<const Eigen::MatrixXd>> G_,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> lb_,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> ub_) {
  if (workspace.strategy_ == 0) {
    auto cost_dim = static_cast<int>(Q.cols());
    auto eq_dim = static_cast<int>(A.rows());

    double tol = 1e-7;

    Eigen::VectorXd primal_residual(eq_dim);
    Eigen::VectorXd dual_residual(cost_dim);

    Eigen::MatrixXd &kkt = workspace.kkt_[thread_id];
    Eigen::VectorXd &rhs = workspace.rhs_[batch_position];
    Eigen::VectorXd &temp = workspace.temp_vec_[thread_id];
    Eigen::FullPivLU<Eigen::MatrixXd> &lu = workspace.lu_[thread_id];
    rhs.setZero();

    kkt.topLeftCorner(cost_dim, cost_dim) = Q;
    kkt.topLeftCorner(cost_dim, cost_dim).diagonal().array() += bias;

    kkt.topRightCorner(cost_dim, eq_dim) = A.transpose();
    kkt.bottomLeftCorner(eq_dim, cost_dim) = A;
    kkt.bottomRightCorner(eq_dim, eq_dim).setZero();
    kkt.bottomRightCorner(eq_dim, eq_dim).diagonal().array() = -mu;
    workspace.kkt_mem_[batch_position] = kkt;
    lu.compute(kkt);

    for (int i = 0; i < n_iters; i++) {
      rhs.head(cost_dim) = bias * rhs.head(cost_dim);
      rhs.head(cost_dim) -= p;
      rhs.tail(eq_dim) = -mu * rhs.tail(eq_dim);
      rhs.tail(eq_dim) += b;
      temp.noalias() = lu.solve(rhs);
      rhs.noalias() = temp;

      Eigen::VectorXd x = rhs.head(cost_dim);
      Eigen::VectorXd lambda = rhs.tail(eq_dim);

      primal_residual = A * x - b;
      dual_residual = Q * x + A.transpose() * lambda + p;

      double primal_error = primal_residual.norm();
      double dual_error = dual_residual.norm();

      if (primal_error < tol && dual_error < tol) {
        break;
      }
      if (i == n_iters - 1) {
        std::cout << "algorithm did not converged" << std::endl;
        std::cout << "Q" << Q << std::endl;
        std::cout << "p" << p << std::endl;
        std::cout << "A" << A << std::endl;
        std::cout << "b" << b << std::endl;
        printConditionNumber(Q);
        printConditionNumber(A);
        printConditionNumber(kkt);
      }
    }
    workspace.sol_[batch_position] = rhs;
    if (batch_position == 0) {
      std::cout << rhs.head(cost_dim) << std::endl;
      std::cout << Q << std::endl;
      std::cout << A << std::endl;
      std::cout << p << std::endl;
      std::cout << b << std::endl;
    }
    return rhs.head(cost_dim);
  } else if (workspace.strategy_ == 1) {
    auto cost_dim = static_cast<int>(Q.cols());
    auto eq_dim = static_cast<int>(A.rows());
    if (eq_dim != 1) {
      throw runtime_error("pas le bon eq_dim");
    }
    Eigen::MatrixXd &identity = workspace.identity[thread_id];

    Eigen::VectorXd lb(cost_dim);
    lb.setConstant(-1000);
    Eigen::VectorXd output;

    workspace.qp[batch_position]->init(Q, p, A, b, identity, lb, -lb);
    workspace.qp[batch_position]->solve();
    output = workspace.qp[batch_position]->results.x;
    return output;
  } else if (workspace.strategy_ == 2) {
    auto cost_dim = static_cast<int>(Q.cols());
    Eigen::MatrixXd &identity = workspace.identity[thread_id];

    Eigen::VectorXd &lb = workspace.lb[thread_id];
    Eigen::VectorXd &ub = workspace.ub[thread_id];
    Eigen::VectorXd &output = workspace.output[thread_id];

    workspace.qp[batch_position]->init(Q, p, proxsuite::nullopt,
                                       proxsuite::nullopt, proxsuite::nullopt,
                                       proxsuite::nullopt, proxsuite::nullopt);
    // if (batch_position == 3) {
    //   std::cout << "H" << Q << std::endl;
    //   std::cout << "p" << p << std::endl;
    //   std::cout << "G" << identity << std::endl;
    //   std::cout << "lb" << lb << std::endl;
    //   std::cout << "ub" << ub << std::endl;
    // }
    workspace.qp[batch_position]->solve(workspace.warm_start_x[batch_position],
                                        proxsuite::nullopt, proxsuite::nullopt);
    output = workspace.qp[batch_position]->results.x;
    // std::cout << "batch" << batch_position << std::endl;
    // std::cout << "output" << output << std::endl;
    return output;
  } else if (workspace.strategy_ == 3 && G_.has_value()) {
    Eigen::VectorXd &output = workspace.output[thread_id];
    workspace.qp[batch_position]->init(Q, p, proxsuite::nullopt,
                                       proxsuite::nullopt, G_, lb_, ub_);
    workspace.qp[batch_position]->solve(
        workspace.warm_start_x[batch_position], proxsuite::nullopt,
        workspace.warm_start_neq[batch_position]);
    output = workspace.qp[batch_position]->results.x;
    // std::cout << "Q" << Q << std::endl;
    // std::cout << "p" << p << std::endl;
    // std::cout << "G" << *G_ << std::endl;
    // std::cout << "u" << *ub_ << std::endl;
    // std::cout << "l" << *lb_ << std::endl;
    // std::cout << "output" << output << std::endl;
    // std::cin.get();
    return output;
  } else if (workspace.strategy_ == 3 && !G_.has_value()) {
    throw "G has no values";
  }
  throw NotImplementedError("Strategy " + std::to_string(workspace.strategy_) +
                            " not implemented");
}

void QP_backward(Qp_Workspace &workspace,
                 Eigen::Ref<Eigen::VectorXd> grad_output, int thread_id,
                 int batch_position) {
  int cost_dim = workspace.cost_dim_;
  int eq_dim = workspace.eq_dim_;

  if (workspace.strategy_ == 0) {
    Eigen::FullPivLU<Eigen::MatrixXd> &lu = workspace.lu_[thread_id];
    lu.compute(workspace.kkt_mem_[batch_position].transpose());
    workspace.grad_rhs_mem_[batch_position] =
        lu.solve(grad_output.head(cost_dim + eq_dim));
    workspace.grad_KKT_mem_[batch_position] =
        -workspace.grad_rhs_mem_[batch_position] *
        workspace.sol_[batch_position].transpose();
  } else if (workspace.strategy_ == 1) {

    dense::compute_backward<double>(*workspace.qp[batch_position], grad_output,
                                    1e-7, 1e-7, 1e-7);
    workspace.grad_KKT_mem_[batch_position].setZero();
    workspace.grad_rhs_mem_[batch_position].setZero();
    workspace.grad_KKT_mem_[batch_position].topLeftCorner(cost_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dH;
    workspace.grad_KKT_mem_[batch_position].bottomLeftCorner(eq_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dA;
    workspace.grad_rhs_mem_[batch_position].head(cost_dim) =
        -workspace.qp[batch_position]->model.backward_data.dL_dg;
    workspace.grad_rhs_mem_[batch_position].tail(eq_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_db;
  } else if (workspace.strategy_ == 2) {
    double norm = grad_output.norm();
    if (norm > 1.0) {
      // grad_output /= norm;
      // grad_output *= 0;
      // if (norm > 10.0) {
      //   std::cout << "issue" << std::endl;
      // }
      std::cout << "issue" << std::endl;
      // std::cout << "Q" << workspace.qp[batch_position]->model.H << std::endl;
      // std::cout << "p" << workspace.qp[batch_position]->model.H << std::endl;
      // std::cout << "G" << workspace.qp[batch_position]->model.C << std::endl;
      // std::cout << "lb" << workspace.qp[batch_position]->model.l <<
      // std::endl; std::cout << "ub" << workspace.qp[batch_position]->model.u
      // << std::endl; grad_output(0) = 1;
    }
    dense::compute_backward<double>(*workspace.qp[batch_position], grad_output,
                                    1e-10, 1e-10, 1e-10);

    workspace.grad_KKT_mem_[batch_position].setZero();
    workspace.grad_rhs_mem_[batch_position].setZero();
    workspace.grad_KKT_mem_[batch_position].topLeftCorner(cost_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dH;
    workspace.grad_rhs_mem_[batch_position].head(cost_dim) =
        -workspace.qp[batch_position]->model.backward_data.dL_dg;
  } else if (workspace.strategy_ == 3) {
    dense::compute_backward<double>(*workspace.qp[batch_position], grad_output,
                                    1e-10, 1e-10, 1e-10);
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

double test_qp(int strategy) { return static_cast<double>(strategy) / 1000; }
