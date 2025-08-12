#include "qp.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <chrono>
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

void assert_grad_KKT(Eigen::MatrixXd grad_matrix, int batch_id) {
  // if (grad_matrix.norm() > 1e4 && batch_id < 10) {
  //   std::cout << "marche matrix pas à " << batch_id << std::endl;
  // }
}
void assert_grad_rhs(Eigen::VectorXd grad_vector, int batch_id) {
  // if (grad_vector.norm() > 1e4 && batch_id < 10) {
  //   std::cout << "marche vector pas à " << batch_id << std::endl;
  // }
  // if (grad_vector.norm() > 1e4 && batch_id > 10) {
  //   std::cout << "marche vector pas à " << batch_id << std::endl;
  //   throw;
  // }
}

void assertPositiveEigenvalues(const Eigen::MatrixXd &mat) {
  assert(mat.rows() == mat.cols() && "Matrix must be square.");

  // Compute eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(mat);
  assert(eigensolver.info() == Eigen::Success &&
         "Eigenvalue decomposition failed.");

  const auto &eigenvalues = eigensolver.eigenvalues();

  for (int i = 0; i < eigenvalues.size(); ++i) {
    assert(eigenvalues[i] > 0 && "Matrix has a non-positive eigenvalue.");
  }
}

void analyze_qp(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &A,
                const Eigen::VectorXd &b) {
  using namespace Eigen;

  bool is_symmetric = Q.isApprox(Q.transpose(), 1e-10);
  std::cout << "Q is symmetric : " << std::boolalpha << is_symmetric
            << std::endl;
  if (!is_symmetric) {
    std::cout << Q << std::endl;
  }
  SelfAdjointEigenSolver<MatrixXd> eigensolver(Q);
  if (eigensolver.info() != Success) {
    std::cerr << "could not compute Q eigenvals" << std::endl;
    return;
  }

  VectorXd eigenvalues = eigensolver.eigenvalues();
  std::cout << "Q lambda :\n" << eigenvalues.transpose() << std::endl;

  bool is_pd = (eigenvalues.array() > 1e-10).all();
  std::cout << "Q sdp : " << std::boolalpha << is_pd << std::endl;

  JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
  VectorXd x = svd.solve(b);
  VectorXd residual = A * x - b;
  double residual_norm = residual.norm();
  std::cout << "residual norm ||Ax - b|| : " << residual_norm << std::endl;
  std::cout << "b ∈ Im(A) ? " << std::boolalpha << (residual_norm < 1e-8)
            << std::endl;

  std::cout << "singular values of A :\n"
            << svd.singularValues().transpose() << std::endl;
}

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

void assert_allclose_v(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                       double atol = 1e-8, string name = "") {
  if (a.size() != b.size()) {
    throw std::runtime_error("Size mismatch: " + std::to_string(a.size()) +
                             " vs " + std::to_string(b.size()));
  }

  Eigen::VectorXd diff = (a - b).cwiseAbs();
  double max_diff = diff.maxCoeff();
  std::cout << "error :" << name << " : " << max_diff << std::endl;
  if (max_diff > atol) {
    std::cout << "a" << a << std::endl;
    std::cout << "b" << b << std::endl;
    std::cout << "diff" << a - b << std::endl;
    throw std::runtime_error("Arrays : " + name + " not close: max diff = " +
                             std::to_string(max_diff) +
                             " > atol = " + std::to_string(atol));
  }
}

void assert_allclose_m(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
                       double atol = 1e-8, string name = "") {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::runtime_error("Shape mismatch: (" + std::to_string(a.rows()) +
                             "," + std::to_string(a.cols()) + ") vs (" +
                             std::to_string(b.rows()) + "," +
                             std::to_string(b.cols()) + ")");
  }

  Eigen::MatrixXd diff = (a - b).cwiseAbs();
  double max_diff = diff.maxCoeff();
  std::cout << "error :" << name << " : " << max_diff << std::endl;

  if (max_diff > atol) {
    std::cout << "a" << a << std::endl;
    std::cout << "b" << b << std::endl;
    std::cout << "diff" << a - b << std::endl;
    throw std::runtime_error("Matrices: " + name + " not close: max diff = " +
                             std::to_string(max_diff) +
                             " > atol = " + std::to_string(atol));
  }
}

void thresholdSmallValues(Eigen::VectorXd &v, double t = 1e-10) {
  v = (v.array().abs() < t).select(0., v);
}

void thresholdSmallValues_mat(Eigen::MatrixXd &mat, double threshold = 1e-10) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      if (std::abs(mat(i, j)) < threshold) {
        mat(i, j) = 0.0;
      }
    }
  }
}

void computeAndPrintEigenvalues(const Eigen::MatrixXd &mat) {
  Eigen::EigenSolver<Eigen::MatrixXd> solver(mat);

  if (solver.info() != Eigen::Success) {
    std::cerr << "Error during eigenvalue computation!" << std::endl;
    return;
  }

  Eigen::VectorXcd eigenvalues = solver.eigenvalues();

  std::cout << "Eigenvalues:" << std::endl;
  for (int i = 0; i < eigenvalues.size(); ++i) {
    std::cout << "  Eigenvalue " << i << " : " << eigenvalues[i] << std::endl;
    if (eigenvalues[i].real() < 0) {
      throw std::invalid_argument(
          "Negative value found where a positive value was expected.");
    }
  }
}

void Qp_Workspace::reset() {
  for (auto &mat : kkt_) {
    mat.setZero();
  }
  for (auto &mat : kkt_mem_) {
    mat.setZero();
  }
  for (auto &mat : grad_KKT_mem_) {
    mat.setZero();
  }
  for (auto &mat : grad_KKT_) {
    mat.setZero();
  }
  for (auto &vec : rhs_) {
    vec.setZero();
  }
  for (auto &vec : grad_rhs_mem_) {
    vec.setZero();
  }
  for (auto &vec : grad_rhs_) {
    vec.setZero();
  }
  for (auto &vec : sol_) {
    vec.setZero();
  }
  for (auto &vec : temp_vec_) {
    vec.setZero();
  }

  for (auto &qp_solver : qp) {
    if (qp_solver.has_value()) {
    }
  }
}
class NotImplementedError : public std::exception {
  std::string message_;

public:
  explicit NotImplementedError(const std::string &msg) : message_(msg) {}

  const char *what() const noexcept override { return message_.c_str(); }
};

void Qp_Workspace::allocate(int batch_size, int cost_dim, int eq_dim,
                            int n_threads, int strategy) {
  if (false || strategy == 0) {
    if (strategy_ != strategy || batch_size != batch_size_ ||
        cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
      strategy_ = strategy;
      batch_size_ = batch_size;
      cost_dim_ = cost_dim;
      eq_dim_ = eq_dim;
      n_threads_ = n_threads;

      kkt_mem_.clear();
      kkt_mem_.reserve(batch_size);
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
      temp_vec_.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        temp_vec_.emplace_back(cost_dim + eq_dim);
      }
    }
  }
  if (false || strategy == 1) {
    if (true || strategy_ != strategy || batch_size != batch_size_ ||
        cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
      strategy_ = strategy;
      batch_size_ = batch_size;
      cost_dim_ = cost_dim;
      eq_dim_ = eq_dim;
      n_threads_ = n_threads;

      double eps_abs = 5e-7;
      double eps_rel = 5e-7;
      qp.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        qp[i].emplace(cost_dim, eq_dim, cost_dim);
        qp[i]->settings.eps_abs = eps_abs;
        qp[i]->settings.eps_rel = eps_rel;
        qp[i]->settings.primal_infeasibility_solving = true;
        qp[i]->settings.eps_primal_inf = 1e-5;
        qp[i]->settings.eps_dual_inf = 1e-5;
        qp[i]->settings.verbose = false;
      }
      identity.clear();
      identity.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        identity[i] = Eigen::MatrixXd(cost_dim, cost_dim);
        identity[i].setZero();
        identity[i].diagonal().array() = 1;
      }

      grad_rhs_mem_.clear();
      grad_rhs_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
        grad_rhs_mem_[i].setZero();
      }

      grad_KKT_mem_.clear();
      grad_KKT_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
        grad_KKT_mem_[i].setZero();
      }
    }
  }
  if (false || strategy == 2) {
    if (false || strategy_ != strategy || batch_size != batch_size_ ||
        cost_dim != cost_dim_ || eq_dim != eq_dim_ || n_threads != n_threads_) {
      strategy_ = strategy;
      batch_size_ = batch_size;
      cost_dim_ = cost_dim;
      eq_dim_ = eq_dim;
      n_threads_ = n_threads;

      double eps_abs = 5e-7;
      double eps_rel = 5e-7;
      qp.resize(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        qp[i].emplace(cost_dim, 0, cost_dim);
        qp[i]->settings.eps_abs = eps_abs;
        qp[i]->settings.eps_rel = eps_rel;
        qp[i]->settings.primal_infeasibility_solving = true;
        qp[i]->settings.eps_primal_inf = 1e-5;
        qp[i]->settings.eps_dual_inf = 1e-5;
        qp[i]->settings.verbose = false;
      }

      grad_rhs_mem_.clear();
      grad_rhs_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_rhs_mem_.emplace_back(cost_dim + eq_dim);
        grad_rhs_mem_[i].setZero();
      }

      grad_KKT_mem_.clear();
      grad_KKT_mem_.reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        grad_KKT_mem_.emplace_back(cost_dim + eq_dim, cost_dim + eq_dim);
        grad_KKT_mem_[i].setZero();
      }
      identity.clear();
      identity.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        identity[i] = Eigen::MatrixXd(cost_dim, cost_dim);
        identity[i].setZero();
        identity[i].diagonal().array() = 1;
      }
      lb.clear();
      lb.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        lb[i] = Eigen::VectorXd(cost_dim);
        lb[i].setConstant(bound);
      }
      ub.clear();
      ub.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        ub[i] = Eigen::VectorXd(cost_dim);
        ub[i].setConstant(-bound);
      }
      output.clear();
      output.reserve(n_threads);
      for (int i = 0; i < n_threads; ++i) {
        output[i] = Eigen::VectorXd(2 * cost_dim + eq_dim);
        output[i].setZero();
      }
    }
  }
}

void Qp_Workspace::change_bound(double bound_) { bound = bound_; }

Eigen::Vector<double, Eigen::Dynamic>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, double bias, double mu, int n_iters, int thread_id,
   int batch_position) {
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
                                       proxsuite::nullopt, identity, lb, ub);
    workspace.qp[batch_position]->solve();
    output = workspace.qp[batch_position]->results.x;
    return output;
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

    dense::compute_backward<double>(*workspace.qp[batch_position], grad_output,
                                    1e-7, 1e-7, 1e-7);
    // Beware for errors if you do not setZero ?
    workspace.grad_KKT_mem_[batch_position].setZero();
    workspace.grad_rhs_mem_[batch_position].setZero();
    workspace.grad_KKT_mem_[batch_position].topLeftCorner(cost_dim, cost_dim) =
        workspace.qp[batch_position]->model.backward_data.dL_dH;
    workspace.grad_rhs_mem_[batch_position].head(cost_dim) =
        -workspace.qp[batch_position]->model.backward_data.dL_dg;
    assert_grad_KKT(workspace.qp[batch_position]->model.backward_data.dL_dH,
                    batch_position % 600);
    assert_grad_rhs(workspace.qp[batch_position]->model.backward_data.dL_dg,
                    batch_position % 600);
  }
}

double test_qp(int strategy) { return static_cast<double>(strategy) / 1000; }
