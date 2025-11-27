#pragma once
#include <Eigen/Dense>
#include <optional>
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

struct Qp_Workspace {
  size_t batch_size_ = 0;
  double bound = -1000;
  size_t eq_dim_ = 0;
  size_t cost_dim_ = 0;
  size_t n_threads_ = 0;
  size_t strategy_ = 0;

  std::vector<Eigen::MatrixXd> kkt_;
  std::vector<Eigen::MatrixXd> kkt_mem_;
  std::vector<Eigen::MatrixXd> grad_KKT_mem_;
  std::vector<Eigen::MatrixXd> grad_KKT_;
  std::vector<Eigen::MatrixXd> grad_G_mem_;
  std::vector<Eigen::VectorXd> grad_lb_mem_;
  std::vector<Eigen::VectorXd> grad_ub_mem_;

  std::vector<Eigen::VectorXd> rhs_;
  std::vector<Eigen::VectorXd> grad_rhs_mem_;
  std::vector<Eigen::VectorXd> grad_rhs_;

  std::vector<Eigen::FullPivLU<Eigen::MatrixXd>> lu_;
  std::vector<Eigen::VectorXd> sol_;
  std::vector<Eigen::VectorXd> temp_vec_;

  std::vector<std::optional<proxsuite::proxqp::dense::QP<double>>> qp;

  std::vector<Eigen::MatrixXd> identity;
  std::vector<Eigen::VectorXd> lb;
  std::vector<Eigen::VectorXd> ub;
  std::vector<Eigen::VectorXd> output;

  std::vector<std::optional<Eigen::VectorXd>> warm_start_x;
  std::vector<std::optional<Eigen::VectorXd>> warm_start_eq;
  std::vector<std::optional<Eigen::VectorXd>> warm_start_neq;

  void allocate(size_t batch_size, size_t cost_dim, size_t eq_dim,
                size_t n_threads, size_t strategy, size_t ineq_dim);
  void reset();
  void change_bound(double bound);
};

Eigen::Ref<Eigen::Vector<double, Eigen::Dynamic>>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, size_t thread_id, size_t batch_position);

void QP_backward(Qp_Workspace &workspace,
                 Eigen::Ref<Eigen::VectorXd> grad_output,
                 size_t batch_position);

Eigen::Ref<Eigen::Vector<double, Eigen::Dynamic>>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, size_t thread_id, size_t batch_position,
   std::optional<Eigen::Ref<const Eigen::MatrixXd>> G_ = std::nullopt,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> lb_ = std::nullopt,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> ub_ = std::nullopt);

double test_qp(size_t strategy);
