#pragma once
#include <Eigen/Dense>
#include <optional>
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

using namespace Eigen;
using namespace std;
using namespace proxsuite;
using namespace proxsuite::proxqp;

struct Qp_Workspace {
  int batch_size_ = -1;
  double bound = -1000;
  int eq_dim_ = -1;
  int cost_dim_ = -1;
  int n_threads_ = -1;
  int strategy_ = 1;

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

  std::vector<std::optional<dense::QP<double>>> qp;

  std::vector<Eigen::MatrixXd> identity;
  std::vector<Eigen::VectorXd> lb;
  std::vector<Eigen::VectorXd> ub;
  std::vector<Eigen::VectorXd> output;

  std::vector<std::optional<Eigen::VectorXd>> warm_start_x;
  std::vector<std::optional<Eigen::VectorXd>> warm_start_eq;
  std::vector<std::optional<Eigen::VectorXd>> warm_start_neq;

  void allocate(int batch_size, int cost_dim, int eq_dim, int n_threads,
                int strategy);
  void reset();
  void change_bound(double bound);
};

Eigen::Vector<double, Eigen::Dynamic>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, double bias, double mu, int n_iters, int thread_id,
   int batch_position);

void QP_backward(Qp_Workspace &workspace,
                 Eigen::Ref<Eigen::VectorXd> grad_output, int thread_id,
                 int batch_position);

Eigen::Vector<double, Eigen::Dynamic>
QP(Eigen::Ref<const Eigen::MatrixXd> Q, Eigen::Ref<const Eigen::VectorXd> p,
   Eigen::Ref<const Eigen::MatrixXd> A, Eigen::Ref<const Eigen::VectorXd> b,
   Qp_Workspace &workspace, double bias, double mu, int n_iters, int thread_id,
   int batch_position,
   std::optional<Eigen::Ref<const Eigen::MatrixXd>> G_ = std::nullopt,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> lb_ = std::nullopt,
   std::optional<Eigen::Ref<const Eigen::VectorXd>> ub_ = std::nullopt);

double test_qp(int strategy);

Eigen::VectorXd solve_qp_simple(const Eigen::MatrixXd &Q,
                                const Eigen::VectorXd &p,
                                const Eigen::MatrixXd &A,
                                const Eigen::VectorXd &b);

double test_batch_qp(int B);
