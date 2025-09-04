#pragma once
#include "qp.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <omp.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <pinocchio/spatial/log.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace pinocchio;

struct QP_pass_workspace {
  double lambda = -1;
  int batch_size_ = -1;
  int seq_len_ = -1;
  int cost_dim_ = -1;
  int eq_dim_ = -1;
  int num_thread_ = -1;
  double mu = 1e-8;
  double bias = 1e-5;
  int n_iter = 1000;
  double dt = 1;
  int tool_id = -1;
  double lambda_L1 = 0;
  double rot_w = 1;
  double q_reg = 1e-5;
  double stopping_criterion_treshold = 1e-7;
  int min_iters = 100;

  Eigen::Tensor<double, 3, Eigen::RowMajor> p_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> A_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> b_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> positions_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> articular_speed_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output_;

  std::vector<Eigen::VectorXd> localPosition;
  std::vector<Eigen::MatrixXd> jacobians_;
  std::vector<Eigen::MatrixXd> grad_J_;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      grad_Q_;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      grad_A_;
  std::vector<int> steps_per_batch;
  std::vector<double> errors_per_batch;
  std::vector<Eigen::VectorXd> grad_err_;
  std::vector<Eigen::VectorXd> grad_p_;
  std::vector<Eigen::VectorXd> grad_b_;
  std::vector<Eigen::MatrixXd> dJdvq_vec;
  std::vector<Eigen::MatrixXd> dJdaq_vec;
  std::vector<Eigen::VectorXd> v_vec;
  std::vector<Eigen::VectorXd> a_vec;
  std::vector<pinocchio::Data> data_vec_;
  std::vector<Eigen::MatrixXd> Q_vec_;
  std::vector<Eigen::MatrixXd> J_vec_;
  std::vector<Eigen::VectorXd> p_thread_mem;
  std::vector<Eigen::MatrixXd> A_thread_mem;
  std::vector<Eigen::MatrixXd> grad_AJ;
  std::vector<Eigen::MatrixXd> grad_Jeq;
  std::vector<Eigen::MatrixXd> gradJ_Q;
  std::vector<Eigen::MatrixXd> adj;
  std::vector<Eigen::MatrixXd> J_frame;
  std::vector<Eigen::MatrixXd> Adj_backward;
  std::vector<Eigen::MatrixXd> J_log;
  std::vector<Eigen::MatrixXd> adj_diff;
  std::vector<Eigen::VectorXd> temp;
  std::vector<Eigen::Quaterniond> target_quat;
  std::vector<Eigen::VectorXd> last_q;
  std::vector<Eigen::VectorXd> log_diff;
  std::vector<Eigen::VectorXd> grad_target;
  std::vector<Eigen::VectorXd> e;
  std::vector<Eigen::RowVectorXd> dloss_dq;
  std::vector<pinocchio::SE3> last_T;
  std::vector<pinocchio::Motion> last_logT;
  Eigen::VectorXd losses;
  std::vector<pinocchio::SE3> diff;
  std::vector<pinocchio::Motion> target;
  std::vector<Eigen::Tensor<double, 3, Eigen::RowMajor>> Hessian;
  std::vector<double> N_Jtv;
  std::vector<double> N_kine_err;
  void set_L1_weight(double L1_w);
  void set_rot_weight(double L1_w);
  void set_q_reg(double q_reg);
  void set_lambda(double lambda);
  void set_tool_id(int id);
  void set_bound(double bound);

  Qp_Workspace workspace_;

  void allocate(const pinocchio::Model &model, int batch_size, int seq_len,
                int cost_dim, int eq_dim, int num_thread);
  void reset();

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  grad_A();
  std::vector<Eigen::VectorXd> get_last_q();
  Eigen::Tensor<double, 3, Eigen::RowMajor> Get_positions_();
  std::vector<Eigen::VectorXd> grad_p();
  std::vector<Eigen::VectorXd> grad_b();
};

void backward_pass(QP_pass_workspace &workspace, pinocchio::Model &model,
                   const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
                   int num_thread, int batch_size);

Eigen::VectorXd forward_pass(QP_pass_workspace &workspace,
                             const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
                             const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
                             const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
                             const Eigen::MatrixXd initial_position,
                             const pinocchio::Model &model, int num_thread,
                             const PINOCCHIO_ALIGNED_STD_VECTOR(SE3) & T_star,
                             double dt);
