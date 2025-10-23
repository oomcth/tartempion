#include "forward_pass.hpp"
#include "qp.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <coal/collision.h>
#include <diffcoal/spatial.hpp>
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

void QP_pass_workspace::reset() {}

Eigen::Tensor<double, 3, Eigen::RowMajor> QP_pass_workspace::Get_positions_() {
  return positions_;
}

void print_tensor3dlocal(
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &tensor) {
  for (int i = 0; i < tensor.dimension(0); ++i) {
    std::cout << "Slice [" << i << "]:\n";
    for (int j = 0; j < tensor.dimension(1); ++j) {
      std::cout << "  Row " << j << ": ";
      for (int k = 0; k < tensor.dimension(2); ++k) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                  << tensor(i, j, k) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

void QP_pass_workspace::set_L1_weight(double weight) { lambda_L1 = weight; }
void QP_pass_workspace::set_rot_weight(double weight) { rot_w = weight; }
void QP_pass_workspace::set_q_reg(double q_reg_) { q_reg = q_reg_; }
void QP_pass_workspace::set_lambda(double lambda_) { lambda = lambda_; }
void QP_pass_workspace::set_tool_id(int id) { tool_id = id; }
void QP_pass_workspace::set_bound(double bound) {
  assert(bound < 0 && "bound must be strictly negative");
  workspace_.change_bound(bound);
}

std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
QP_pass_workspace::grad_A() {
  return grad_A_;
};

std::vector<Eigen::VectorXd> QP_pass_workspace::get_last_q() { return last_q; };

std::vector<Eigen::VectorXd> QP_pass_workspace::grad_p() { return grad_p_; };

std::vector<Eigen::VectorXd> QP_pass_workspace::grad_b() { return grad_b_; };

void QP_pass_workspace::allocate(const pinocchio::Model &model, int batch_size,
                                 int seq_len, int cost_dim, int eq_dim,
                                 int num_thread) {
  if (batch_size != batch_size_ || seq_len != seq_len_ ||
      cost_dim != cost_dim_ || eq_dim != eq_dim_ || num_thread != num_thread_) {
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    cost_dim_ = cost_dim;
    eq_dim_ = eq_dim;
    num_thread_ = num_thread;

    int strategy = 2;
    workspace_.allocate(batch_size * seq_len, cost_dim, eq_dim, num_thread,
                        strategy);

    p_.resize(batch_size, seq_len, 6);
    p_.setZero();

    A_.resize(batch_size * seq_len, eq_dim, 6);
    A_.setZero();

    b_.resize(batch_size, seq_len, eq_dim);
    b_.setZero();

    grad_output_.resize(batch_size, seq_len, cost_dim);
    grad_output_.setZero();

    positions_ = Eigen::Tensor<double, 3, Eigen::RowMajor>(
        batch_size, seq_len + 1, cost_dim);
    positions_.setZero();

    articular_speed_ = Eigen::Tensor<double, 3, Eigen::RowMajor>(
        batch_size, seq_len, cost_dim);
    articular_speed_.setZero();

    last_T.clear();
    last_T.reserve(batch_size);

    last_logT.clear();
    last_logT.reserve(batch_size);

    last_q.clear();
    last_q.reserve(batch_size);

    steps_per_batch.resize(batch_size, 0);

    losses = Eigen::VectorXd::Zero(batch_size);

    for (int i = 0; i < batch_size; ++i) {
      last_T.emplace_back(pinocchio::SE3::Identity());
      last_logT.emplace_back(Eigen::VectorXd::Zero(6));
      last_q.emplace_back(Eigen::VectorXd::Zero(eq_dim));
    }

    const int total = batch_size * seq_len;

    grad_err_.clear();
    grad_err_.reserve(total);
    grad_p_.clear();
    grad_p_.reserve(total);
    grad_b_.clear();
    grad_b_.reserve(total);
    jacobians_.clear();
    jacobians_.reserve(total);
    grad_Q_.clear();
    grad_Q_.reserve(total);
    grad_A_.clear();
    grad_A_.reserve(total);

    diff.resize(total);
    target.resize(total);
    adj_diff.resize(total);
    adj.resize(total);
    errors_per_batch.resize(total);

    for (int i = 0; i < total; ++i) {
      grad_err_.emplace_back(6);
      grad_p_.emplace_back(6);
      grad_b_.emplace_back(eq_dim);
      jacobians_.emplace_back(6, model.nq);
      grad_Q_.emplace_back(cost_dim, cost_dim);
      grad_A_.emplace_back(eq_dim, 6);

      diff[i] = pinocchio::SE3::Identity();
      target[i] = pinocchio::Motion::Zero();
      adj_diff[i] = Eigen::MatrixXd::Zero(6, 6);
      adj[i] = Eigen::MatrixXd::Zero(6, 6);
    }

    const int n_thread = num_thread;

    Hessian.clear();
    Hessian.reserve(n_thread);
    data_vec_.clear();
    data_vec_.reserve(n_thread);
    Q_vec_.clear();
    Q_vec_.reserve(n_thread);
    grad_J_.clear();
    grad_J_.reserve(n_thread);
    dJdvq_vec.clear();
    dJdvq_vec.reserve(n_thread);
    dJdaq_vec.clear();
    dJdaq_vec.reserve(n_thread);
    v_vec.clear();
    v_vec.reserve(n_thread);
    a_vec.clear();
    a_vec.reserve(n_thread);

    localPosition.resize(n_thread);
    p_thread_mem.resize(n_thread);
    temp.resize(n_thread);
    A_thread_mem.resize(n_thread);
    grad_AJ.resize(n_thread);
    grad_Jeq.resize(n_thread);
    gradJ_Q.resize(n_thread);

    for (int i = 0; i < n_thread; ++i) {
      data_vec_.emplace_back(model);

      Hessian.emplace_back(6, cost_dim, cost_dim);
      Hessian[i].setZero();

      localPosition[i] = Eigen::VectorXd::Zero(cost_dim);
      p_thread_mem[i] = Eigen::VectorXd::Zero(cost_dim);
      temp[i] = Eigen::VectorXd::Zero(cost_dim);
      A_thread_mem[i] = Eigen::MatrixXd::Zero(eq_dim, cost_dim);
      grad_AJ[i] = Eigen::MatrixXd::Zero(eq_dim, cost_dim);
      grad_Jeq[i] = Eigen::MatrixXd::Zero(6, cost_dim);
      gradJ_Q[i] = Eigen::MatrixXd::Zero(6, cost_dim);

      dJdvq_vec.emplace_back(6, model.nv);
      dJdaq_vec.emplace_back(6, model.nv);

      v_vec.emplace_back(cost_dim);
      a_vec.emplace_back(cost_dim);

      Q_vec_.emplace_back(cost_dim, cost_dim);
      grad_J_.emplace_back(6, model.nq);
    }
  }
}

void single_forward_pass(QP_pass_workspace &workspace,
                         const pinocchio::Model &model, int thread_id,
                         int batch_id, int batch_size, int seq_len,
                         int cost_dim, int eq_dim, int tool_id,
                         pinocchio::SE3 T_star) {

  double lambda = workspace.lambda;

  for (int time = 0; time < seq_len; time++) {
    double *p_ptr = workspace.p_.data() + batch_id * seq_len * 6 + time * 6;
    Eigen::Map<Eigen::VectorXd> p(p_ptr, 6);

    double *A_ptr =
        workspace.A_.data() + (batch_id * seq_len + time) * eq_dim * 6;
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A(A_ptr, eq_dim, 6);

    double *b_ptr =
        workspace.b_.data() + batch_id * seq_len * eq_dim + time * eq_dim;
    Eigen::Map<Eigen::VectorXd> b(b_ptr, eq_dim);

    double *q_ptr = workspace.positions_.data() +
                    batch_id * (seq_len + 1) * cost_dim + time * cost_dim;
    Eigen::Map<VectorXd> q(q_ptr, cost_dim);

    double *q_next_ptr = workspace.positions_.data() +
                         batch_id * (seq_len + 1) * cost_dim +
                         (time + 1) * cost_dim;
    Eigen::Map<VectorXd> q_next(q_next_ptr, cost_dim);

    pinocchio::Data &data = workspace.data_vec_[thread_id];
    pinocchio::framesForwardKinematics(model, data, q);
    Eigen::MatrixXd &jac = workspace.jacobians_[batch_id * seq_len + time];
    jac.setZero();
    pinocchio::computeFrameJacobian(model, data, q, tool_id, pinocchio::LOCAL,
                                    jac);

    Eigen::MatrixXd &Q = workspace.Q_vec_[thread_id];
    Q = jac.transpose() * jac;
    Q.diagonal().array() += workspace.q_reg;

    workspace.A_thread_mem[thread_id] = A * jac;

    double *articular_speed_ptr = workspace.articular_speed_.data() +
                                  batch_id * seq_len * cost_dim +
                                  time * cost_dim;
    Eigen::Map<VectorXd> articular_speed(articular_speed_ptr, cost_dim);

    pinocchio::SE3 current_placement = data.oMf[tool_id];
    pinocchio::Motion target_lie(p.head(3), p.tail(3));
    pinocchio::SE3 target_placement = pinocchio::exp6(target_lie);
    pinocchio::SE3 &diff = workspace.diff[batch_id * seq_len + time];
    Eigen::MatrixXd &adj = workspace.adj[batch_id * seq_len + time];
    Eigen::MatrixXd &adj_diff = workspace.adj_diff[batch_id * seq_len + time];
    diff = current_placement.actInv(target_placement);
    adj = current_placement.inverse().toActionMatrix();
    workspace.target[batch_id * seq_len + time] = target_lie;
    adj_diff = diff.toActionMatrixInverse();
    Eigen::VectorXd err = pinocchio::log6(diff).toVector();

    if (time > 0) {
      workspace.workspace_.warm_start_x[batch_size * batch_id + time] =
          workspace.workspace_.qp[batch_size * batch_id + time - 1]->results.x;
      workspace.workspace_.warm_start_eq[batch_size * batch_id + time] =
          std::nullopt;
      workspace.workspace_.warm_start_neq[batch_size * batch_id + time] =
          workspace.workspace_.qp[batch_size * batch_id + time - 1]->results.z;
    }

    Eigen::VectorXd target = lambda * jac.transpose() * err;
    if (target.squaredNorm() * 100 < err.squaredNorm()) {
      target.setConstant(0);
    }

    articular_speed =
        QP(Q, target, workspace.A_thread_mem[thread_id] * 0, b * 0,
           workspace.workspace_, workspace.bias, workspace.mu, workspace.n_iter,
           thread_id, batch_id * seq_len + time, std::nullopt, std::nullopt,
           std::nullopt);
    q_next.noalias() = q + workspace.dt * articular_speed;

    double tracking_error = err.squaredNorm();
    workspace.errors_per_batch[batch_id * seq_len + time] = tracking_error;
    if (time > workspace.min_iters - 1 || time == seq_len - 1) {
      if (tracking_error < -pow(workspace.stopping_criterion_treshold, 2) ||
          time == seq_len - 1 ||
          target.squaredNorm() * 1000 < -err.squaredNorm()) {
        if (target.squaredNorm() * 1000 < err.squaredNorm()) {
          std::cout << "\033[31m" << "stopped" << time << "\033[0m"
                    << std::endl;
        } else {
          std::cout << "stopped" << time << std::endl;
        }
        workspace.steps_per_batch[batch_id] = time;
        workspace.last_q[batch_id] = q_next;
        pinocchio::framesForwardKinematics(model, data, q_next);
        pinocchio::updateFramePlacement(model, data, tool_id);
        workspace.last_T[batch_id] = data.oMf[tool_id].actInv(T_star);
        workspace.last_logT[batch_id] =
            pinocchio::log6(workspace.last_T[batch_id]);
        Eigen::VectorXd log_vec = workspace.last_logT[batch_id].toVector();
        log_vec.tail(3) *= workspace.rot_w;
        double loss_L2 = log_vec.squaredNorm();
        double loss_L1 = log_vec.lpNorm<1>();
        workspace.losses[batch_id] = loss_L2 + workspace.lambda_L1 * loss_L1;
        if (batch_id == 0) {
          std::cout << "end" << data.oMf[tool_id] << std::endl;
          std::cout << "star" << T_star << std::endl;
        }
        break;
      }
    }
  }
}

Eigen::VectorXd
forward_pass(QP_pass_workspace &workspace,
             const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
             const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
             const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
             const Eigen::MatrixXd &initial_position,
             const pinocchio::Model &model, int num_thread,
             const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
             double dt) {
  auto batch_size = static_cast<int>(p.dimension(0));
  auto seq_len = static_cast<int>(p.dimension(1));
  assert(workspace.tool_id != -1 &&
         "You must set workspace's tool id. (workspace.set_tool_id(tool_id))");
  assert(seq_len > 100 && "seq_len must be greater than 100.");
  assert(workspace.min_iters >= 100 &&
         "min_iter must be greater than 100. (you must change cpp code)");
  int cost_dim = model.nq;
  auto eq_dim = static_cast<int>(A.dimension(1));
  workspace.allocate(model, batch_size, seq_len, cost_dim, eq_dim, num_thread);

  workspace.dt = dt;
  workspace.b_ = b;
  workspace.A_ = A;
  workspace.p_ = p;

  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    double *q_ptr =
        workspace.positions_.data() + batch_id * (seq_len + 1) * cost_dim;
    Eigen::Map<VectorXd> q(q_ptr, cost_dim);

    q = initial_position.row(batch_id);
  }
  omp_set_num_threads(num_thread);

#pragma omp parallel for schedule(static)
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    int thread_id = omp_get_thread_num();
    single_forward_pass(workspace, model, thread_id, batch_id, batch_size,
                        seq_len, cost_dim, eq_dim, workspace.tool_id,
                        T_star[batch_id]);
  }
  return workspace.losses;
}

void compute_frame_hessian(QP_pass_workspace &workspace,
                           pinocchio::Model &model, int &thread_id,
                           int &cost_dim, int &tool_id, pinocchio::Data &data,
                           const Eigen::Map<VectorXd> &q, Eigen::VectorXd &v,
                           Eigen::VectorXd &a, Eigen::MatrixXd &v_partial_dq,
                           Eigen::MatrixXd &v_partial_dv) {
  for (int k = 0; k < cost_dim; ++k) {
    v.setZero();
    v(k) = 1;
    v_partial_dq.setZero();
    v_partial_dv.setZero();
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);
    pinocchio::getFrameVelocityDerivatives(
        model, data, tool_id, pinocchio::LOCAL, v_partial_dq, v_partial_dv);
    for (int i = 0; i < v_partial_dq.rows(); ++i) {
      for (int j = 0; j < v_partial_dq.cols(); ++j) {
        workspace.Hessian[thread_id](i, k, j) = v_partial_dq(i, j);
      }
    }
  }
}

void compute_direct_grad_path(Eigen::Map<Eigen::VectorXd> grad_vec_local,
                              QP_pass_workspace &workspace, int thread_id,
                              int cost_dim, double dt) {
  Eigen::VectorXd &temp = workspace.temp[thread_id];
  temp.setZero();
  for (int j = 0; j < cost_dim; ++j) {
    double acc = 0.0;
    for (int k = 0; k < 6; ++k) {
      for (int l = 0; l < cost_dim; ++l) {
        acc += workspace.grad_J_[thread_id](k, l) *
               workspace.Hessian[thread_id](k, l, j);
      }
    }
    temp(j) = acc;
  }
  grad_vec_local += dt * temp;
}

void update_indirect_grad_path_1(Eigen::Map<Eigen::VectorXd> grad_vec_local,
                                 pinocchio::Model &model, pinocchio::SE3 &diff,
                                 Eigen::VectorXd &rhs_grad, double lambda,
                                 QP_pass_workspace &workspace, int thread_id,
                                 int cost_dim, double dt) {
  Eigen::VectorXd &temp = workspace.temp[thread_id];
  temp.setZero();
  Eigen::VectorXd log = pinocchio::log6(diff).toVector();

  for (int j = 0; j < model.nq; ++j) {
    double acc = 0.0;
    for (int k = 0; k < 6; ++k) {
      for (int l = 0; l < model.nq; ++l) {
        acc += -rhs_grad.head(model.nq)(l) * lambda *
               workspace.Hessian[thread_id](k, l, j) * log(k);
      }
    }
    temp(j) = acc;
  }
  grad_vec_local += dt * temp;
}
void update_indirect_grad_path_2(Eigen::Map<Eigen::VectorXd> grad_vec_local,
                                 pinocchio::Model &model, pinocchio::SE3 &diff,
                                 Eigen::VectorXd &rhs_grad, double lambda,
                                 QP_pass_workspace &workspace, int thread_id,
                                 int cost_dim, double dt, int batch_id,
                                 int seq_len, int time) {
  Eigen::VectorXd &temp = workspace.temp[thread_id];
  temp.setZero();

  temp = -workspace.jacobians_[batch_id * seq_len + time].transpose() *
         (-workspace.adj_diff[batch_id * seq_len + time].transpose()) *
         pinocchio::Jlog6(diff).transpose() *
         workspace.jacobians_[batch_id * seq_len + time] *
         rhs_grad.head(model.nq) * lambda;
  grad_vec_local += temp * dt;
}

void single_backward_pass(
    QP_pass_workspace &workspace, pinocchio::Model &model, int thread_id,
    int batch_id, int seq_len, int cost_dim, int eq_dim, int tool_id, double dt,
    Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output) {

  Eigen::VectorXd w(6);
  w << 1, 1, 1, workspace.rot_w, workspace.rot_w, workspace.rot_w;
  pinocchio::Data &data = workspace.data_vec_[thread_id];
  Eigen::VectorXd e = workspace.last_logT[batch_id].toVector();
  Eigen::VectorXd e_scaled = w.array() * e.array();
  double loss_L2 = e_scaled.squaredNorm();
  double loss_L1 = e_scaled.lpNorm<1>();
  double final_loss = loss_L2 + workspace.lambda_L1 * loss_L1;
  Eigen::VectorXd grad_e = 2.0 * e_scaled.array() * w.array();
  Eigen::VectorXd sign_e_scaled = e_scaled.unaryExpr(
      [](double x) { return static_cast<double>((x > 0) - (x < 0)); });
  grad_e += (workspace.lambda_L1 * sign_e_scaled.array() * w.array()).matrix();
  Eigen::MatrixXd Adj = workspace.last_T[batch_id].inverse().toActionMatrix();
  Eigen::MatrixXd Jlog = pinocchio::Jlog6(workspace.last_T[batch_id]);
  Eigen::MatrixXd J_frame(6, model.nv);
  J_frame.setZero();
  pinocchio::computeFrameJacobian(model, data, workspace.last_q[batch_id],
                                  tool_id, pinocchio::LOCAL, J_frame);

  Eigen::RowVectorXd dloss_dq =
      J_frame.transpose() * (-Adj.transpose()) * Jlog.transpose() * grad_e;

  for (int i = 0; i < workspace.steps_per_batch[batch_id]; ++i) {
    double *grad_ptr = grad_output.data() +
                       batch_id * seq_len * grad_output.dimension(2) +
                       i * grad_output.dimension(2);
    Eigen::Map<Eigen::VectorXd> grad(grad_ptr, grad_output.dimension(2));
    grad.setZero();
    grad.head(model.nq) = dloss_dq * dt;
  }
  for (int time = workspace.steps_per_batch[batch_id]; time >= 0; time--) {
    int idx = batch_id * seq_len + time;
    double lambda = workspace.lambda;
    auto grad_dim = static_cast<int>(grad_output.dimension(2));
    Eigen::Map<Eigen::VectorXd> grad_vec(
        grad_output.data() + batch_id * seq_len * grad_dim + time * grad_dim,
        grad_dim);
    QP_backward(workspace.workspace_, grad_vec, thread_id, idx);
    Eigen::MatrixXd &KKT_grad = workspace.workspace_.grad_KKT_mem_[idx];
    Eigen::VectorXd &rhs_grad = workspace.workspace_.grad_rhs_mem_[idx];

    Eigen::MatrixXd &grad_AJ = workspace.grad_AJ[thread_id];
    grad_AJ.setZero();

    for (int i = 0; i < cost_dim; ++i) {
      for (int j = 0; j < eq_dim; ++j) {
        grad_AJ(j, i) = KKT_grad(i, cost_dim + j);
      }
    }
    for (int i = 0; i < eq_dim; ++i) {
      for (int j = 0; j < cost_dim; ++j) {
        grad_AJ(i, j) += KKT_grad(cost_dim + i, j);
      }
    }
    workspace.grad_A_[idx] = grad_AJ * workspace.jacobians_[idx].transpose();
    workspace.grad_err_[idx] =
        -workspace.jacobians_[idx] * rhs_grad.head(model.nq) * lambda;
    pinocchio::SE3 &diff = workspace.diff[idx];
    Eigen::MatrixXd &adj_diff = workspace.adj_diff[batch_id * seq_len + time];
    Eigen::VectorXd grad_target =
        pinocchio::Jlog6(diff).transpose() * workspace.grad_err_[idx]; // TODO
    Eigen::VectorXd &grad_p = workspace.grad_p_[idx];
    grad_p = pinocchio::Jexp6(workspace.target[idx]).transpose() * grad_target;

    workspace.grad_b_[idx] = rhs_grad.tail(eq_dim);
    Eigen::MatrixXd &J = workspace.jacobians_[idx];
    if (workspace.workspace_.strategy_ == 0) {
      workspace.grad_J_[thread_id] =
          J * (KKT_grad.block(0, 0, cost_dim, cost_dim).transpose() +
               KKT_grad.block(0, 0, cost_dim, cost_dim));
    } else if (workspace.workspace_.strategy_ == 1 ||
               workspace.workspace_.strategy_ == 2) {
      workspace.grad_J_[thread_id] =
          2 * J * KKT_grad.block(0, 0, cost_dim, cost_dim);
    }

    double *q_ptr = workspace.positions_.data() +
                    batch_id * (seq_len + 1) * cost_dim + (time + 1) * cost_dim;
    const Eigen::Map<VectorXd> q(q_ptr, cost_dim);
    Eigen::VectorXd &v = workspace.v_vec[thread_id];
    Eigen::VectorXd &a = workspace.a_vec[thread_id];
    Eigen::MatrixXd &v_partial_dq = workspace.dJdvq_vec[thread_id];
    Eigen::MatrixXd &v_partial_dv = workspace.dJdaq_vec[thread_id];

    compute_frame_hessian(workspace, model, thread_id, cost_dim, tool_id, data,
                          q, v, a, v_partial_dq, v_partial_dv);

    for (int i = 0; i < time; ++i) {
      Eigen::Map<Eigen::VectorXd> grad_vec_local(
          grad_output.data() + batch_id * seq_len * grad_dim + i * grad_dim,
          cost_dim);
      compute_direct_grad_path(grad_vec_local, workspace, thread_id, cost_dim,
                               dt);
      update_indirect_grad_path_1(grad_vec_local, model, diff, rhs_grad, lambda,
                                  workspace, thread_id, cost_dim, dt);
      update_indirect_grad_path_2(grad_vec_local, model, diff, rhs_grad, lambda,
                                  workspace, thread_id, cost_dim, dt, batch_id,
                                  seq_len, time);
      // Eigen::VectorXd &temp = workspace.temp[thread_id];
      // temp.setZero();
      // for (int j = 0; j < cost_dim; ++j) {
      //   double acc = 0.0;
      //   for (int k = 0; k < 6; ++k) {
      //     for (int l = 0; l < cost_dim; ++l) {
      //       acc += workspace.grad_J_[thread_id](k, l) *
      //              workspace.Hessian[thread_id](k, l, j);
      //     }
      //   }
      //   temp(j) = acc;
      // }
      // grad_vec_local += dt * temp;

      // Eigen::VectorXd &temp = workspace.temp[thread_id];
      // temp.setZero();
      // Eigen::VectorXd log = pinocchio::log6(diff).toVector();

      // for (int j = 0; j < model.nq; ++j) {
      //   double acc = 0.0;
      //   for (int k = 0; k < 6; ++k) {
      //     for (int l = 0; l < model.nq; ++l) {
      //       acc += -rhs_grad.head(model.nq)(l) * lambda *
      //              workspace.Hessian[thread_id](k, l, j) * log(k);
      //     }
      //   }
      //   temp(j) = acc;
      // }
      // grad_vec_local += dt * temp;

      // Eigen::VectorXd &temp = workspace.temp[thread_id];
      // temp.setZero();

      // temp = -workspace.jacobians_[batch_id * seq_len + time].transpose() *
      //        (-workspace.adj_diff[batch_id * seq_len + time].transpose()) *
      //        pinocchio::Jlog6(diff).transpose() *
      //        workspace.jacobians_[batch_id * seq_len + time] *
      //        rhs_grad.head(model.nq) * lambda;
      // grad_vec_local += temp * dt;
      // grad_vec_local *=
      //     static_cast<double>(seq_len) / workspace.steps_per_batch[batch_id];
    }
  }
}

void backward_pass(QP_pass_workspace &workspace, pinocchio::Model &model,
                   const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
                   int num_thread, int batch_size) {
  int cost_dim = model.nq;
  int eq_dim = static_cast<int>(workspace.b_.dimension(2));
  int seq_len = static_cast<int>(workspace.b_.dimension(1));
  int tool_id = workspace.tool_id;
  double dt = workspace.dt;
  workspace.grad_output_ = grad_output;

  omp_set_num_threads(num_thread);
#pragma omp parallel for schedule(static)
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    int thread_id = omp_get_thread_num();
    single_backward_pass(workspace, model, thread_id, batch_id, seq_len,
                         cost_dim, eq_dim, tool_id, dt, grad_output);
  }
}

int main() { return 0; }