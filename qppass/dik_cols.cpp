#include "dik_cols.hpp"
#include "qp.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <coal/collision.h>
#include <coal/collision_object.h>
#include <csignal>
#include <diffcoal/contact_derivative.hpp>
#include <diffcoal/contact_derivative_data.hpp>
#include <diffcoal/spatial.hpp>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <omp.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/fcl-pinocchio-conversions.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/multibody/fcl.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/geometry-object.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <pinocchio/spatial/log.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace pinocchio;

void QP_pass_workspace2::reset() {}

void QP_pass_workspace2::init_geometry(pinocchio::Model rmodel) {
  geom_end_eff = pinocchio::GeometryObject(
      "end eff", tool_id, rmodel.frames[tool_id].parentJoint,
      std::make_shared<coal::Sphere>(effector_ball),
      pinocchio::SE3::Identity());
  geom_base = pinocchio::GeometryObject(
      "base", 0, 0, std::make_shared<coal::Sphere>(base_ball),
      pinocchio::SE3::Identity());
  geom_elbow = pinocchio::GeometryObject(
      "elbow", elbow_id, rmodel.frames[elbow_id].parentJoint,
      std::make_shared<coal::Sphere>(elbow_ball), pinocchio::SE3::Identity());
  geom_plane = pinocchio::GeometryObject("plane", 0, 0,
                                         std::make_shared<coal::Box>(plane),
                                         pinocchio::SE3::Identity());
}

Eigen::Tensor<double, 3, Eigen::RowMajor> QP_pass_workspace2::Get_positions_() {
  return positions_;
}

double t_norm2(const SE3 &transform) {
  Vector3d translation = transform.translation();
  return translation.squaredNorm();
}

void QP_pass_workspace2::set_L1_weight(double weight) { lambda_L1 = weight; }
void QP_pass_workspace2::set_rot_weight(double weight) { rot_w = weight; }
void QP_pass_workspace2::set_q_reg(double q_reg_) { q_reg = q_reg_; }
void QP_pass_workspace2::set_lambda(double lambda_) { lambda = lambda_; }
void QP_pass_workspace2::set_tool_id(int id) { tool_id = id; }
void QP_pass_workspace2::set_bound(double bound) {
  assert(bound < 0 && "bound must be strictly negative");
  workspace_.change_bound(bound);
}

std::vector<
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
QP_pass_workspace2::grad_A() {
  return grad_A_;
};

std::vector<Eigen::VectorXd> QP_pass_workspace2::get_last_q() {
  return last_q;
};

std::vector<Eigen::VectorXd> QP_pass_workspace2::grad_p() { return grad_p_; };

std::vector<Eigen::VectorXd> QP_pass_workspace2::grad_b() { return grad_b_; };

Eigen::VectorXd QP_pass_workspace2::dloss_dqf(int i) {
  return dloss_dq[i].transpose();
}

void QP_pass_workspace2::allocate(const pinocchio::Model &model, int batch_size,
                                  int seq_len, int cost_dim, int eq_dim,
                                  int num_thread) {
  if (batch_size != batch_size_ || seq_len != seq_len_ ||
      cost_dim != cost_dim_ || num_thread != num_thread_) {
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    cost_dim_ = cost_dim;
    num_thread_ = num_thread;
    int strategy;
    if (collisions) {
      strategy = 3;
    } else {
      strategy = 2;
    }
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

    positions_.resize(batch_size, seq_len + 1, cost_dim);
    positions_.setZero();

    articular_speed_.resize(batch_size, seq_len, cost_dim);
    articular_speed_.setZero();

    last_T.resize(batch_size, pinocchio::SE3::Identity());
    last_logT.resize(batch_size, pinocchio::Motion().Zero());
    last_q.resize(batch_size, Eigen::VectorXd::Zero(eq_dim));
    steps_per_batch.resize(batch_size, 0);
    dloss_dq.resize(batch_size, Eigen::VectorXd::Zero(model.nq));
    dloss_dq_diff.resize(batch_size, Eigen::VectorXd::Zero(model.nq));
    losses = Eigen::VectorXd::Zero(batch_size);

    const int total = batch_size * seq_len;
    grad_err_.resize(total, Eigen::VectorXd::Zero(6));
    grad_p_.resize(total, Eigen::VectorXd::Zero(6));
    grad_b_.resize(total, Eigen::VectorXd::Zero(eq_dim));
    jacobians_.resize(total, Eigen::MatrixXd::Zero(6, model.nv));
    grad_Q_.resize(total, Eigen::MatrixXd::Zero(model.nv, model.nv));
    grad_A_.resize(total, Eigen::MatrixXd::Zero(eq_dim, model.nv));
    diff.resize(total, pinocchio::SE3::Identity());
    target.resize(total, pinocchio::Motion::Zero());
    adj_diff.resize(total, Eigen::MatrixXd::Zero(6, 6));
    adj.resize(total, Eigen::MatrixXd::Zero(6, 6));
    errors_per_batch.resize(total, 0);

    creq.resize(total);
    cres.resize(total);
    cdreq.resize(total);
    cdres.resize(total);
    for (int i = 0; i < total; ++i) {
      creq[i] = coal::CollisionRequest();
      creq[i].security_margin = 10000;
      cres[i] = coal::CollisionResult();
      cdreq[i] = diffcoal::ContactDerivativeRequest();
      cdres[i] = diffcoal::ContactDerivative();
    }

    const int n_thread = num_thread;

    Hessian.clear();
    Hessian.reserve(n_thread);
    for (int i = 0; i < n_thread; ++i) {
      Hessian.emplace_back();
      Hessian.back().resize(6, cost_dim, cost_dim);
      Hessian.back().setZero();
    }
    data_vec_.clear();
    data_vec_.reserve(n_thread);
    Q_vec_.resize(n_thread, Eigen::MatrixXd::Zero(cost_dim, cost_dim));
    grad_J_.resize(n_thread, Eigen::MatrixXd::Zero(6, cost_dim));
    dJdvq_vec.resize(n_thread, Eigen::MatrixXd::Zero(6, model.nv));
    dJdaq_vec.resize(n_thread, Eigen::MatrixXd::Zero(6, model.nv));
    v_vec.resize(n_thread, Eigen::VectorXd::Zero(cost_dim));
    a_vec.resize(n_thread, Eigen::VectorXd::Zero(cost_dim));
    gmodel.reserve(n_thread);
    gdata.reserve(n_thread);
    localPosition.resize(n_thread, Eigen::VectorXd::Zero(cost_dim));
    p_thread_mem.resize(n_thread, Eigen::VectorXd::Zero(cost_dim));
    temp.resize(n_thread, Eigen::VectorXd::Zero(cost_dim));
    A_thread_mem.resize(n_thread, Eigen::MatrixXd::Zero(eq_dim, cost_dim));
    grad_AJ.resize(n_thread, Eigen::MatrixXd::Zero(eq_dim, cost_dim));
    grad_Jeq.resize(n_thread, Eigen::MatrixXd::Zero(6, cost_dim));
    gradJ_Q.resize(n_thread, Eigen::MatrixXd::Zero(6, cost_dim));
    Adj_vec.resize(n_thread, Eigen::MatrixXd::Zero(6, 6));
    Jlog_vec.resize(n_thread, Eigen::MatrixXd::Zero(6, 6));
    Jlog_v4.resize(n_thread, Eigen::MatrixXd::Zero(6, 6));
    J_frame_vec.resize(n_thread, Eigen::MatrixXd::Zero(6, model.nv));
    target_placement_vec.resize(n_thread, pinocchio::SE3::Identity());
    current_placement_vec.resize(n_thread, pinocchio::SE3::Identity());
    err_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    padded.resize(n_thread);
    for (auto &v : padded)
      v = Eigen::VectorXd::Zero(2 * model.nv);
    target_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    last_log_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    w_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    e_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    temp_direct.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    e_scaled_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    grad_target_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    v1.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    v2.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    v3.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    sign_e_scaled_vec.resize(n_thread,
                             Eigen::Vector<double, 6>().setConstant(0));
    grad_e_vec.resize(n_thread, Eigen::Vector<double, 6>().setConstant(0));
    log_indirect_1_vec.resize(n_thread,
                              Eigen::Vector<double, 6>().setConstant(0));

    end_eff_placement.clear();
    end_eff_placement.reserve(n_thread);
    base_placement.clear();
    base_placement.reserve(n_thread);
    elbow_placement.clear();
    elbow_placement.reserve(n_thread);
    plane_placement.clear();
    plane_placement.reserve(n_thread);
    dn_dq.resize(n_thread, Eigen::MatrixXd::Zero(6, model.nq));
    dw1_dq.resize(n_thread, Eigen::MatrixXd::Zero(3, model.nq));
    dw2_dq.resize(n_thread, Eigen::MatrixXd::Zero(3, model.nq));

    for (int i = 0; i < n_thread; ++i) {
      data_vec_.emplace_back(model);

      gmodel.emplace_back();
      gmodel[i].addGeometryObject(geom_end_eff.value());
      gmodel[i].addGeometryObject(geom_base.value());
      gmodel[i].addGeometryObject(geom_elbow.value());
      gmodel[i].addGeometryObject(geom_plane.value());
      end_eff_placement.emplace_back();
      base_placement.emplace_back();
      elbow_placement.emplace_back();
      plane_placement.emplace_back();
      gdata.emplace_back();
      gdata[i] = pinocchio::GeometryData(gmodel[i]);
    }
  }
}

void single_forward_pass(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, int thread_id,
                         int batch_id, int batch_size, int seq_len,
                         int cost_dim, int eq_dim, int tool_id,
                         pinocchio::SE3 T_star) {
  T_star.rotation() =
      pinocchio::orthogonalProjection(T_star.rotation()); // Should be useless.
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
    pinocchio::framesForwardKinematics(model, workspace.data_vec_[thread_id],
                                       q);
    Eigen::MatrixXd &jac = workspace.jacobians_[batch_id * seq_len + time];
    jac.setZero();
    pinocchio::computeFrameJacobian(model, workspace.data_vec_[thread_id], q,
                                    tool_id, pinocchio::LOCAL, jac);
    VectorXd ub;
    VectorXd lb;
    if (workspace.collisions) {
      pinocchio::updateGeometryPlacements(
          model, data, workspace.gmodel[thread_id], workspace.gdata[thread_id]);
      Eigen::MatrixXd G = Eigen::MatrixXd(1, model.nq);
      G.setZero();

      ub = VectorXd(G.rows());
      lb = -VectorXd(G.rows());

      coal::CollisionResult &res = workspace.cres[batch_id * seq_len + time];
      coal::CollisionRequest &req = workspace.creq[batch_id * seq_len + time];

      coal::collide(
          &workspace.effector_ball,
          pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[0]),
          &workspace.plane,
          pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[3]), req,
          res);
      if (res.getContact(0).penetration_depth < 0) {
        std::cout << "critical error collision" << std::endl;
        throw std::runtime_error("Critical error: collision 2");
      } else {
        pinocchio::updateGlobalPlacements(model,
                                          workspace.data_vec_[thread_id]);
        pinocchio::computeJointJacobians(model, workspace.data_vec_[thread_id],
                                         q);
        int j1_id = workspace.geom_base->parentJoint;
        int j2_id = workspace.geom_end_eff->parentJoint;
        Vector3d w1 = res.getContact(0).nearest_points[0];
        Vector3d w2 = res.getContact(0).nearest_points[1];
        Vector3d w_diff = w1 - w2;
        double norm = w_diff.norm();
        Vector3d n = w_diff / norm;

        MatrixXd J_1(6, model.nq);
        MatrixXd J_2(6, model.nq);

        getJointJacobian(model, workspace.data_vec_[thread_id], j1_id,
                         LOCAL_WORLD_ALIGNED, J_1);
        getJointJacobian(model, workspace.data_vec_[thread_id], j2_id,
                         LOCAL_WORLD_ALIGNED, J_2);

        Vector3d r1 =
            w1 - workspace.data_vec_[thread_id].oMi[j1_id].translation();
        Vector3d r2 =
            w2 - workspace.data_vec_[thread_id].oMi[j2_id].translation();
        Matrix3d skew_r1, skew_r2;
        skew_r1 << 0, -r1(2), r1(1), r1(2), 0, -r1(0), -r1(1), r1(0), 0;

        skew_r2 << 0, -r2(2), r2(1), r2(2), 0, -r2(0), -r2(1), r2(0), 0;
        RowVectorXd J_coll(model.nq);
        J_coll = n.transpose() * J_1.block(0, 0, 3, model.nq) +
                 (pinocchio::skew(r1) * n).transpose() *
                     J_1.block(3, 0, 3, model.nq);
        J_coll -= n.transpose() * J_2.block(0, 0, 3, model.nq) +
                  (pinocchio::skew(r2) * n).transpose() *
                      J_2.block(3, 0, 3, model.nq);
        int constraint_idx = 0;
        G.row(constraint_idx) = J_coll / workspace.dt;
        ub(constraint_idx) =
            20.0 * (res.getContact(0).penetration_depth - 0.05);
        lb(constraint_idx) = -1e10;
      }
    }

    Eigen::MatrixXd &Q = workspace.Q_vec_[thread_id];
    Q.noalias() = jac.transpose() * jac;
    Q.diagonal().array() += workspace.q_reg;

    workspace.A_thread_mem[thread_id] = A * jac;

    double *articular_speed_ptr = workspace.articular_speed_.data() +
                                  batch_id * seq_len * cost_dim +
                                  time * cost_dim;
    Eigen::Map<VectorXd> articular_speed(articular_speed_ptr, cost_dim);

    pinocchio::SE3 &current_placement =
        workspace.current_placement_vec[thread_id];
    pinocchio::Motion target_lie(p.head(3), p.tail(3));
    pinocchio::SE3 &target_placement =
        workspace.target_placement_vec[thread_id];
    pinocchio::SE3 &diff = workspace.diff[batch_id * seq_len + time];
    Eigen::MatrixXd &adj = workspace.adj[batch_id * seq_len + time];
    Eigen::MatrixXd &adj_diff = workspace.adj_diff[batch_id * seq_len + time];
    Eigen::VectorXd &err = workspace.err_vec[thread_id];
    Eigen::VectorXd &target = workspace.target_vec[thread_id];

    current_placement = workspace.data_vec_[thread_id].oMf[tool_id];
    target_placement = pinocchio::exp6(target_lie);
    diff = current_placement.actInv(target_placement);
    adj = current_placement.toActionMatrixInverse();
    workspace.target[batch_id * seq_len + time] = target_lie;
    adj_diff = diff.toActionMatrixInverse();
    err = pinocchio::log6(diff).toVector();

    if (time > 0) {
      workspace.workspace_.warm_start_x[batch_size * batch_id + time] =
          workspace.workspace_.qp[batch_size * batch_id + time - 1]->results.x;
      workspace.workspace_.warm_start_eq[batch_size * batch_id + time] =
          std::nullopt;
      workspace.workspace_.warm_start_neq[batch_size * batch_id + time] =
          workspace.workspace_.qp[batch_size * batch_id + time - 1]->results.z;
    }

    target = lambda * jac.transpose() * err;

    if (workspace.collisions) {
      articular_speed =
          QP(Q, target, workspace.A_thread_mem[thread_id] * 0, b * 0,
             workspace.workspace_, workspace.bias, workspace.mu,
             workspace.n_iter, thread_id, batch_id * seq_len + time, Q, lb, ub);
    } else {
      articular_speed =
          QP(Q, target, workspace.A_thread_mem[thread_id] * 0, b * 0,
             workspace.workspace_, workspace.bias, workspace.mu,
             workspace.n_iter, thread_id, batch_id * seq_len + time,
             std::nullopt, std::nullopt, std::nullopt);
    }
    q_next.noalias() = q + workspace.dt * articular_speed;

    double tracking_error = err.squaredNorm();
    workspace.errors_per_batch[batch_id * seq_len + time] = tracking_error;
    if (time == seq_len - 1) {
      Eigen::VectorXd &log_vec = workspace.last_log_vec[thread_id];
      workspace.steps_per_batch[batch_id] = time;
      workspace.last_q[batch_id] = q_next;
      workspace.last_T[batch_id] =
          workspace.data_vec_[thread_id].oMf[tool_id].actInv(T_star);
      workspace.last_logT[batch_id] =
          pinocchio::log6(workspace.last_T[batch_id]);
      log_vec = workspace.last_logT[batch_id].toVector();
      log_vec.tail(3) *= workspace.rot_w;
      double loss_L2 = log_vec.squaredNorm();
      double loss_L1 = log_vec.lpNorm<1>();
      workspace.losses[batch_id] = loss_L2 + workspace.lambda_L1 * loss_L1;
    }
  }
}

Eigen::VectorXd
forward_pass2(QP_pass_workspace2 &workspace,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, int num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(SE3) & T_star, double dt) {
  const auto batch_size = static_cast<int>(p.dimension(0));
  const auto seq_len = static_cast<int>(p.dimension(1));
  const auto eq_dim = static_cast<int>(A.dimension(1));
  const int cost_dim = model.nq;

  assert(workspace.tool_id != -1 &&
         "You must set workspace's tool id. (workspace.set_tool_id(tool_id))");

  workspace.init_geometry(model);
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
    const int thread_id = omp_get_thread_num();
    single_forward_pass(workspace, model, thread_id, batch_id, batch_size,
                        seq_len, cost_dim, eq_dim, workspace.tool_id,
                        T_star[batch_id]);
  }
  return workspace.losses;
}

void compute_frame_hessian(QP_pass_workspace2 &workspace,
                           pinocchio::Model &model, int &thread_id,
                           int &cost_dim, int &tool_id, pinocchio::Data &data,
                           const Eigen::Map<VectorXd> &q, Eigen::VectorXd &v,
                           Eigen::VectorXd &a, Eigen::MatrixXd &v_partial_dq,
                           Eigen::MatrixXd &v_partial_dv) {
  v.setZero();
  for (int k = 0; k < cost_dim; ++k) {
    v(k) = 1.0;
    v_partial_dq.setZero();
    v_partial_dv.setZero();
    pinocchio::computeForwardKinematicsDerivatives(
        model, workspace.data_vec_[thread_id], q, v, a);
    pinocchio::getFrameVelocityDerivatives(
        model, workspace.data_vec_[thread_id], tool_id, pinocchio::LOCAL,
        v_partial_dq, v_partial_dv);
    for (int j = 0; j < v_partial_dq.cols(); ++j) {
      for (int i = 0; i < v_partial_dq.rows(); ++i) {
        workspace.Hessian[thread_id](i, k, j) = v_partial_dq(i, j);
      }
    }
    v(k) = 0.0;
  }
}

void backpropagateThroughQ(Eigen::Ref<VectorXd> grad_vec_local,
                           QP_pass_workspace2 &workspace, int thread_id,
                           int cost_dim, double dt) {
  Map<const VectorXd> g(workspace.grad_J_[thread_id].data(), 6 * cost_dim);
  Map<const Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> H(
      workspace.Hessian[thread_id].data(), 6 * cost_dim, cost_dim);

  VectorXd &temp = workspace.temp_direct[thread_id];
  const double s = std::max(1e-9, g.norm());
  temp.noalias() = (H.transpose() * (g / s)) * s;

  grad_vec_local.noalias() += dt * temp;
}

void backpropagateThroughJ0(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                            const pinocchio::Model &model,
                            const pinocchio::SE3 &diff,
                            const Eigen::VectorXd &rhs_grad, double lambda,
                            QP_pass_workspace2 &workspace, int thread_id,
                            int cost_dim, double dt) {
  const int nq = model.nq;
  Eigen::VectorXd &temp = workspace.temp[thread_id];
  Eigen::VectorXd &log = workspace.log_indirect_1_vec[thread_id];
  const auto &H = workspace.Hessian[thread_id];

  log = lambda * pinocchio::log6(diff).toVector();
  const auto rhs_head = rhs_grad.head(nq);

  for (int j = 0; j < nq; ++j) {
    double acc = 0.0;
    double c = 0.0;

    for (int l = 0; l < nq; ++l) {
      const double rhs_l = rhs_head(l);
      for (int k = 0; k < 6; ++k) {
        double term = -rhs_l * H(k, l, j) * log(k);

        double y = term - c;
        double t = acc + y;
        c = (t - acc) - y;
        acc = t;
      }
    }
    temp(j) = acc;
  }
  grad_vec_local.noalias() += dt * temp;
}

void backpropagateThroughT(Eigen::Ref<VectorXd> grad_vec_local,
                           pinocchio::Model &model, pinocchio::SE3 &diff,
                           Eigen::VectorXd &rhs_grad, double lambda,
                           QP_pass_workspace2 &workspace, int thread_id,
                           int cost_dim, double dt, int batch_id, int seq_len,
                           int time) {
  const double scale = lambda * dt;
  const auto rhs_q = rhs_grad.head(model.nq);
  const int idx = batch_id * seq_len + time;
  const auto &J = workspace.jacobians_[idx];
  const auto &adj = workspace.adj_diff[idx];
  Eigen::MatrixXd &Jlog = workspace.Jlog_v4[thread_id];
  Jlog = pinocchio::Jlog6(diff);

  VectorXd &v1 = workspace.v1[thread_id];
  v1 = J * rhs_q;
  VectorXd &v2 = workspace.v2[thread_id];
  v2 = Jlog.transpose() * v1;
  VectorXd &v3 = workspace.v3[thread_id];
  v3 = adj.transpose() * v2;
  grad_vec_local += scale * (J.transpose() * v3);
}

void backpropagateThroughCollisions() {}

void compute_dn_dq(QP_pass_workspace2 &workspace, pinocchio::Model &model,
                   pinocchio::Data &data, int j1_id, int j2_id, int batch_id,
                   int seq_len, int time, Eigen::MatrixXd &dn_dq_) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> J1(6, model.nq);
  Eigen::Matrix<double, 6, Eigen::Dynamic> J2(6, model.nq);
  pinocchio::getJointJacobian(model, data, j1_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J1);
  pinocchio::getJointJacobian(model, data, j2_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J2);
  dn_dq_ = -(workspace.cdres[batch_id * seq_len + time].dnormal_dM1 * J1 +
             workspace.cdres[batch_id * seq_len + time].dnormal_dM2 * J2);
}

void compute_dw1_dq(QP_pass_workspace2 &workspace, pinocchio::Model &model,
                    pinocchio::Data &data, int j1_id, int j2_id, int batch_id,
                    int seq_len, int time, Eigen::MatrixXd &dw1_dq_) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> J1(6, model.nq);
  Eigen::Matrix<double, 6, Eigen::Dynamic> J2(6, model.nq);
  pinocchio::getJointJacobian(model, data, j1_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J1);
  dw1_dq_ = (workspace.cdres[batch_id * seq_len + time].dcpos_dM1 -
             workspace.cdres[batch_id * seq_len + time].dvsep_dM1 / 2) *
            J1;
}
void compute_dw2_dq(QP_pass_workspace2 &workspace, pinocchio::Model &model,
                    pinocchio::Data &data, int j1_id, int j2_id, int batch_id,
                    int seq_len, int time, Eigen::MatrixXd &dw2_dq_) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> J1(6, model.nq);
  Eigen::Matrix<double, 6, Eigen::Dynamic> J2(6, model.nq);
  pinocchio::getJointJacobian(model, data, j2_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J2);
  dw2_dq_ = (workspace.cdres[batch_id * seq_len + time].dcpos_dM2 -
             workspace.cdres[batch_id * seq_len + time].dvsep_dM2 / 2) *
            J2;
}

void compute_d_dist_and_d_Jcoll(QP_pass_workspace2 &workspace,
                                pinocchio::Model &model, pinocchio::Data &data,
                                int j1_id, int j2_id, int batch_id, int seq_len,
                                int time, Eigen::VectorXd &ddist,
                                Eigen::MatrixXd &dJcoll_dq, int thread_id,
                                Eigen::VectorXd q, int coll) {
  coal::CollisionResult &cres = workspace.cres[batch_id * seq_len + time];
  if (coll == 2) {
    coal::CollisionResult &cres = workspace.cres[batch_id * seq_len + time];
  } else {
    throw "coll does not exit";
  }
  Eigen::MatrixXd &dn_dq = workspace.dn_dq[thread_id];
  Eigen::MatrixXd &dw1_dq = workspace.dw1_dq[thread_id];
  Eigen::MatrixXd &dw2_dq = workspace.dw2_dq[thread_id];
  compute_dn_dq(workspace, model, workspace.data_vec_[thread_id], j1_id, j2_id,
                batch_id, seq_len, time, dn_dq);
  compute_dw1_dq(workspace, model, workspace.data_vec_[thread_id], j1_id, j2_id,
                 batch_id, seq_len, time, dw1_dq);
  compute_dw2_dq(workspace, model, workspace.data_vec_[thread_id], j1_id, j2_id,
                 batch_id, seq_len, time, dw2_dq);
  ddist = dw1_dq.transpose() *
          (cres.getContact(0).nearest_points[0] -
           cres.getContact(0).nearest_points[1]) /
          cres.getContact(0).penetration_depth;
  dJcoll_dq.setZero();
  pinocchio::forwardKinematics(model, workspace.data_vec_[thread_id], q);
  pinocchio::framesForwardKinematics(model, workspace.data_vec_[thread_id], q);
  pinocchio::computeForwardKinematicsDerivatives(
      model, workspace.data_vec_[thread_id], q, q, q);
  pinocchio::computeJointKinematicHessians(model,
                                           workspace.data_vec_[thread_id]);
  Eigen::MatrixXd J1(6, model.nq);
  Eigen::MatrixXd J2(6, model.nq);
  pinocchio::computeJointJacobians(model, workspace.data_vec_[thread_id], q);
  pinocchio::getJointJacobian(model, workspace.data_vec_[thread_id], j1_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J1);
  pinocchio::getJointJacobian(model, workspace.data_vec_[thread_id], j2_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J2);
  Eigen::Tensor<double, 3> H1(6, model.nv, model.nv);
  Eigen::Tensor<double, 3> H2(6, model.nv, model.nv);
  pinocchio::getJointKinematicHessian(model, workspace.data_vec_[thread_id],
                                      j1_id, pinocchio::LOCAL_WORLD_ALIGNED,
                                      H1);
  pinocchio::getJointKinematicHessian(model, workspace.data_vec_[thread_id],
                                      j2_id, pinocchio::LOCAL_WORLD_ALIGNED,
                                      H2);
  Eigen::MatrixXd term_A(model.nq, model.nq);
  Vector3d w1 = cres.getContact(0).nearest_points[0];
  Vector3d w2 = cres.getContact(0).nearest_points[1];
  Vector3d w_diff = w1 - w2;
  double norm = w_diff.norm();
  Vector3d n = w_diff / norm;
  for (int q = 0; q < model.nq; ++q) {
    Eigen::Tensor<double, 2> H1_slice = H1.chip(q, 2);
    Eigen::Tensor<double, 2> H2_slice = H2.chip(q, 2);
    Eigen::MatrixXd H_diff_slice =
        Eigen::Map<Eigen::MatrixXd>(H1_slice.data(), H1_slice.dimension(0),
                                    H1_slice.dimension(1)) -
        Eigen::Map<Eigen::MatrixXd>(H2_slice.data(), H2_slice.dimension(0),
                                    H2_slice.dimension(1));

    term_A.col(q) = H_diff_slice.transpose() * n;
  }
  auto J_diff = J1.topRows(3) - J2.topRows(3);

  Eigen::MatrixXd term_B = J_diff.transpose() * dn_dq;
  dJcoll_dq = term_A + term_B;
}

void update_q_limits_path(Eigen::Map<Eigen::VectorXd> grad_vec_local,
                          pinocchio::Model &model, pinocchio::SE3 &diff,
                          Eigen::VectorXd &rhs_grad, double lambda,
                          QP_pass_workspace2 &workspace, int thread_id,
                          int cost_dim, double dt, int batch_id, int seq_len,
                          int time) {
  grad_vec_local(2) -=
      dt * workspace.workspace_.grad_ub_mem_[batch_id * seq_len + time](6);
  grad_vec_local(2) -=
      dt * workspace.workspace_.grad_lb_mem_[batch_id * seq_len + time](6);
}

void single_backward_pass(
    QP_pass_workspace2 &workspace, pinocchio::Model &model, int thread_id,
    int batch_id, int seq_len, int cost_dim, int eq_dim, int tool_id, double dt,
    Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output) {

  Eigen::VectorXd &w = workspace.w_vec[thread_id];
  Eigen::VectorXd &e = workspace.e_vec[thread_id];
  Eigen::VectorXd &e_scaled = workspace.e_scaled_vec[thread_id];
  pinocchio::Data &data = workspace.data_vec_[thread_id];
  Eigen::VectorXd &grad_e = workspace.grad_e_vec[thread_id];
  Eigen::VectorXd &sign_e_scaled = workspace.sign_e_scaled_vec[thread_id];
  Eigen::MatrixXd &Adj = workspace.Adj_vec[thread_id];
  Eigen::MatrixXd &Jlog = workspace.Jlog_vec[thread_id];
  Eigen::MatrixXd &J_frame = workspace.J_frame_vec[thread_id];
  Eigen::VectorXd &dloss_dq = workspace.dloss_dq[batch_id];
  Eigen::VectorXd &dloss_dq_diff = workspace.dloss_dq_diff[batch_id];

  w << 1, 1, 1, workspace.rot_w, workspace.rot_w, workspace.rot_w;
  e = workspace.last_logT[batch_id].toVector();
  e_scaled = w.array() * e.array();
  double loss_L2 = e_scaled.squaredNorm();
  double loss_L1 = e_scaled.lpNorm<1>();
  double final_loss = loss_L2 + workspace.lambda_L1 * loss_L1;
  grad_e = 2.0 * e_scaled.array() * w.array();
  sign_e_scaled = e_scaled.unaryExpr(
      [](double x) { return static_cast<double>((x > 0) - (x < 0)); });
  grad_e += (workspace.lambda_L1 * sign_e_scaled.array() * w.array()).matrix();
  J_frame.setZero();
  Adj.setZero();
  Jlog.setZero();
  Adj = workspace.last_T[batch_id].toActionMatrixInverse();
  Jlog = pinocchio::Jlog6(workspace.last_T[batch_id]);
  pinocchio::computeFrameJacobian(model, data, workspace.last_q[batch_id],
                                  tool_id, LOCAL, J_frame);

  dloss_dq =
      dt * J_frame.transpose() * (-Adj.transpose()) * Jlog.transpose() * grad_e;
  dloss_dq_diff.setZero();

  for (int time = workspace.steps_per_batch[batch_id]; time >= 0; time--) {
    Eigen::VectorXd &grad_target = workspace.grad_target_vec[thread_id];
    int idx = batch_id * seq_len + time;
    double lambda = workspace.lambda;
    auto grad_dim = static_cast<int>(grad_output.dimension(2));
    Eigen::Map<Eigen::VectorXd> grad_vec(
        grad_output.data() + batch_id * seq_len * grad_dim + time * grad_dim,
        grad_dim);
    const int nq = model.nq;
    const int target_size = static_cast<int>(grad_output.dimension(2));

    Eigen::VectorXd &padded = workspace.padded[thread_id];
    padded.setZero();
    padded.head(nq) = dloss_dq + dloss_dq_diff;

    QP_backward(workspace.workspace_, padded, thread_id, idx);

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
    grad_target = pinocchio::Jlog6(diff).transpose() * workspace.grad_err_[idx];
    Eigen::VectorXd &grad_p = workspace.grad_p_[idx];
    grad_p.noalias() =
        pinocchio::Jexp6(workspace.target[idx]).transpose() * grad_target;

    workspace.grad_b_[idx] = rhs_grad.tail(eq_dim);
    Eigen::MatrixXd &J = workspace.jacobians_[idx];
    if (workspace.workspace_.strategy_ == 0) {
      workspace.grad_J_[thread_id] =
          J * (KKT_grad.block(0, 0, cost_dim, cost_dim).transpose() +
               KKT_grad.block(0, 0, cost_dim, cost_dim));
    } else if (workspace.workspace_.strategy_ == 1 ||
               workspace.workspace_.strategy_ == 2 ||
               workspace.workspace_.strategy_ == 3 || true) {
      workspace.grad_J_[thread_id] =
          2 * J * KKT_grad.block(0, 0, cost_dim, cost_dim);
    }
    Eigen::VectorXd ddist1;
    Eigen::VectorXd ddist2;
    Eigen::MatrixXd dJcoll_dq1;
    Eigen::MatrixXd dJcoll_dq2;
    double *q_ptr = workspace.positions_.data() +
                    batch_id * (seq_len + 1) * cost_dim + (time + 1) * cost_dim;
    const Eigen::Map<VectorXd> q(q_ptr, cost_dim);
    if (workspace.collisions) {
      dJcoll_dq1 = Eigen::MatrixXd(model.nq, model.nq);
      dJcoll_dq2 = Eigen::MatrixXd(model.nq, model.nq);
      compute_d_dist_and_d_Jcoll(
          workspace, model, data, workspace.geom_end_eff->parentJoint,
          workspace.geom_base->parentJoint, batch_id, seq_len, time, ddist1,
          dJcoll_dq1, thread_id, q, 0);
    }

    Eigen::VectorXd &v = workspace.v_vec[thread_id];
    Eigen::VectorXd &a = workspace.a_vec[thread_id];
    Eigen::MatrixXd &v_partial_dq = workspace.dJdvq_vec[thread_id];
    Eigen::MatrixXd &v_partial_dv = workspace.dJdaq_vec[thread_id];

    compute_frame_hessian(workspace, model, thread_id, cost_dim, tool_id,
                          workspace.data_vec_[thread_id], q, v, a, v_partial_dq,
                          v_partial_dv);

    backpropagateThroughQ(dloss_dq_diff, workspace, thread_id, cost_dim, dt);
    backpropagateThroughJ0(dloss_dq_diff, model, diff, rhs_grad, lambda,
                           workspace, thread_id, cost_dim, dt);
    backpropagateThroughT(dloss_dq_diff, model, diff, rhs_grad, lambda,
                          workspace, thread_id, cost_dim, dt, batch_id, seq_len,
                          time);
    if (workspace.collisions) {
      backpropagateThroughCollisions();
    }
  }
}

void backward_pass2(
    QP_pass_workspace2 &workspace, pinocchio::Model &model,
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
