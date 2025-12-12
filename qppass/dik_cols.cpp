#include "dik_cols.hpp"
#include "qp.hpp"
#include <Eigen/Dense>
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
#include <tracy/Tracy.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

using Vector6d = Eigen::Vector<double, 6>;
using Matrix66d = Eigen::Matrix<double, 6, 6>;
using Matrix3xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

template <typename T, typename = std::enable_if_t<
                          std::is_base_of_v<Eigen::DenseBase<T>, T>, void>>
void setZero(std::vector<T> &vec) {
  for (auto &x : vec) {
    x.setZero();
  }
}

void QP_pass_workspace2::reset() {}

void QP_pass_workspace2::init_geometry(pinocchio::Model model,
                                       size_t batch_size) {
  gmodel.clear();
  gdata.clear();
  geom_end_eff.clear();
  geom_arm_cylinder.clear();
  geom_plane.clear();
  geom_cylinder.clear();
  geom_ball.clear();

  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    geom_end_eff.emplace_back();
    geom_end_eff[batch_id] = pinocchio::GeometryObject(
        "end eff", tool_id, model.frames[tool_id].parentJoint,
        std::make_shared<coal::Sphere>(effector_ball[batch_id]),
        pinocchio::SE3(end_eff_rot[batch_id], end_eff_pos[batch_id]));

    geom_arm_cylinder.emplace_back();
    geom_arm_cylinder[batch_id] = pinocchio::GeometryObject(
        "arm cylinder", 209, model.frames[209].parentJoint,
        std::make_shared<coal::Capsule>(arm_cylinder[batch_id]),
        pinocchio::SE3(arm_cylinder_rot[batch_id], arm_cylinder_pos[batch_id]));

    geom_plane.emplace_back();
    geom_plane[batch_id] = pinocchio::GeometryObject(
        "plane", 0, 0, std::make_shared<coal::Box>(plane[batch_id]),
        pinocchio::SE3(plane_rot[batch_id], plane_pos[batch_id]));

    geom_cylinder.emplace_back();
    geom_cylinder[batch_id] = pinocchio::GeometryObject(
        "cylinder", 0, 0, std::make_shared<coal::Capsule>(cylinder[batch_id]),
        pinocchio::SE3(cylinder_rot[batch_id], cylinder_pos[batch_id]));

    geom_ball.emplace_back();
    geom_ball[batch_id] = pinocchio::GeometryObject(
        "ball", 0, 0, std::make_shared<coal::Sphere>(ball[batch_id]),
        pinocchio::SE3(ball_rot[batch_id], ball_pos[batch_id]));

    for (size_t i = 0; i < num_thread_; ++i) {
      gmodel.emplace_back();
      gmodel[i].addGeometryObject(geom_end_eff[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_cylinder[batch_id].value());
      gmodel[i].addGeometryObject(geom_plane[batch_id].value());
      gmodel[i].addGeometryObject(geom_cylinder[batch_id].value());
      gmodel[i].addGeometryObject(geom_ball[batch_id].value());
      gdata.emplace_back();
      gdata[i] = pinocchio::GeometryData(gmodel[i]);
    }
  }
}

Eigen::Tensor<double, 3, Eigen::RowMajor> QP_pass_workspace2::Get_positions_() {
  return positions_;
}

[[nodiscard]] inline double t_norm2(const pinocchio::SE3 &transform) {
  return transform.translation().squaredNorm();
}

void QP_pass_workspace2::set_L1_weight(double weight) { lambda_L1 = weight; }
void QP_pass_workspace2::set_rot_weight(double weight) { rot_w = weight; }
void QP_pass_workspace2::set_q_reg(double q_reg_) { q_reg = q_reg_; }
void QP_pass_workspace2::set_lambda(double lambda_) { lambda = lambda_; }
void QP_pass_workspace2::set_tool_id(size_t id) { tool_id = id; }
void QP_pass_workspace2::set_bound(double bound) {
  assert(bound < 0 && "bound must be strictly negative");
  workspace_.change_bound(bound);
}

std::vector<Eigen::VectorXd> QP_pass_workspace2::get_last_q() {
  return last_q;
};

std::vector<Vector6d> QP_pass_workspace2::grad_p() { return grad_p_; };

Eigen::Ref<Eigen::VectorXd> QP_pass_workspace2::dloss_dqf(size_t i) {
  return dloss_dq[i];
}

void QP_pass_workspace2::set_collisions_safety_margin(double margin) {
  if (collisions) {
    std::cout << "collisions are activated" << std::endl;
  } else {
    std::cout << "collisions are activated" << std::endl;
  }
  safety_margin = margin;
}

void QP_pass_workspace2::set_collisions_strength(double strength) {
  collision_strength = strength;
}

void QP_pass_workspace2::pre_allocate(size_t batch_size) {

  if (effector_ball.size() != batch_size) {
    end_eff_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_cylinder_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    plane_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    cylinder_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    ball_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());

    end_eff_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_cylinder_rot.resize(batch_size,
                            Eigen::Matrix<double, 3, 3>::Identity());
    plane_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    cylinder_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    ball_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());

    effector_ball.resize(batch_size, coal::Sphere(0.1));
    arm_cylinder.resize(batch_size, coal::Capsule(0.05, 0.5));
    plane.resize(batch_size, coal::Box(1e6, 1e6, 10));
    cylinder.resize(batch_size);
    ball.resize(batch_size);
  }
}
void QP_pass_workspace2::allocate(const pinocchio::Model &model,
                                  size_t batch_size, size_t seq_len,
                                  size_t cost_dim, size_t eq_dim,
                                  size_t num_thread) {
  ZoneScopedN("allocate");
  if (batch_size != batch_size_ || seq_len != seq_len_ ||
      cost_dim != cost_dim_ || num_thread != num_thread_) {
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    cost_dim_ = cost_dim;
    num_thread_ = num_thread;
    unsigned int strategy;
    if constexpr (collisions) {
      if (pairs.size() == 0) {
        strategy = 2;
      } else {
        strategy = 3;
      }
    } else {
      strategy = 2;
    }
    workspace_.allocate(batch_size * seq_len, cost_dim, eq_dim, num_thread,
                        strategy, pairs.size());
    p_.resize(static_cast<Eigen::Index>(batch_size),
              static_cast<Eigen::Index>(seq_len), 6);
    p_.setZero();

    A_.resize(static_cast<Eigen::Index>(batch_size * seq_len),
              static_cast<Eigen::Index>(eq_dim), 6);
    A_.setZero();

    b_.resize(static_cast<Eigen::Index>(batch_size),
              static_cast<Eigen::Index>(seq_len),
              static_cast<Eigen::Index>(eq_dim));
    b_.setZero();

    positions_.resize(static_cast<Eigen::Index>(batch_size),
                      static_cast<Eigen::Index>(seq_len + 1),
                      static_cast<Eigen::Index>(cost_dim));
    positions_.setZero();

    articular_speed_.resize(static_cast<Eigen::Index>(batch_size),
                            static_cast<Eigen::Index>(seq_len),
                            static_cast<Eigen::Index>(cost_dim));
    articular_speed_.setZero();

    discarded.resize(batch_size, false);
    last_q.resize(batch_size,
                  Eigen::VectorXd::Zero(static_cast<Eigen::Index>(cost_dim)));

    dloss_dq.resize(batch_size,
                    Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.nv)));
    dloss_dq_diff.resize(
        batch_size, Eigen::VectorXd::Zero(static_cast<Eigen::Index>(model.nv)));
    last_T.resize(batch_size, pinocchio::SE3::Identity());
    last_logT.resize(batch_size, pinocchio::Motion::Zero());

    losses = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(batch_size));
    steps_per_batch.resize(batch_size, 0);

    const size_t total = batch_size * seq_len;
    jacobians_.resize(total,
                      Matrix6xd::Zero(6, static_cast<Eigen::Index>(model.nv)));
    adj_diff.resize(total, Matrix66d::Zero());
    grad_err_.resize(total, Vector6d::Zero());
    grad_p_.resize(total, Vector6d::Zero());
    diff.resize(total, pinocchio::SE3::Identity());
    target.resize(total, pinocchio::Motion::Zero());

    const size_t n_thread = num_thread;

    Hessian.clear();
    Hessian.reserve(2 * n_thread);
    for (unsigned int i = 0; i < 2 * n_thread; ++i) {
      Hessian.emplace_back();
      Hessian.back().resize(6, static_cast<Eigen::Index>(cost_dim),
                            static_cast<Eigen::Index>(cost_dim));
      Hessian.back().setZero();
    }
    data_vec_.clear();
    data_vec_.reserve(n_thread);
    gmodel.reserve(n_thread);
    gdata.reserve(n_thread);
    Q_vec_.resize(n_thread,
                  Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                        static_cast<Eigen::Index>(cost_dim)));
    grad_J_.resize(n_thread,
                   Matrix6xd::Zero(6, static_cast<Eigen::Index>(cost_dim)));
    J_coll.resize(n_thread, Eigen::RowVectorXd::Zero(
                                static_cast<Eigen::Index>(model.nv)));
    J1.resize(n_thread,
              Matrix6xd::Zero(6, static_cast<Eigen::Index>(cost_dim)));
    J2.resize(n_thread,
              Matrix6xd::Zero(6, static_cast<Eigen::Index>(cost_dim)));
    J_1.resize(n_thread,
               Matrix6xd::Zero(6, static_cast<Eigen::Index>(cost_dim)));
    J_2.resize(n_thread,
               Matrix6xd::Zero(6, static_cast<Eigen::Index>(cost_dim)));
    A_thread_mem.resize(
        n_thread, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(eq_dim),
                                        static_cast<Eigen::Index>(cost_dim)));
    term_A.resize(n_thread,
                  Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                        static_cast<Eigen::Index>(cost_dim)));
    temp_tensor.resize(
        n_thread, Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, model.nv));
    for (auto &vec : temp_tensor) {
      vec.setZero();
    }
    M.resize(n_thread,
             Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                   static_cast<Eigen::Index>(cost_dim)));
    dout.resize(n_thread,
                Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                      static_cast<Eigen::Index>(cost_dim)));
    term_B.resize(n_thread,
                  Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                        static_cast<Eigen::Index>(cost_dim)));
    J_diff.resize(n_thread,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    dJcoll_dq.resize(
        n_thread, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                        static_cast<Eigen::Index>(cost_dim)));
    R.resize(n_thread, Eigen::Matrix3d::Identity());
    c.resize(num_thread, Eigen::Vector3d::Zero());
    dcj.resize(n_thread, Eigen::Vector3d::Zero());
    term1.resize(num_thread, Eigen::VectorXd::Zero(model.nv));
    dr1_dq.resize(n_thread,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    tmp1_dr1_dq.resize(n_thread, Eigen::Matrix<double, 3, 6>::Zero());
    tmp2_dr1_dq.resize(n_thread, Eigen::Matrix<double, 3, 6>::Zero());
    tmp3_dr1_dq.resize(
        n_thread, Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    tmp4_dr1_dq.resize(
        n_thread, Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    Adj_vec.resize(n_thread, Matrix66d::Zero());
    dloss_dq_tmp1.resize(n_thread, Matrix66d::Zero());
    dloss_dq_tmp2.resize(
        n_thread, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(cost_dim),
                                        static_cast<Eigen::Index>(6)));
    dloss_dq_tmp3.resize(
        n_thread, Eigen::VectorXd::Zero(static_cast<Eigen::Index>(cost_dim)));

    Jlog_vec.resize(n_thread, Matrix66d::Zero());
    Jlog_v4.resize(n_thread, Matrix66d::Zero());
    J_frame_vec.resize(n_thread,
                       Matrix6xd::Zero(6, static_cast<Eigen::Index>(model.nv)));
    padded.resize(n_thread);

    for (auto &v : padded)
      v = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(2 * model.nv));
    ddist.resize(n_thread,
                 Eigen::VectorXd::Zero(static_cast<Eigen::Index>(cost_dim)));
    temp_ddist.resize(n_thread, Eigen::RowVector<double, 6>::Zero());
    r1.resize(n_thread, Eigen::Vector3d::Zero());
    r2.resize(n_thread, Eigen::Vector3d::Zero());
    w1.resize(n_thread, Eigen::Vector3d::Zero());
    w2.resize(n_thread, Eigen::Vector3d::Zero());
    w_diff.resize(n_thread, Eigen::Vector3d::Zero());
    n.resize(n_thread, Eigen::Vector3d::Zero());
    ub.resize(n_thread,
              Eigen::Vector<double, Eigen::Dynamic>::Zero(pairs.size()));
    lb.resize(n_thread,
              Eigen::Vector<double, Eigen::Dynamic>::Zero(pairs.size()));
    G.resize(n_thread,
             Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                 pairs.size(), static_cast<Eigen::Index>(model.nv)));
    temp.resize(n_thread,
                Eigen::VectorXd::Zero(static_cast<Eigen::Index>(cost_dim)));
    target_vec.resize(n_thread, Vector6d().setConstant(0));
    err_vec.resize(n_thread, Vector6d().setConstant(0));
    last_log_vec.resize(n_thread, Vector6d().setConstant(0));
    w_vec.resize(n_thread, Vector6d().setConstant(0));
    e_vec.resize(n_thread, Vector6d().setConstant(0));
    temp_direct.resize(n_thread, Vector6d().setConstant(0));
    e_scaled_vec.resize(n_thread, Vector6d().setConstant(0));
    grad_target_vec.resize(n_thread, Vector6d().setConstant(0));
    v1.resize(n_thread, Vector6d().setConstant(0));
    v2.resize(n_thread, Vector6d().setConstant(0));
    v3.resize(n_thread, Vector6d().setConstant(0));

    sign_e_scaled_vec.resize(n_thread, Vector6d().setConstant(0));
    grad_e_vec.resize(n_thread, Vector6d().setConstant(0));
    log_indirect_1_vec.resize(n_thread, Vector6d().setConstant(0));
    target_placement_vec.resize(n_thread, pinocchio::SE3::Identity());
    current_placement_vec.resize(n_thread, pinocchio::SE3::Identity());

    dn_dq.resize(n_thread,
                 Matrix3xd::Zero(3, static_cast<Eigen::Index>(model.nv)));
    dw_dq.resize(n_thread,
                 Matrix3xd::Zero(3, static_cast<Eigen::Index>(model.nv)));
    dw2_dq.resize(n_thread,
                  Matrix3xd::Zero(3, static_cast<Eigen::Index>(model.nv)));

    for (unsigned int i = 0; i < n_thread; ++i) {
      data_vec_.emplace_back(model);
    }

    pinocchio::forwardKinematics(model, data_vec_[0],
                                 pinocchio::neutral(model));
    pinocchio::framesForwardKinematics(model, data_vec_[0],
                                       pinocchio::neutral(model));
    pinocchio::SE3 M_world_frame = data_vec_[0].oMf[tool_id];
    pinocchio::JointIndex parent_jid = model.frames[tool_id].parent;
    pinocchio::SE3 M_world_joint = data_vec_[0].oMi[parent_jid];
    pinocchio::SE3 M_frame_to_joint = M_world_frame.inverse() * M_world_joint;
    joint_to_frame_action.resize(num_thread, M_frame_to_joint.toActionMatrix());
  }
  size_t total = batch_size * seq_len;
  size_t n_coll = pairs.size();

  creq.resize(n_coll);
  cres.resize(n_coll);
  cdreq.resize(n_coll);
  cdres.resize(n_coll);
  for (size_t j = 0; j < n_coll; ++j) {
    creq[j].resize(total);
    cres[j].resize(total);
    cdreq[j].resize(total);
    cdres[j].resize(total);
    for (size_t i = 0; i < total; ++i) {
      creq[j][i] = coal::CollisionRequest();
      creq[j][i].security_margin = 1000;
      cres[j][i] = coal::CollisionResult();
      cdreq[j][i] = diffcoal::ContactDerivativeRequest();
      cdres[j][i] = diffcoal::ContactDerivative();
    }
  }
}

void single_forward_pass(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, size_t thread_id,
                         size_t batch_id, size_t seq_len, size_t cost_dim,
                         size_t eq_dim, size_t tool_id, pinocchio::SE3 T_star) {
  ZoneScopedN("single forward pass");
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
#endif

  T_star.rotation() =
      pinocchio::orthogonalProjection(T_star.rotation()); // Should be useless.
  double lambda = workspace.lambda;
  for (unsigned int time = 0; time < seq_len; time++) {
    ZoneScopedN("single forward pass iter");
    double *p_ptr = workspace.p_.data() + batch_id * seq_len * 6 + time * 6;
    Eigen::Map<Eigen::VectorXd> p(p_ptr, 6);
    size_t idx = batch_id * seq_len + time;
    double *A_ptr = workspace.A_.data() + (idx)*eq_dim * 6;
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A(A_ptr, static_cast<Eigen::Index>(eq_dim), 6);

    double *b_ptr =
        workspace.b_.data() + batch_id * seq_len * eq_dim + time * eq_dim;
    Eigen::Map<Eigen::VectorXd> b(b_ptr, static_cast<Eigen::Index>(eq_dim));

    double *q_ptr = workspace.positions_.data() +
                    batch_id * (seq_len + 1) * cost_dim + time * cost_dim;
    Eigen::Map<Eigen::VectorXd> q(q_ptr, static_cast<Eigen::Index>(cost_dim));
    double *q_next_ptr = workspace.positions_.data() +
                         batch_id * (seq_len + 1) * cost_dim +
                         (time + 1) * cost_dim;
    Eigen::Map<Eigen::VectorXd> q_next(q_next_ptr,
                                       static_cast<Eigen::Index>(cost_dim));

    pinocchio::Data &data = workspace.data_vec_[thread_id];
    pinocchio::framesForwardKinematics(model, data, q);
    Matrix6xd &jac = workspace.jacobians_[idx];
    jac.setZero();
    pinocchio::computeFrameJacobian(model, data, q, tool_id, pinocchio::LOCAL,
                                    jac);
    auto &ub = workspace.ub[thread_id];
    auto &lb = workspace.lb[thread_id];
    auto &G = workspace.G[thread_id];
    if constexpr (collisions) {
      pinocchio::updateFramePlacement(model, data, tool_id);
      pinocchio::updateGeometryPlacements(
          model, data, workspace.gmodel[thread_id], workspace.gdata[thread_id]);
      G.setZero();
      for (size_t n_coll = 0; n_coll < workspace.pairs.size(); ++n_coll) {
        auto [coll_a, coll_b] = workspace.pairs[n_coll];

        coal::CollisionResult &res = workspace.cres[n_coll][idx];
        coal::CollisionRequest &req = workspace.creq[n_coll][idx];
        diffcoal::ContactDerivative &dres = workspace.cdres[n_coll][idx];
        diffcoal::ContactDerivativeRequest &dreq = workspace.cdreq[n_coll][idx];
        dreq.derivative_type = diffcoal::ContactDerivativeType::FirstOrder;
        dreq.derivative_type =
            diffcoal::ContactDerivativeType::FiniteDifference;
        dreq.finite_differences_options.eps_fd = 1e-6;
        pinocchio::updateGlobalPlacements(model, data);

        coal::collide(
            &workspace.get_coal_obj(coll_a, batch_id),
            pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
            &workspace.get_coal_obj(coll_b, batch_id),
            pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]),
            req, res);
        if (res.getContact(0).penetration_depth < 0) {
          workspace.discarded[batch_id] = true;
          if (workspace.echo) {
            std::cout << "critical error collision" << std::endl;
            std::cout << "time : " << time << std::endl;
            std::cout << "distance" << res.getContact(0).penetration_depth
                      << std::endl;

            std::cout << "detected collisions at collision pair : " << coll_a
                      << "," << coll_b << std::endl;
            // throw std::runtime_error("Critical error: collision");
          }
          goto END;
          break;
        } else {
          ZoneScopedN("Collisions forward pass");
          diffcoal::computeContactDerivative(
              &workspace.get_coal_obj(coll_a, batch_id),
              pinocchio::toFclTransform3f(
                  workspace.gdata[thread_id].oMg[coll_a]),
              &workspace.get_coal_obj(coll_b, batch_id),
              pinocchio::toFclTransform3f(
                  workspace.gdata[thread_id].oMg[coll_b]),
              res.getContact(0), dreq, dres);
          size_t j1_id = workspace.geom_end_eff[batch_id]->parentJoint;
          size_t j2_id = workspace.geom_plane[batch_id]->parentJoint;
          auto &w1 = workspace.w1[thread_id];
          auto &w2 = workspace.w2[thread_id];
          auto &w_diff = workspace.w_diff[thread_id];
          auto &n = workspace.n[thread_id];
          auto &r1 = workspace.r1[thread_id];
          auto &r2 = workspace.r2[thread_id];
          w1 = res.getContact(0).nearest_points[0];
          w2 = res.getContact(0).nearest_points[1];
          w_diff.noalias() = w1 - w2;
          n = w_diff.normalized();
          auto &J_1 = workspace.J_1[thread_id];
          auto &J_2 = workspace.J_2[thread_id];
          pinocchio::computeJointJacobians(model, data, q);
          getJointJacobian(model, data, j1_id, pinocchio::LOCAL_WORLD_ALIGNED,
                           J_1);
          getJointJacobian(model, data, j2_id, pinocchio::LOCAL_WORLD_ALIGNED,
                           J_2);

          r1.noalias() = w1 - data.oMi[j1_id].translation();
          r2.noalias() = w2 - data.oMi[j2_id].translation();
          auto &J_coll = workspace.J_coll[thread_id];
          J_coll.noalias() = n.transpose() * J_1.block(0, 0, 3, model.nv) +
                             (pinocchio::skew(r1) * n).transpose() *
                                 J_1.block(3, 0, 3, model.nv);
          J_coll.noalias() -= n.transpose() * J_2.block(0, 0, 3, model.nv) +
                              (pinocchio::skew(r2) * n).transpose() *
                                  J_2.block(3, 0, 3, model.nv);
          G.row(n_coll) = -J_coll / workspace.dt;
          ub(n_coll) =
              workspace.collision_strength *
              (res.getContact(0).penetration_depth - workspace.safety_margin);
          lb(n_coll) = -1e10;
        }
      }
    }

    Eigen::MatrixXd &Q = workspace.Q_vec_[thread_id];
    Q.noalias() = jac.transpose() * jac;
    Q.diagonal().array() += workspace.q_reg;

    double *articular_speed_ptr = workspace.articular_speed_.data() +
                                  batch_id * seq_len * cost_dim +
                                  time * cost_dim;
    Eigen::Map<Eigen::VectorXd> articular_speed(
        articular_speed_ptr, static_cast<Eigen::Index>(cost_dim));

    pinocchio::SE3 &current_placement =
        workspace.current_placement_vec[thread_id];
    pinocchio::Motion target_lie(p.head(3), p.tail(3));
    pinocchio::SE3 &target_placement =
        workspace.target_placement_vec[thread_id];
    pinocchio::SE3 &diff = workspace.diff[idx];
    Matrix66d &adj_diff = workspace.adj_diff[idx];
    Vector6d &err = workspace.err_vec[thread_id];
    Vector6d &target = workspace.target_vec[thread_id];

    current_placement = data.oMf[tool_id];
    target_placement = pinocchio::exp6(target_lie);
    diff = current_placement.actInv(target_placement);
    workspace.target[idx] = target_lie;
    adj_diff = diff.toActionMatrixInverse();
    err = pinocchio::log6(diff).toVector();

// we set malloc allowed to true as after allocation was done once, future
// run won't allocate. // TODO we can allocate once in the fct allocate.
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif

    if (time > 0) {
      workspace.workspace_.warm_start_x[idx] =
          workspace.workspace_.qp[idx - 1]->results.x;
      workspace.workspace_.warm_start_eq[idx] = std::nullopt;
      if constexpr (collisions) {
        workspace.workspace_.warm_start_neq[idx] =
            workspace.workspace_.qp[idx - 1]->results.z;
      } else {
        workspace.workspace_.warm_start_neq[idx] = std::nullopt;
      }
    }
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif

    target.noalias() = lambda * jac.transpose() * err;
    if constexpr (collisions) {
      articular_speed.noalias() =
          QP(Q, target, workspace.A_thread_mem[thread_id], b,
             workspace.workspace_, thread_id, idx, G, lb, ub);
    } else {
      articular_speed.noalias() = QP(
          Q, target, workspace.A_thread_mem[thread_id], b, workspace.workspace_,
          thread_id, idx, std::nullopt, std::nullopt, std::nullopt);
    }
    q_next.noalias() = q + workspace.dt * articular_speed;

    double tracking_error = err.squaredNorm();
    if (time == seq_len - 1) {
      ZoneScopedN("final computations");
      Vector6d &log_vec = workspace.last_log_vec[thread_id];
      workspace.steps_per_batch[batch_id] = time;
      workspace.last_q[batch_id].noalias() = q_next;
      workspace.last_T[batch_id] = data.oMf[tool_id].actInv(T_star);
      pinocchio::framesForwardKinematics(model, data, q_next);
      workspace.last_T[batch_id] = data.oMf[tool_id].actInv(T_star);
      workspace.last_logT[batch_id] =
          pinocchio::log6(workspace.last_T[batch_id]);
      log_vec = workspace.last_logT[batch_id].toVector();
      log_vec.tail(3) *= workspace.rot_w;
      double loss_L2 = log_vec.squaredNorm();
      double loss_L1 = log_vec.lpNorm<1>();
      workspace.losses[batch_id] = loss_L2 + workspace.lambda_L1 * loss_L1;
      if (discard_non_convergent_ik && tracking_error > 1e-7) {
        workspace.discarded[batch_id] = true;
        double *p_ptr2 = workspace.p_.data() + batch_id * seq_len * 6;
        Eigen::Map<Eigen::VectorXd> p2(p_ptr2, 6);
        double *q_ptr2 =
            workspace.positions_.data() + batch_id * (seq_len + 1) * cost_dim;
        Eigen::Map<Eigen::VectorXd> q2(q_ptr2,
                                       static_cast<Eigen::Index>(cost_dim));

        std::cerr << "\n[WARNING] Reached end of simulation without achieving "
                     "a decent tracking error\n Error is : "
                  << tracking_error << "\n"
                  << "and batch position is" << batch_id << "\n";
        std::cout << "T_star" << T_star << "\n";
        std::cout << "target_placement" << target_placement << "\n";
        std::cout << "p" << p2 << "\n";
        std::cout << "q0" << q2 << "\n";
      }
      break;
    }
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
  }
END:
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(true);
#endif
}

Eigen::VectorXd
forward_pass2(QP_pass_workspace2 &workspace,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, size_t num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
              double dt) {
  const size_t batch_size = static_cast<size_t>(p.dimension(0));
  const size_t seq_len = static_cast<size_t>(p.dimension(1));
  const size_t eq_dim = static_cast<size_t>(A.dimension(1));
  const size_t cost_dim = static_cast<size_t>(model.nv);
  workspace.positions_.setZero();
  assert(static_cast<int>(workspace.tool_id) != -1 &&
         "You must set workspace's tool id. (workspace.set_tool_id(tool_id))");

  workspace.allocate(model, batch_size, seq_len, cost_dim, eq_dim, num_thread);
  workspace.init_geometry(model, batch_size);

  workspace.dt = dt;
  workspace.b_ = b;
  workspace.A_ = A;
  workspace.p_ = p;
  for (size_t i = 0; i < workspace.discarded.size(); ++i) {
    workspace.discarded[i] = false;
  }

  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    double *q_ptr =
        workspace.positions_.data() + batch_id * (seq_len + 1) * cost_dim;
    Eigen::Map<Eigen::VectorXd> q(q_ptr, static_cast<Eigen::Index>(cost_dim));
    q = initial_position.row(static_cast<Eigen::Index>(batch_id));
  }
  omp_set_num_threads(num_thread);

#pragma omp parallel for schedule(dynamic, 1)
  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
    single_forward_pass(workspace, model, thread_id, batch_id, seq_len,
                        cost_dim, eq_dim, workspace.tool_id, T_star[batch_id]);
  }
  return workspace.losses;
}

void compute_frame_hessian(QP_pass_workspace2 &workspace,
                           const pinocchio::Model &model, size_t thread_id,
                           size_t tool_id, pinocchio::Data &data,
                           const Eigen::Ref<Eigen::VectorXd> q) {
  ZoneScopedN("compute frame hessian");
  auto &H1 = workspace.Hessian[thread_id];
  auto &H2 = workspace.Hessian[workspace.num_thread_ + thread_id];
  H1.setZero();
  H2.setZero();

  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::computeJointKinematicHessians(model, data, q);
  pinocchio::getJointKinematicHessian(
      model, data, model.frames[tool_id].parentJoint, pinocchio::LOCAL, H2);
  auto &m = workspace.joint_to_frame_action[thread_id];
  Eigen::TensorMap<Eigen::Tensor<const double, 2>> X_tensor(m.data(), 6, 6);

  // Tensor.contract malloc for some reasons
  for (int i = 0; i < H1.dimension(0); ++i)
    for (int j = 0; j < H1.dimension(1); ++j)
      for (int l = 0; l < H1.dimension(2); ++l)
        for (int k = 0; k < H2.dimension(0); ++k)
          H1(i, j, l) += X_tensor(i, k) * H2(k, j, l);
}

void backpropagateThroughQ(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                           QP_pass_workspace2 &workspace, size_t thread_id) {
  ZoneScopedN("backpropagate through Q");
  Eigen::Map<Eigen::VectorXd> g(workspace.grad_J_[thread_id].data(),
                                6 * workspace.cost_dim_);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor>>
      H(workspace.Hessian[thread_id].data(), 6 * workspace.cost_dim_,
        workspace.cost_dim_);
  auto &temp = workspace.temp_direct[thread_id];
  temp.noalias() = H.transpose() * g;
  grad_vec_local.noalias() += workspace.dt * temp;
}

void backpropagateThroughJ0(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                            const pinocchio::Model &model,
                            const pinocchio::SE3 &diff,
                            Eigen::Ref<const Eigen::VectorXd> rhs_grad,
                            double lambda, QP_pass_workspace2 &workspace,
                            size_t thread_id) {

  ZoneScopedN("backpropagate through J0");

  const size_t nv = static_cast<size_t>(model.nv);
  auto &temp = workspace.temp[thread_id];
  auto &log = workspace.log_indirect_1_vec[thread_id];
  const auto &H = workspace.Hessian[thread_id];

  log = lambda * pinocchio::log6(diff).toVector();
  const auto rhs_head = rhs_grad.head(nv);
  for (unsigned int j = 0; j < nv; ++j) {
    double acc = 0.0;
    for (unsigned int l = 0; l < nv; ++l) {
      double rhs_l = rhs_head(l);
      for (unsigned int k = 0; k < 6; ++k) {
        acc -= rhs_l * H(k, l, j) * log(k);
      }
    }
    temp(j) = acc;
  }

  grad_vec_local.noalias() += workspace.dt * temp;
}

void backpropagateThroughT(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                           const pinocchio::Model &model, pinocchio::SE3 &diff,
                           Eigen::Ref<Eigen::VectorXd> rhs_grad, double lambda,
                           QP_pass_workspace2 &workspace, size_t thread_id,
                           size_t batch_id, size_t time) {
  ZoneScopedN("backpropagate through T");
  const double scale = lambda * workspace.dt;
  const auto rhs_q = rhs_grad.head(model.nv);
  const size_t idx = batch_id * workspace.seq_len_ + time;
  const auto &J = workspace.jacobians_[idx];
  const auto &adj = workspace.adj_diff[idx];
  auto &Jlog = workspace.Jlog_v4[thread_id];
  Jlog = pinocchio::Jlog6(diff);

  auto &v1 = workspace.v1[thread_id];
  v1 = J * rhs_q;
  auto &v2 = workspace.v2[thread_id];
  v2 = Jlog.transpose() * v1;
  auto &v3 = workspace.v3[thread_id];
  v3 = adj.transpose() * v2;
  grad_vec_local.noalias() += (scale * J.transpose()) * v3;
}

void backpropagateThroughCollisions(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                                    Eigen::Ref<Eigen::MatrixXd> dJcoll_dq,
                                    Eigen::Ref<Eigen::VectorXd> ddist,
                                    QP_pass_workspace2 &workspace, size_t time,
                                    size_t batch_id, size_t seq_len,
                                    size_t n_coll) {
  ZoneScopedN("backpropagate through collisions");
  grad_vec_local.noalias() +=
      workspace.collision_strength *
      workspace.workspace_.qp[batch_id * seq_len + time]
          ->model.backward_data.dL_du(n_coll) *
      ddist * workspace.dt;
  grad_vec_local.noalias() -=
      workspace.workspace_.qp[batch_id * seq_len + time]
          ->model.backward_data.dL_dC.row(n_coll) *
      dJcoll_dq;
}

void compute_dn_dq(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t j1_id, size_t j2_id,
                   size_t batch_id, size_t time, Eigen::Ref<Matrix3xd> dn_dq_,
                   size_t thread_id, size_t n_coll) {
  ZoneScopedN("compute dn/dq");
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  J1.setZero();
  // J2.setZero();
  auto [coll_a, coll_b] = workspace.pairs[n_coll];
  pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
  // pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
  dn_dq_.noalias() =
      (workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
           .dnormal_dM1 *
       workspace.get_coll_pos(coll_a, batch_id).toActionMatrixInverse() *
       J1); /* +
    workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
            .dnormal_dM2 *
        workspace.get_coll_pos(coll_b, batch_id).toActionMatrixInverse() *
        J2);*/
}

void compute_d_dist_and_d_Jcoll(QP_pass_workspace2 &workspace,
                                const pinocchio::Model &model,
                                pinocchio::Data &data, size_t j1_id,
                                size_t j2_id, size_t batch_id, size_t time,
                                Eigen::Ref<Eigen::VectorXd> ddist,
                                Eigen::Ref<Eigen::MatrixXd> dJcoll_dq,
                                size_t thread_id, Eigen::Ref<Eigen::VectorXd> q,
                                size_t n_coll) {
  ZoneScopedN("compute ddist/dq & dJcoll/dq start");
  auto [coll_a, coll_b] = workspace.pairs[n_coll];
  auto &term_A = workspace.term_A[thread_id];
  auto &w1 = workspace.w1[thread_id];
  auto &w2 = workspace.w2[thread_id];
  auto &w_diff = workspace.w_diff[thread_id];
  auto &r1 = workspace.r1[thread_id];
  auto &J_diff = workspace.J_diff[thread_id];
  auto &term_B = workspace.term_B[thread_id];
  coal::CollisionResult &cres =
      workspace.cres[n_coll][batch_id * workspace.seq_len_ + time];
  diffcoal::ContactDerivative &cdres =
      workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time];
  auto &dn_dq = workspace.dn_dq[thread_id];
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  const Eigen::Vector3d &n = workspace.n[thread_id];
  Eigen::Matrix3d &R = workspace.R[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &dr1_dq =
      workspace.dr1_dq[thread_id];
  Eigen::MatrixXd &dout = workspace.dout[thread_id];
  Eigen::Vector3d &c = workspace.c[thread_id];
  Eigen::Vector3d &dcj = workspace.dcj[thread_id];
  Eigen::RowVectorXd &term1 = workspace.term1[thread_id];
  Eigen::Tensor<double, 3> &H1 =
      workspace.Hessian[workspace.num_thread_ + thread_id];
  Eigen::Tensor<double, 3> &H2 = workspace.Hessian[thread_id];
  Eigen::Matrix<double, 6, Eigen::Dynamic> &tmp =
      workspace.temp_tensor[thread_id];
  Eigen::MatrixXd &M = workspace.M[thread_id];
  Eigen::RowVector<double, 6> &temp_ddist = workspace.temp_ddist[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp1 = workspace.tmp1_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp2 = workspace.tmp2_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp3 =
      workspace.tmp3_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp4 =
      workspace.tmp4_dr1_dq[thread_id];

  pinocchio::computeJointJacobians(model, data, q);
  compute_dn_dq(workspace, model, data, j1_id, j2_id, batch_id, time, dn_dq,
                thread_id, n_coll);
  {
    ZoneScopedN("compute ddist");
    J1.setZero();
    // J2.setZero();
    pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
    // pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
    temp_ddist.noalias() =
        workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
            .ddist_dM1.transpose() *
        workspace.get_coll_pos(coll_a, batch_id).toActionMatrixInverse();
    ddist.noalias() = J1.transpose() * temp_ddist.transpose();
  }
  dJcoll_dq.setZero();
  {
    ZoneScopedN("compute kine + kinederivatives + hessian");
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, q, q);
    pinocchio::computeJointKinematicHessians(model, data);
    J1.setZero();
    // J2.setZero();
    pinocchio::getJointJacobian(model, data, j1_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J1);
    // pinocchio::getJointJacobian(model, data, j2_id,
    //                             pinocchio::LOCAL_WORLD_ALIGNED, J2);
    H1.setZero();
    // H2.setZero();
    pinocchio::getJointKinematicHessian(model, data, j1_id,
                                        pinocchio::LOCAL_WORLD_ALIGNED, H1);
    // pinocchio::getJointKinematicHessian(model, data, j2_id,
    //                                     pinocchio::LOCAL_WORLD_ALIGNED, H2);
  }
  {
    ZoneScopedN("term A and B");
    term_A.setZero();
    w1 = cres.getContact(0).nearest_points[0];
    w2 = cres.getContact(0).nearest_points[1];
    w_diff = w1 - w2;
    workspace.n[thread_id] = w_diff.normalized();

    int j_dim = H1.dimension(1);
    int q_dim = H1.dimension(2);
    for (int qqq = 0; qqq < q_dim; ++qqq) {
      for (int j = 0; j < j_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < 3; ++i) {
          s += n(i) * H1(i, j, qqq);
        }
        term_A(j, qqq) = s;
      }
    }
    J_diff.noalias() = J1.topRows(3); // - J2.topRows(3);
    term_B.noalias() = -J_diff.transpose() * dn_dq;
  }
  {
    ZoneScopedN("term c and d");
    J1.setZero();
    // J2.setZero();
    pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
    // pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
    r1.noalias() = w1 - data.oMi[j1_id].translation();
    R = data.oMi[j1_id].rotation();
    tmp1.noalias() = cdres.dcpos_dM1;
    tmp1.noalias() -= cdres.dvsep_dM1 * 0.5;
    tmp2.noalias() =
        tmp1 * workspace.get_coll_pos(coll_a, batch_id).toActionMatrixInverse();
    tmp3.noalias() = tmp2 * J1;
    tmp4.noalias() = R * J1.topRows(3);
    dr1_dq.noalias() = tmp3 - tmp4;

    pinocchio::getJointJacobian(model, data, j1_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J1);
    // pinocchio::getJointJacobian(model, data, j2_id,
    //                             pinocchio::LOCAL_WORLD_ALIGNED, J2);
    dout.setZero();
    c = r1.cross(n);
    for (int j = 0; j < model.nv; ++j) {
      dcj.noalias() = dr1_dq.col(j).template head<3>().cross(n) -
                      r1.cross(dn_dq.col(j).template head<3>());
      term1.noalias() = dcj.transpose() * J1.bottomRows(3);
      dout.col(j) = term1;
    }
    Eigen::DSizes<ptrdiff_t, 3> off(3, 0, 0),
        ext(3, H1.dimension(1), H1.dimension(2));

    tmp.setZero();
    for (int l = 0; l < 3; ++l)
      for (int j = 0; j < tmp.cols(); ++j)
        for (int i = 0; i < tmp.rows(); ++i)
          tmp(i, j) += c[l] * H1(i, j, l);
  }
  dJcoll_dq = term_A;
  dJcoll_dq += term_B;
  dJcoll_dq += dout;
  dJcoll_dq += tmp;
}

void single_backward_pass(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    size_t thread_id, size_t batch_id, size_t seq_len, size_t cost_dim,
    size_t tool_id, double dt,
    Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output) {
  if (workspace.discarded[batch_id]) {
    // element is discarded so gradient stays 0.
  } else {
    ZoneScopedN("single backward pass");

#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    auto &w = workspace.w_vec[thread_id];
    auto &e = workspace.e_vec[thread_id];
    Vector6d &e_scaled = workspace.e_scaled_vec[thread_id];
    pinocchio::Data &data = workspace.data_vec_[thread_id];
    auto &grad_e = workspace.grad_e_vec[thread_id];
    auto &sign_e_scaled = workspace.sign_e_scaled_vec[thread_id];
    auto &Adj = workspace.Adj_vec[thread_id];
    auto &Jlog = workspace.Jlog_vec[thread_id];
    auto &J_frame = workspace.J_frame_vec[thread_id];
    auto &dloss_dq = workspace.dloss_dq[batch_id];
    auto &dloss_dq_diff = workspace.dloss_dq_diff[batch_id];

    w << 1, 1, 1, workspace.rot_w, workspace.rot_w, workspace.rot_w;
    e = workspace.last_logT[batch_id].toVector();
    e_scaled = w.array() * e.array();

    grad_e = 2.0 * e_scaled.array() * w.array();
    sign_e_scaled = e_scaled.unaryExpr(
        [](double x) { return static_cast<double>((x > 0) - (x < 0)); });
    grad_e +=
        (workspace.lambda_L1 * sign_e_scaled.array() * w.array()).matrix();
    J_frame.setZero();
    Adj.setZero();
    Jlog.setZero();
    Adj = workspace.last_T[batch_id].toActionMatrixInverse();
    Jlog = pinocchio::Jlog6(workspace.last_T[batch_id]);
    pinocchio::computeFrameJacobian(model, data, workspace.last_q[batch_id],
                                    tool_id, pinocchio::LOCAL, J_frame);

    workspace.dloss_dq_tmp1[thread_id].noalias() =
        (-Adj.transpose()) * Jlog.transpose();
    workspace.dloss_dq_tmp2[thread_id].noalias() =
        J_frame.transpose() * workspace.dloss_dq_tmp1[thread_id];
    workspace.dloss_dq_tmp3[thread_id].noalias() =
        workspace.dloss_dq_tmp2[thread_id] * grad_e;
    dloss_dq.noalias() = dt * workspace.dloss_dq_tmp3[thread_id];
    dloss_dq_diff.setZero();

    for (Eigen::Index time =
             static_cast<Eigen::Index>(workspace.steps_per_batch[batch_id]);
         time >= 0; time--) {
      auto &grad_target = workspace.grad_target_vec[thread_id];
      size_t idx = batch_id * seq_len + time;
      double lambda = workspace.lambda;
      size_t grad_dim = static_cast<size_t>(grad_output.dimension(2));
      Eigen::Map<Eigen::VectorXd> grad_vec(
          grad_output.data() + batch_id * seq_len * grad_dim + time * grad_dim,
          static_cast<Eigen::Index>(grad_dim));
      const size_t nv = static_cast<size_t>(model.nv);

      auto &padded = workspace.padded[thread_id];
      padded.setZero();
      padded.head(nv) = dloss_dq + dloss_dq_diff;
      QP_backward(workspace.workspace_, padded, idx);

      auto &KKT_grad = workspace.workspace_.grad_KKT_mem_[idx];
      auto &rhs_grad = workspace.workspace_.grad_rhs_mem_[idx];

      workspace.grad_err_[idx] =
          -workspace.jacobians_[idx] * rhs_grad.head(model.nv) * lambda;
      pinocchio::SE3 &diff = workspace.diff[idx];
      grad_target =
          pinocchio::Jlog6(diff).transpose() * workspace.grad_err_[idx];
      auto &grad_p = workspace.grad_p_[idx];
      grad_p.noalias() =
          pinocchio::Jexp6(workspace.target[idx]).transpose() * grad_target;

      auto &J = workspace.jacobians_[idx];
      workspace.grad_J_[thread_id].noalias() =
          2 * J * KKT_grad.block(0, 0, cost_dim, cost_dim);
      auto &ddist = workspace.ddist[thread_id];
      auto &dJcoll_dq = workspace.dJcoll_dq[thread_id];
      double *q_ptr = workspace.positions_.data() +
                      batch_id * (seq_len + 1) * cost_dim + (time)*cost_dim;
      const Eigen::Map<Eigen::VectorXd> q(q_ptr,
                                          static_cast<Eigen::Index>(cost_dim));

      compute_frame_hessian(workspace, model, thread_id, tool_id, data, q);

      backpropagateThroughQ(dloss_dq_diff, workspace, thread_id);
      backpropagateThroughJ0(dloss_dq_diff, model, diff, rhs_grad, lambda,
                             workspace, thread_id);
      backpropagateThroughT(dloss_dq_diff, model, diff, rhs_grad, lambda,
                            workspace, thread_id, batch_id, time);
      if constexpr (collisions) {
        for (size_t n_coll = 0; n_coll < workspace.pairs.size(); ++n_coll) {
          auto [coll_a, coll_b] = workspace.pairs[n_coll];
          compute_d_dist_and_d_Jcoll(
              workspace, model, data,
              workspace.get_geom(coll_a, batch_id).parentJoint,
              workspace.get_geom(coll_b, batch_id).parentJoint, batch_id, time,
              ddist, dJcoll_dq, thread_id, q, n_coll);
          backpropagateThroughCollisions(dloss_dq_diff, dJcoll_dq, ddist,
                                         workspace, time, batch_id, seq_len,
                                         n_coll);
        }
      }
    }
  }
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(true);
#endif
}

void backward_pass2(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
    size_t num_thread, size_t batch_size) {
  size_t cost_dim = static_cast<size_t>(model.nv);
  size_t seq_len = static_cast<size_t>(workspace.b_.dimension(1));
  size_t tool_id = static_cast<size_t>(workspace.tool_id);
  double dt = workspace.dt;

  for (auto vec = workspace.grad_p_.begin(); vec != workspace.grad_p_.end();
       ++vec) {
    vec->setZero();
  }

  omp_set_num_threads(static_cast<int>(num_thread));
#pragma omp parallel for schedule(dynamic, 1)
  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    size_t thread_id = static_cast<size_t>(omp_get_thread_num());
    single_backward_pass(workspace, model, thread_id, batch_id, seq_len,
                         cost_dim, tool_id, dt, grad_output);
  }
}
