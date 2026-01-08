#include "dik_cols.hpp"
#include "pinocchio/algorithm/center-of-mass-derivatives.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
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
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <tracy/Tracy.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

using Vector6d = Eigen::Vector<double, 6>;
using Matrix66d = Eigen::Matrix<double, 6, 6>;
using Matrix3xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

void QP_pass_workspace2::reset() {}

void QP_pass_workspace2::init_geometry(pinocchio::Model model,
                                       size_t batch_size) {
  ZoneScopedN("init geometry");
  gmodel.clear();
  gdata.clear();
  geom_end_eff.clear();
  geom_arm.clear();
  geom_arm_1.clear();
  geom_arm_2.clear();
  geom_arm_3.clear();
  geom_arm_4.clear();
  geom_arm_5.clear();
  geom_plane.clear();
  geom_cylinder.clear();
  geom_ball.clear();
  geom_box1.clear();
  geom_box2.clear();
  geom_box3.clear();
  geom_box4.clear();

  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    geom_end_eff.emplace_back();
    auto frame_ids = parent_frames[batch_id];
    geom_end_eff[batch_id] = pinocchio::GeometryObject(
        "end eff", frame_ids[0], model.frames[frame_ids[0]].parentJoint,
        std::make_shared<coal::Sphere>(effector_ball[batch_id]),
        pinocchio::SE3(end_eff_rot[batch_id], end_eff_pos[batch_id]));

    geom_arm.emplace_back();
    geom_arm[batch_id] = pinocchio::GeometryObject(
        "arm", frame_ids[1], model.frames[frame_ids[1]].parentJoint,
        std::make_shared<coal::Cylinder>(arm[batch_id]),
        pinocchio::SE3(arm_rot[batch_id], arm_pos[batch_id]));

    geom_arm_1.emplace_back();
    geom_arm_1[batch_id] = pinocchio::GeometryObject(
        "arm1", frame_ids[2], model.frames[frame_ids[2]].parentJoint,
        std::make_shared<coal::Sphere>(arm_1[batch_id]),
        pinocchio::SE3(arm_1_rot[batch_id], arm_1_pos[batch_id]));

    geom_arm_2.emplace_back();
    geom_arm_2[batch_id] = pinocchio::GeometryObject(
        "arm2", frame_ids[3], model.frames[frame_ids[3]].parentJoint,
        std::make_shared<coal::Sphere>(arm_2[batch_id]),
        pinocchio::SE3(arm_2_rot[batch_id], arm_2_pos[batch_id]));

    geom_arm_3.emplace_back();
    geom_arm_3[batch_id] = pinocchio::GeometryObject(
        "arm3", frame_ids[4], model.frames[frame_ids[4]].parentJoint,
        std::make_shared<coal::Sphere>(arm_3[batch_id]),
        pinocchio::SE3(arm_3_rot[batch_id], arm_3_pos[batch_id]));

    geom_plane.emplace_back();
    geom_plane[batch_id] = pinocchio::GeometryObject(
        "plane", frame_ids[5], model.frames[frame_ids[5]].parentJoint,
        std::make_shared<coal::Box>(plane[batch_id]),
        pinocchio::SE3(plane_rot[batch_id], plane_pos[batch_id]));

    geom_cylinder.emplace_back();
    geom_cylinder[batch_id] = pinocchio::GeometryObject(
        "cylinder", frame_ids[6], model.frames[frame_ids[6]].parentJoint,
        std::make_shared<coal::Capsule>(cylinder[batch_id]),
        pinocchio::SE3(cylinder_rot[batch_id], cylinder_pos[batch_id]));

    geom_ball.emplace_back();
    geom_ball[batch_id] = pinocchio::GeometryObject(
        "ball", frame_ids[7], model.frames[frame_ids[7]].parentJoint,
        std::make_shared<coal::Sphere>(ball[batch_id]),
        pinocchio::SE3(ball_rot[batch_id], ball_pos[batch_id]));

    geom_box1.emplace_back();
    geom_box1[batch_id] = pinocchio::GeometryObject(
        "box1", frame_ids[8], model.frames[frame_ids[8]].parentJoint,
        std::make_shared<coal::Box>(box1[batch_id]),
        pinocchio::SE3(box_rot1[batch_id], box_pos1[batch_id]));

    geom_box2.emplace_back();
    geom_box2[batch_id] = pinocchio::GeometryObject(
        "box2", frame_ids[9], model.frames[frame_ids[9]].parentJoint,
        std::make_shared<coal::Box>(box2[batch_id]),
        pinocchio::SE3(box_rot2[batch_id], box_pos2[batch_id]));

    geom_box3.emplace_back();
    geom_box3[batch_id] = pinocchio::GeometryObject(
        "box3", frame_ids[10], model.frames[frame_ids[10]].parentJoint,
        std::make_shared<coal::Box>(box3[batch_id]),
        pinocchio::SE3(box_rot3[batch_id], box_pos3[batch_id]));

    geom_box4.emplace_back();
    geom_box4[batch_id] = pinocchio::GeometryObject(
        "box4", frame_ids[11], model.frames[frame_ids[11]].parentJoint,
        std::make_shared<coal::Box>(box4[batch_id]),
        pinocchio::SE3(box_rot4[batch_id], box_pos4[batch_id]));

    geom_arm_4.emplace_back();
    geom_arm_4[batch_id] = pinocchio::GeometryObject(
        "arm4", frame_ids[12], model.frames[frame_ids[12]].parentJoint,
        std::make_shared<coal::Sphere>(arm_4[batch_id]),
        pinocchio::SE3(arm_4_rot[batch_id], arm_4_pos[batch_id]));

    geom_arm_5.emplace_back();
    geom_arm_5[batch_id] = pinocchio::GeometryObject(
        "arm5", frame_ids[13], model.frames[frame_ids[13]].parentJoint,
        std::make_shared<coal::Sphere>(arm_5[batch_id]),
        pinocchio::SE3(arm_5_rot[batch_id], arm_5_pos[batch_id]));

    for (size_t i = 0; i < num_thread_; ++i) {
      gmodel.emplace_back();
      gmodel[i].addGeometryObject(geom_end_eff[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_1[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_2[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_3[batch_id].value());
      gmodel[i].addGeometryObject(geom_plane[batch_id].value());
      gmodel[i].addGeometryObject(geom_cylinder[batch_id].value());
      gmodel[i].addGeometryObject(geom_ball[batch_id].value());
      gmodel[i].addGeometryObject(geom_box1[batch_id].value());
      gmodel[i].addGeometryObject(geom_box2[batch_id].value());
      gmodel[i].addGeometryObject(geom_box3[batch_id].value());
      gmodel[i].addGeometryObject(geom_box4[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_4[batch_id].value());
      gmodel[i].addGeometryObject(geom_arm_5[batch_id].value());
      gdata.emplace_back();
      gdata[i] = pinocchio::GeometryData(gmodel[i]);
    }
  }
}

Eigen::Tensor<double, 3, Eigen::RowMajor> QP_pass_workspace2::Get_positions_() {
  return positions_;
}

void QP_pass_workspace2::set_L1_weight(double weight) {
  if (weight < 0.0) {
    spdlog::error("set_L1_weight(): weight must be  positive (got {:.3f})",
                  weight);
    throw std::invalid_argument("L1 weight must be positive");
  }
  lambda_L1 = weight;
}

void QP_pass_workspace2::set_rot_weight(double weight) {
  if (weight < 0.0) {
    spdlog::error("set_rot_weight(): weight must be positive (got {:.3f})",
                  weight);
    throw std::invalid_argument("Rotation weight must be positive");
  }
  rot_w = weight;
}

void QP_pass_workspace2::set_q_reg(double q_reg_) {
  if (q_reg_ <= 0.0) {
    spdlog::error("set_q_reg(): q_reg must be strictly positive (got {:.3f})",
                  q_reg_);
    throw std::invalid_argument("q_reg must be positive");
  }
  q_reg = q_reg_;
}

void QP_pass_workspace2::set_lambda(double lambda_) {
  if (lambda_ >= 0.0) {
    spdlog::error("set_lambda(): lambda must be strictly negative (got {:.3f})",
                  lambda_);
    throw std::invalid_argument("lambda must be negative");
  }
  lambda = lambda_;
}

void QP_pass_workspace2::set_tool_id(size_t id) { tool_id = id; }

void QP_pass_workspace2::set_collisions_strength(double strength) {
  if (strength < 0.0) {
    spdlog::error("set_collisions_strength(): strength must be "
                  "positive (got {:.3f})",
                  strength);
    throw std::invalid_argument("collision strength must be positive");
  }
  collision_strength = strength;
}

std::vector<Eigen::VectorXd> QP_pass_workspace2::get_last_q() {
  return last_q;
};

std::vector<Vector6d> QP_pass_workspace2::grad_log_target() {
  return grad_log_target_;
};

Eigen::Ref<Eigen::VectorXd> QP_pass_workspace2::dloss_dqf(size_t i) {
  return dloss_dq[i];
}

void QP_pass_workspace2::set_collisions_safety_margin(double margin) {
  if (collisions) {
    spdlog::info("Collisions are activated (safety margin = {:.3f})", margin);
  } else {
    spdlog::info("Collisions are deactivated (safety margin = {:.3f})", margin);
  }
  safety_margin = margin;
}

void QP_pass_workspace2::pre_allocate(size_t batch_size) {
  batch_size_ = batch_size;
  ZoneScopedN("pre allocate");
  if (effector_ball.size() != batch_size) {
    end_eff_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_1_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_2_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_3_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_4_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    arm_5_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    plane_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    cylinder_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    ball_pos.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    box_pos1.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    box_pos2.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    box_pos3.resize(batch_size, Eigen::Vector<double, 3>::Zero());
    box_pos4.resize(batch_size, Eigen::Vector<double, 3>::Zero());

    end_eff_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_1_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_2_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_3_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_4_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    arm_5_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    plane_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    cylinder_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    ball_rot.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    box_rot1.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    box_rot2.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    box_rot3.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());
    box_rot4.resize(batch_size, Eigen::Matrix<double, 3, 3>::Identity());

    effector_ball.resize(batch_size, coal::Sphere(0.1));
    arm.resize(batch_size, coal::Cylinder(0.05, 0.5));
    arm_1.resize(batch_size, coal::Sphere(0.08));
    arm_2.resize(batch_size, coal::Sphere(0.10));
    arm_3.resize(batch_size, coal::Sphere(0.08));
    arm_4.resize(batch_size, coal::Sphere(0.05));
    arm_5.resize(batch_size, coal::Sphere(0.05));
    plane.resize(batch_size, coal::Box(1e6, 1e6, 10));
    cylinder.resize(batch_size, coal::Capsule(0.05, 0.4));
    ball.resize(batch_size);
    box1.resize(batch_size);
    box2.resize(batch_size);
    box3.resize(batch_size);
    box4.resize(batch_size);
  }
  pre_allocated = true;
  parent_frames.resize(batch_size);
  for (auto &v : parent_frames) {
    v.resize(14);
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
        if (equilibrium) {
          strategy = 2;
        } else {
          strategy = 2;
        }
      } else {
        equilibrium = false;
        strategy = 3;
      }
    } else {
      strategy = 2;
    }
    if (equilibrium) {
      workspace_.allocate(batch_size * seq_len, model.nv, eq_dim, num_thread,
                          strategy, 4);
    } else {
      workspace_.allocate(batch_size * seq_len, model.nv, eq_dim, num_thread,
                          strategy, pairs.size());
    }
    log_target.resize(batch_size, seq_len, 6);
    log_target.setZero();

    A_.resize(batch_size * seq_len, eq_dim, model.nv);
    A_.setZero();

    b_.resize(batch_size, seq_len, eq_dim);
    b_.setZero();

    positions_.resize(batch_size, seq_len + 1, model.nv);
    positions_.setZero();

    articular_speed_.resize(batch_size, seq_len, model.nv);
    articular_speed_.setZero();

    discarded.resize(batch_size, false);
    last_q.resize(batch_size, Eigen::VectorXd::Zero(model.nv));

    dloss_dq.resize(batch_size, Eigen::VectorXd::Zero(model.nv));
    dloss_dq_diff.resize(batch_size, Eigen::VectorXd::Zero(model.nv));
    last_T.resize(batch_size, pinocchio::SE3::Identity());
    last_logT.resize(batch_size, pinocchio::Motion::Zero());

    losses = Eigen::VectorXd::Zero(batch_size);
    steps_per_batch.resize(batch_size, 0);
    intermediate_goals.resize(batch_size);
    intermediate_geom_goals.resize(batch_size);

    const size_t total = batch_size * seq_len;
    jacobians_.resize(total, Matrix6xd::Zero(6, model.nv));
    adj_diff.resize(total, Matrix66d::Zero());
    grad_err_.resize(total, Vector6d::Zero());
    grad_log_target_.resize(total, Vector6d::Zero());
    diff.resize(total, pinocchio::SE3::Identity());
    target.resize(total, pinocchio::Motion::Zero());

    const size_t n_thread = num_thread;

    Hessian.clear();
    Hessian.reserve(2 * n_thread);
    for (unsigned int i = 0; i < 2 * n_thread; ++i) {
      Hessian.emplace_back();
      Hessian.back().resize(6, model.nv, model.nv);
      Hessian.back().setZero();
    }
    data_vec_.clear();
    data_vec_.reserve(n_thread);
    gmodel.reserve(n_thread);
    gdata.reserve(n_thread);
    Q_vec_.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    grad_J_.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    J_coll.resize(n_thread, Eigen::RowVectorXd::Zero(model.nv));
    J1.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    J2.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    J_1.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    J_2.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    A_thread_mem.resize(
        n_thread,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(eq_dim), model.nv));
    term_1_A.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    term_2_B.resize(n_thread,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                        model.nv, model.nv));
    M.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    term_2_A.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    term_1_B.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    J_diff.resize(n_thread,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    dJcoll_dq.resize(n_thread, Eigen::MatrixXd::Zero(model.nv, model.nv));
    R.resize(n_thread, Eigen::Matrix3d::Identity());
    c.resize(num_thread, Eigen::Vector3d::Zero());
    dcj.resize(n_thread, Eigen::Vector3d::Zero());
    term1.resize(num_thread, Eigen::VectorXd::Zero(model.nv));
    dr1_dq.resize(n_thread,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    dr2_dq.resize(n_thread,
                  Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    tmp1_dr1_dq.resize(n_thread, Eigen::Matrix<double, 3, 6>::Zero());
    tmp2_dr1_dq.resize(n_thread, Eigen::Matrix<double, 3, 6>::Zero());
    tmp3_dr1_dq.resize(
        n_thread, Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    tmp4_dr1_dq.resize(
        n_thread, Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, model.nv));
    Adj_vec.resize(n_thread, Matrix66d::Zero());
    dloss_dq_tmp1.resize(n_thread, Matrix66d::Zero());
    dloss_dq_tmp2.resize(n_thread, Eigen::MatrixXd::Zero(
                                       model.nv, static_cast<Eigen::Index>(6)));
    dloss_dq_tmp3.resize(n_thread, Eigen::VectorXd::Zero(model.nv));

    Jlog_vec.resize(n_thread, Matrix66d::Zero());
    Jlog_v4.resize(n_thread, Matrix66d::Zero());
    J_frame_vec.resize(n_thread, Matrix6xd::Zero(6, model.nv));
    padded.resize(n_thread);

    for (auto &v : padded)
      v = Eigen::VectorXd::Zero(
          static_cast<Eigen::Index>(model.nv + pairs.size()));
    ddist.resize(n_thread, Eigen::VectorXd::Zero(model.nv));
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
                 pairs.size(), model.nv));
    temp.resize(n_thread, Eigen::VectorXd::Zero(model.nv));
    target_vec.resize(n_thread, Eigen::VectorXd(model.nv).setConstant(0));
    err_vec.resize(n_thread, Vector6d().setConstant(0));
    last_log_vec.resize(n_thread, Vector6d().setConstant(0));
    w_vec.resize(n_thread, Vector6d().setConstant(0));
    e_vec.resize(n_thread, Vector6d().setConstant(0));
    temp_direct.resize(n_thread, Eigen::VectorXd(model.nv).setConstant(0));
    e_scaled_vec.resize(n_thread, Vector6d().setConstant(0));
    grad_target_vec.resize(n_thread, Vector6d().setConstant(0));
    v1.resize(n_thread, Vector6d().setConstant(0));
    v2.resize(n_thread, Vector6d().setConstant(0));
    v3.resize(n_thread, Vector6d().setConstant(0));

    sign_e_scaled_vec.resize(n_thread, Vector6d().setConstant(0));
    grad_e_vec.resize(n_thread, Vector6d().setConstant(0));
    log_indirect_1_vec.resize(n_thread, Vector6d().setConstant(0));
    target_placement_vec.resize(n_thread, pinocchio::SE3::Identity());
    current_placement_vec.resize(total, pinocchio::SE3::Identity());

    dn_dq.resize(n_thread, Matrix3xd::Zero(3, model.nv));
    dw_dq.resize(n_thread, Matrix3xd::Zero(3, model.nv));
    dw2_dq.resize(n_thread, Matrix3xd::Zero(3, model.nv));

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

template <bool compute_first_term, bool compute_second_term>
bool compute_jcoll(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t thread_id, size_t n_coll,
                   size_t idx, size_t coll_a, size_t coll_b, size_t batch_id,
                   size_t time, Eigen::Ref<Eigen::VectorXd> ub,
                   Eigen::Ref<Eigen::VectorXd> lb,
                   Eigen::Ref<Eigen::MatrixXd> G, Eigen::Ref<Eigen::VectorXd> q,
                   bool compute_kine) {
  ZoneScopedN("compute jcoll");

  auto &w1 = workspace.w1[thread_id];
  auto &w2 = workspace.w2[thread_id];
  auto &w_diff = workspace.w_diff[thread_id];
  auto &n = workspace.n[thread_id];
  auto &r1 = workspace.r1[thread_id];
  auto &r2 = workspace.r2[thread_id];
  auto &J_1 = workspace.J_1[thread_id];
  auto &J_2 = workspace.J_2[thread_id];
  auto &J_coll = workspace.J_coll[thread_id];

  coal::CollisionResult &res = workspace.cres[n_coll][idx];
  coal::CollisionRequest &req = workspace.creq[n_coll][idx];
  diffcoal::ContactDerivative &dres = workspace.cdres[n_coll][idx];
  diffcoal::ContactDerivativeRequest &dreq = workspace.cdreq[n_coll][idx];

  if (unlikely(compute_kine)) {
    pinocchio::framesForwardKinematics(model, data, q);
    pinocchio::updateFramePlacement(model, data, workspace.tool_id);
    pinocchio::updateGeometryPlacements(
        model, data, workspace.gmodel[thread_id], workspace.gdata[thread_id]);
    pinocchio::updateGlobalPlacements(model, data);
    dres.clear();
    res.clear();
  }
  if (isBox(coll_a) ||
      isBox(coll_b || isCapsule(coll_a) || isCapsule(coll_b))) {
    dreq.derivative_type = diffcoal::ContactDerivativeType::FiniteDifference;
    dreq.finite_differences_options.eps_fd = 1e-7;
  } else {
    dreq.derivative_type = diffcoal::ContactDerivativeType::FirstOrder;
  }
  dreq.derivative_type = diffcoal::ContactDerivativeType::FiniteDifference;
  dreq.finite_differences_options.eps_fd = 1e-7;

  coal::collide(
      &workspace.get_coal_obj(coll_a, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
      &workspace.get_coal_obj(coll_b, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]), req,
      res);

  if (res.getContact(0).penetration_depth < 0 && !workspace.allow_collisions) {
    workspace.discarded[batch_id] = true;
    if (likely(workspace.echo)) {
      spdlog::critical(fmt::runtime("\n====================[ CRITICAL ERROR "
                                    "COLLISION ]====================\n"
                                    "Time: {}\n"
                                    "Distance: {:+.6f}\n"
                                    "Collision pair detected: ({}, {})\n"
                                    "=========================================="
                                    "==========================\n"),
                       time, res.getContact(0).penetration_depth, coll_a,
                       coll_b);
      return false;
    }

  } else {
    ZoneScopedN("Collisions forward pass");
    diffcoal::computeContactDerivative(
        &workspace.get_coal_obj(coll_a, batch_id),
        pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
        &workspace.get_coal_obj(coll_b, batch_id),
        pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]),
        res.getContact(0), dreq, dres);

    size_t j1_id = workspace.get_geom(coll_a, batch_id).parentJoint;
    size_t j2_id = workspace.get_geom(coll_b, batch_id).parentJoint;

    w1 = res.getContact(0).nearest_points[0];
    w2 = res.getContact(0).nearest_points[1];
    w_diff.noalias() = w1 - w2;
    n = w_diff.normalized();
    r1.noalias() = w1 - data.oMi[j1_id].translation();
    r2.noalias() = w2 - data.oMi[j2_id].translation();

    J_1.setZero();
    J_2.setZero();
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::getJointJacobian(model, data, j1_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_1);
    pinocchio::getJointJacobian(model, data, j2_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_2);

    J_coll.setZero();
    if (likely(j1_id != 0)) {
      if constexpr (compute_first_term) {
        J_coll.noalias() += n.transpose() * J_1.block(0, 0, 3, model.nv);
      }
      if constexpr (compute_second_term) {
        J_coll.noalias() += (pinocchio::skew(r1) * n).transpose() *
                            J_1.block(3, 0, 3, model.nv);
      }
    }
    if (unlikely(j2_id != 0)) {
      if constexpr (compute_first_term) {
        J_coll.noalias() -= n.transpose() * J_2.block(0, 0, 3, model.nv);
      }
      if constexpr (compute_second_term) {
        J_coll.noalias() -= (pinocchio::skew(r2) * n).transpose() *
                            J_2.block(3, 0, 3, model.nv);
      }
    }
    if (likely(!compute_kine)) {
      G.row(n_coll) = -J_coll / workspace.dt;
      ub(n_coll) =
          (workspace.collision_strength) *
          (res.getContact(0).penetration_depth - workspace.safety_margin);
      lb(n_coll) = -1e10;
    } else {
      G.row(0) = -J_coll / workspace.dt;
      ub(0) = (workspace.collision_strength) *
              (res.getContact(0).penetration_depth - workspace.safety_margin);
      lb(0) = -1e10;
    }
  }
  return true;
}

template bool compute_jcoll<true, true>(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    pinocchio::Data &data, size_t thread_id, size_t n_coll, size_t idx,
    size_t coll_a, size_t coll_b, size_t batch_id, size_t time,
    Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
    Eigen::Ref<Eigen::MatrixXd> G, Eigen::Ref<Eigen::VectorXd> q,
    bool compute_kine);

template bool compute_jcoll<true, false>(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    pinocchio::Data &data, size_t thread_id, size_t n_coll, size_t idx,
    size_t coll_a, size_t coll_b, size_t batch_id, size_t time,
    Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
    Eigen::Ref<Eigen::MatrixXd> G, Eigen::Ref<Eigen::VectorXd> q,
    bool compute_kine);

template bool compute_jcoll<false, true>(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    pinocchio::Data &data, size_t thread_id, size_t n_coll, size_t idx,
    size_t coll_a, size_t coll_b, size_t batch_id, size_t time,
    Eigen::Ref<Eigen::VectorXd> ub, Eigen::Ref<Eigen::VectorXd> lb,
    Eigen::Ref<Eigen::MatrixXd> G, Eigen::Ref<Eigen::VectorXd> q,
    bool compute_kine);

bool compute_coll_matrix(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, size_t thread_id,
                         size_t batch_id, size_t tool_id, unsigned int time,
                         size_t idx, Eigen::Ref<Eigen::VectorXd> q,
                         pinocchio::Data &data, Eigen::Ref<Eigen::VectorXd> ub,
                         Eigen::Ref<Eigen::VectorXd> lb,
                         Eigen::Ref<Eigen::MatrixXd> G) {
  ZoneScopedN("compute coll matrix");
  pinocchio::updateFramePlacement(model, data, tool_id);
  pinocchio::updateGeometryPlacements(model, data, workspace.gmodel[thread_id],
                                      workspace.gdata[thread_id]);
  for (size_t n_coll = 0; n_coll < workspace.pairs.size(); ++n_coll) {
    auto [coll_a, coll_b] = workspace.pairs[n_coll];
    bool safe =
        compute_jcoll(workspace, model, data, thread_id, n_coll, idx, coll_a,
                      coll_b, batch_id, time, ub, lb, G, q, false);
    if (!safe) {
      return false;
    }
  }
  return true;
}

void pre_allocate_qp(QP_pass_workspace2 &workspace, unsigned int &time,
                     size_t &idx) {
  ZoneScopedN("pre allocate qp");
  // we set malloc allowed to true as after allocation was done once, future
// run won't allocate.
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
}

void forward_pass_final_computation(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    size_t thread_id, size_t batch_id, size_t seq_len, size_t tool_id,
    const pinocchio::SE3 &T_star, unsigned int time,
    const Eigen::Ref<const Eigen::VectorXd> q_next, pinocchio::Data &data) {
  if (time == seq_len - 1) {
    ZoneScopedN("final computations");
    Vector6d &log_vec = workspace.last_log_vec[thread_id];
    pinocchio::framesForwardKinematics(model, data, q_next);

    workspace.steps_per_batch[batch_id] = time;
    workspace.last_q[batch_id].noalias() = q_next;
    workspace.last_T[batch_id] = data.oMf[tool_id].actInv(T_star);
    workspace.last_logT[batch_id] = pinocchio::log6(workspace.last_T[batch_id]);

    log_vec = workspace.last_logT[batch_id].toVector();
    log_vec.tail(3) *= workspace.rot_w;
    const double loss_L2 = log_vec.squaredNorm();
    const double loss_L1 = log_vec.lpNorm<1>();
    workspace.losses[batch_id] = loss_L2 + workspace.lambda_L1 * loss_L1;
  }
}

void compute_cost(QP_pass_workspace2 &workspace, size_t thread_id, size_t idx,
                  const pinocchio::Model &model, pinocchio::Data &data,
                  const Eigen::Ref<const Eigen::VectorXd> q) {
  ZoneScopedN("compute cost");

  Matrix6xd &jac = workspace.jacobians_[idx];
  Eigen::MatrixXd &Q = workspace.Q_vec_[thread_id];

  Q.noalias() = jac.transpose() * jac;

  if (workspace.equilibrium) {
    Eigen::MatrixXd j2(jac.rows(), jac.cols());
    j2.setZero();

    pinocchio::computeFrameJacobian(
        model, data, q, workspace.equilibrium_tool_id, pinocchio::LOCAL, j2);

    Q += j2.transpose() * j2;
  }

  Q.diagonal().array() += workspace.q_reg;
}

void compute_target(QP_pass_workspace2 &workspace,
                    const pinocchio::Model &model, pinocchio::Data &data,
                    const Eigen::Map<const Eigen::VectorXd> p, size_t thread_id,
                    size_t idx, size_t tool_id,
                    const Eigen::Ref<const Eigen::VectorXd> q) {
  ZoneScopedN("compute target");

  const double lambda = workspace.lambda;

  auto &jac = workspace.jacobians_[idx];
  auto &err = workspace.err_vec[thread_id];
  auto &diff = workspace.diff[idx];
  auto &adj_diff = workspace.adj_diff[idx];
  auto &target = workspace.target_vec[thread_id];
  auto &current_placement = workspace.current_placement_vec[idx];
  auto &target_placement = workspace.target_placement_vec[thread_id];

  current_placement = data.oMf[tool_id];
  const pinocchio::Motion target_lie(p.head<3>(), p.tail<3>());
  target_placement = pinocchio::exp6(target_lie);
  diff = current_placement.actInv(target_placement);

  workspace.target[idx] = target_lie;

  adj_diff = diff.toActionMatrixInverse();
  err = pinocchio::log6(diff).toVector();
  target.noalias() = lambda * jac.transpose() * err;

  if (unlikely(workspace.equilibrium)) {
    auto placement2 = data.oMf[workspace.equilibrium_tool_id];
    const pinocchio::Motion target_lie_2(
        workspace.equilibrium_second_target.row(idx).head<3>(),
        workspace.equilibrium_second_target.row(idx).tail<3>());
    auto diff2 = placement2.actInv(pinocchio::exp6(target_lie_2));
    auto err2 = pinocchio::log6(diff2).toVector();
    Eigen::MatrixXd j2(jac.rows(), jac.cols());
    j2.setZero();

    pinocchio::computeFrameJacobian(
        model, data, q, workspace.equilibrium_tool_id, pinocchio::LOCAL, j2);

    target += lambda * j2.transpose() * err2;
  }
}

void single_forward_pass(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, size_t thread_id,
                         size_t batch_id, size_t seq_len, size_t eq_dim,
                         size_t tool_id, pinocchio::SE3 T_star) {
  ZoneScopedN("single forward pass");
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(false);
#endif

  for (unsigned int time = 0; time < seq_len; time++) {
    ZoneScopedN("single forward pass iter");
    auto &ub = workspace.ub[thread_id];
    auto &lb = workspace.lb[thread_id];
    auto &G = workspace.G[thread_id];
    Eigen::MatrixXd &Q = workspace.Q_vec_[thread_id];
    auto &target = workspace.target_vec[thread_id];

    size_t idx = batch_id * seq_len + time;

    double *log_target_ptr =
        workspace.log_target.data() + batch_id * seq_len * 6 + time * 6;
    const Eigen::Map<const Eigen::VectorXd> log_target(log_target_ptr, 6);

    double *A_ptr = workspace.A_.data() + (idx)*eq_dim * 6;
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        A(A_ptr, static_cast<Eigen::Index>(eq_dim), 6);

    double *b_ptr =
        workspace.b_.data() + batch_id * seq_len * eq_dim + time * eq_dim;
    Eigen::Map<Eigen::VectorXd> b(b_ptr, static_cast<Eigen::Index>(eq_dim));

    double *q_ptr = workspace.positions_.data() +
                    batch_id * (seq_len + 1) * model.nv + time * model.nv;
    Eigen::Map<Eigen::VectorXd> q(q_ptr, model.nv);

    double *q_next_ptr = workspace.positions_.data() +
                         batch_id * (seq_len + 1) * model.nv +
                         (time + 1) * model.nv;
    Eigen::Map<Eigen::VectorXd> q_next(q_next_ptr, model.nv);

    pinocchio::Data &data = workspace.data_vec_[thread_id];
    pinocchio::framesForwardKinematics(model, data, q);
    Matrix6xd &jac = workspace.jacobians_[idx];
    jac.setZero();
    pinocchio::computeFrameJacobian(model, data, q, tool_id, pinocchio::LOCAL,
                                    jac);

    if constexpr (collisions) {
      bool safe = compute_coll_matrix(workspace, model, thread_id, batch_id,
                                      tool_id, time, idx, q, data, ub, lb, G);
      if (!safe) {
        goto END;
      }
    }
    if (workspace.equilibrium) {
      pinocchio::centerOfMass(model, data);
      auto com = data.com[0];
      pinocchio::jacobianCenterOfMass(model, data);
      auto J_com = data.Jcom;
      G = workspace.A_supp * J_com.topRows(2);
      ub = workspace.b_supp - workspace.A_supp * com.topRows(2);
      lb.setConstant(-1e10);
    }

    compute_cost(workspace, thread_id, idx, model, data, q);
    compute_target(workspace, model, data, log_target, thread_id, idx, tool_id,
                   q);

    double *articular_speed_ptr = workspace.articular_speed_.data() +
                                  batch_id * seq_len * model.nv +
                                  time * model.nv;
    Eigen::Map<Eigen::VectorXd> articular_speed(articular_speed_ptr, model.nv);

    pre_allocate_qp(workspace, time, idx);

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

    forward_pass_final_computation(workspace, model, thread_id, batch_id,
                                   seq_len, tool_id, T_star, time, q_next,
                                   data);
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
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &log_target,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, size_t num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
              double dt) {
  ZoneScopedN("single forward");
  if (workspace.tool_id < 0) {
    throw std::runtime_error(
        "You must set workspace's tool id. (workspace.set_tool_id(tool_id))");
  }

  const size_t batch_size = static_cast<size_t>(log_target.dimension(0));
  const size_t seq_len = static_cast<size_t>(log_target.dimension(1));
  const size_t eq_dim = static_cast<size_t>(A.dimension(1));

  workspace.positions_.setZero();
  workspace.allocate(model, batch_size, seq_len, model.nv, eq_dim, num_thread);
  workspace.init_geometry(model, batch_size);
  workspace.dt = dt;
  workspace.b_ = b;
  workspace.A_ = A;
  workspace.log_target = log_target;
  for (size_t i = 0; i < workspace.discarded.size(); ++i) {
    workspace.discarded[i] = false;
  }

  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    double *q_ptr =
        workspace.positions_.data() + batch_id * (seq_len + 1) * model.nv;
    Eigen::Map<Eigen::VectorXd> q(q_ptr, model.nv);
    q = initial_position.row(batch_id);
  }

  omp_set_num_threads(num_thread);
  // #pragma omp parallel for schedule(static, 1)
  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
    single_forward_pass(workspace, model, thread_id, batch_id, seq_len, eq_dim,
                        workspace.tool_id, T_star[batch_id]);
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
                                    QP_pass_workspace2 &workspace, size_t time,
                                    size_t batch_id, size_t seq_len,
                                    size_t n_coll, size_t thread_id) {
  ZoneScopedN("backpropagate through collisions");
  auto &ddist = workspace.ddist[thread_id];
  auto &dJcoll_dq = workspace.dJcoll_dq[thread_id];
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
                   size_t batch_id, size_t time, size_t thread_id,
                   size_t n_coll) {
  ZoneScopedN("compute dn/dq");
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  auto &dn_dq = workspace.dn_dq[thread_id];
  auto [coll_a, coll_b] = workspace.pairs[n_coll];
  J1.setZero();
  J2.setZero();
  dn_dq.setZero();
  if (j1_id != 0) {
    J1.setZero();
    pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
    dn_dq.noalias() +=
        (workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
             .dnormal_dM1 *
         workspace.get_coll_pos(coll_a, batch_id).toActionMatrixInverse() * J1);
  }
  if (j2_id != 0) {
    J2.setZero();
    pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
    dn_dq.noalias() +=
        workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
            .dnormal_dM2 *
        workspace.get_coll_pos(coll_b, batch_id).toActionMatrixInverse() * J2;
  }
}

void compute_dr1_dq(const pinocchio::Model &model, pinocchio::Data &data,
                    size_t j1_id, size_t j2_id,
                    diffcoal::ContactDerivative &cdres, size_t coll,
                    size_t coll_2, size_t batch_id,
                    QP_pass_workspace2 &workspace, size_t thread_id) {
  ZoneScopedN("compute dr1_dq");
  Eigen::Matrix<double, 3, Eigen::Dynamic> &dr1_dq =
      workspace.dr1_dq[thread_id];
  auto &r1 = workspace.r1[thread_id];
  auto &w1 = workspace.w1[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp1 = workspace.tmp1_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp2 = workspace.tmp2_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp3 =
      workspace.tmp3_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp4 =
      workspace.tmp4_dr1_dq[thread_id];
  Eigen::Matrix3d &R = workspace.R[thread_id];
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  J1.setZero();
  J2.setZero();
  pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
  pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
  r1.noalias() = w1 - data.oMi[j1_id].translation();
  R = data.oMi[j1_id].rotation();
  tmp1.noalias() = cdres.dcpos_dM1;
  tmp1.noalias() -= cdres.dvsep_dM1 * 0.5;
  tmp2.noalias() =
      tmp1 * workspace.get_coll_pos(coll, batch_id).toActionMatrixInverse();
  tmp3.noalias() = tmp2 * J1;

  tmp4.noalias() = R * J1.block(0, 0, 3, model.nv);
  dr1_dq.noalias() = tmp3 - tmp4;
  if (unlikely(j2_id != 0)) {
    tmp1.noalias() = cdres.dcpos_dM2;
    tmp1.noalias() -= cdres.dvsep_dM2 * 0.5;
    tmp2.noalias() =
        tmp1 * workspace.get_coll_pos(coll_2, batch_id).toActionMatrixInverse();
    tmp3.noalias() = tmp2 * J2;
    dr1_dq.noalias() += tmp3;
  }
}

void compute_dr2_dq(const pinocchio::Model &model, pinocchio::Data &data,
                    size_t j1_id, size_t j2_id,
                    diffcoal::ContactDerivative &cdres, size_t coll,
                    size_t coll_2, size_t batch_id,
                    QP_pass_workspace2 &workspace, size_t thread_id) {
  ZoneScopedN("compute dr2_dq");
  Eigen::Matrix<double, 3, Eigen::Dynamic> &dr2_dq =
      workspace.dr2_dq[thread_id];
  auto &r2 = workspace.r2[thread_id];
  auto &w2 = workspace.w2[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp1 = workspace.tmp1_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, 6> &tmp2 = workspace.tmp2_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp3 =
      workspace.tmp3_dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &tmp4 =
      workspace.tmp4_dr1_dq[thread_id];
  Eigen::Matrix3d &R = workspace.R[thread_id];
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  J1.setZero();
  J2.setZero();
  pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
  pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
  r2.noalias() = w2 - data.oMi[j2_id].translation();
  R = data.oMi[j2_id].rotation();
  tmp1.noalias() = cdres.dcpos_dM2;
  tmp1.noalias() += cdres.dvsep_dM2 * 0.5;
  tmp2.noalias() =
      tmp1 * workspace.get_coll_pos(coll, batch_id).toActionMatrixInverse();
  tmp3.noalias() = tmp2 * J2;
  tmp4.noalias() = R * J2.block(0, 0, 3, model.nv);
  dr2_dq.noalias() = tmp3 - tmp4;
  if (likely(j1_id != 0)) {
    tmp1.noalias() = cdres.dcpos_dM1;
    tmp1.noalias() += cdres.dvsep_dM1 * 0.5;
    tmp2.noalias() =
        tmp1 * workspace.get_coll_pos(coll_2, batch_id).toActionMatrixInverse();
    tmp3.noalias() = tmp2 * J1;
    dr2_dq.noalias() += tmp3;
  }
}

void dJ_coll_first_term(QP_pass_workspace2 &workspace,
                        const pinocchio::Model &model, pinocchio::Data &data,
                        coal::CollisionResult &cres, size_t thread_id,
                        size_t j1_id, size_t j2_id) {
  ZoneScopedN("compute djcoll first term");
  auto &w1 = workspace.w1[thread_id];
  auto &w2 = workspace.w2[thread_id];
  auto &w_diff = workspace.w_diff[thread_id];
  auto &J_diff = workspace.J_diff[thread_id];
  auto &dn_dq = workspace.dn_dq[thread_id];
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  const Eigen::Vector3d &n = workspace.n[thread_id];

  Eigen::Tensor<double, 3> &H1 =
      workspace.Hessian[workspace.num_thread_ + thread_id];
  Eigen::Tensor<double, 3> &H2 = workspace.Hessian[thread_id];

  auto &term_1_A = workspace.term_1_A[thread_id];
  auto &term_1_B = workspace.term_1_B[thread_id];
  term_1_A.setZero();
  w1 = cres.getContact(0).nearest_points[0];
  w2 = cres.getContact(0).nearest_points[1];
  w_diff = w1 - w2;
  workspace.n[thread_id] = w_diff.normalized();
  pinocchio::getJointKinematicHessian(model, data, j1_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H1);
  pinocchio::getJointKinematicHessian(model, data, j2_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H2);
  int j_dim = H1.dimension(1);
  int q_dim = H1.dimension(2);
  if (likely(j1_id != 0 && j2_id == 0)) {
    for (int qqq = 0; qqq < q_dim; ++qqq) {
      for (int j = 0; j < j_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < 3; ++i)
          s += n(i) * H1(i, j, qqq);
        term_1_A(j, qqq) = s;
      }
    }

  } else if (likely(j1_id != 0 && j2_id != 0)) {
    for (int qqq = 0; qqq < q_dim; ++qqq) {
      for (int j = 0; j < j_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < 3; ++i)
          s += n(i) * (H1(i, j, qqq) - H2(i, j, qqq));
        term_1_A(j, qqq) = s;
      }
    }

  } else if (unlikely(j2_id != 0)) {
    for (int qqq = 0; qqq < q_dim; ++qqq) {
      for (int j = 0; j < j_dim; ++j) {
        double s = 0.0;
        for (int i = 0; i < 3; ++i)
          s -= n(i) * H2(i, j, qqq);
        term_1_A(j, qqq) = s;
      }
    }

  } else {
    term_1_A.setZero();
  }

  J1.setZero();
  J2.setZero();
  pinocchio::getJointJacobian(model, data, j1_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J1);
  pinocchio::getJointJacobian(model, data, j2_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J2);
  J_diff.setZero();
  if (likely(j1_id != 0)) {
    J_diff.noalias() += J1.topRows(3);
  }
  if (unlikely(j2_id != 0)) {
    J_diff.noalias() -= J2.topRows(3);
  }
  term_1_B.noalias() = -J_diff.transpose() * dn_dq;
}

void dJ_coll_second_term(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, pinocchio::Data &data,
                         size_t j1_id, size_t j2_id, size_t batch_id,
                         size_t thread_id, size_t coll_a, size_t coll_b,
                         diffcoal::ContactDerivative &cdres) {
  ZoneScopedN("compute djcoll second term");
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];

  auto &r1 = workspace.r1[thread_id];
  auto &r2 = workspace.r2[thread_id];

  const Eigen::Vector3d &n = workspace.n[thread_id];
  auto &dn_dq = workspace.dn_dq[thread_id];

  Eigen::Vector3d &c = workspace.c[thread_id];

  Eigen::MatrixXd &term_2_A = workspace.term_2_A[thread_id];
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &term_2_B =
      workspace.term_2_B[thread_id];

  Eigen::Matrix<double, 3, Eigen::Dynamic> &dr1_dq =
      workspace.dr1_dq[thread_id];
  Eigen::Matrix<double, 3, Eigen::Dynamic> &dr2_dq =
      workspace.dr2_dq[thread_id];

  Eigen::Tensor<double, 3> &H1 =
      workspace.Hessian[workspace.num_thread_ + thread_id];
  Eigen::Tensor<double, 3> &H2 = workspace.Hessian[thread_id];
  pinocchio::getJointKinematicHessian(model, data, j1_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H1);
  pinocchio::getJointKinematicHessian(model, data, j2_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H2);

  compute_dr1_dq(model, data, j1_id, j2_id, cdres, coll_a, coll_b, batch_id,
                 workspace, thread_id);
  compute_dr2_dq(model, data, j1_id, j2_id, cdres, coll_b, coll_a, batch_id,
                 workspace, thread_id);
  J1.setZero();
  J2.setZero();
  pinocchio::getJointJacobian(model, data, j1_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J1);
  pinocchio::getJointJacobian(model, data, j2_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, J2);

  term_2_A.setZero();
  term_2_B.setZero();

  if (likely(j1_id != 0)) {
    c = r1.cross(n);
    term_2_A = J1.block(3, 0, 3, model.nv).transpose() *
               (-pinocchio::skew(r1) * dn_dq - pinocchio::skew(n) * dr1_dq);

    for (int l = 0; l < 3; ++l)
      for (int j = 0; j < term_2_B.cols(); ++j)
        for (int i = 0; i < term_2_B.rows(); ++i)
          term_2_B(i, j) += c(l) * H1(3 + l, i, j);
  }

  if (unlikely(j2_id != 0)) {
    c = r2.cross(n);
    term_2_A -= J2.block(3, 0, 3, model.nv).transpose() *
                (-pinocchio::skew(r2) * dn_dq - pinocchio::skew(n) * dr2_dq);
    for (int l = 0; l < 3; ++l)
      for (int j = 0; j < term_2_B.cols(); ++j)
        for (int i = 0; i < term_2_B.rows(); ++i)
          term_2_B(i, j) -= c(l) * H2(3 + l, i, j);
  }
}

void compute_ddist(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t j1_id, size_t j2_id,
                   size_t batch_id, size_t time, size_t n_coll, size_t coll_a,
                   size_t coll_b, size_t thread_id) {
  ZoneScopedN("compute ddist");
  auto &J1 = workspace.J1[thread_id];
  auto &J2 = workspace.J2[thread_id];
  auto &ddist = workspace.ddist[thread_id];
  Eigen::RowVector<double, 6> &temp_ddist = workspace.temp_ddist[thread_id];

  J1.setZero();
  J2.setZero();
  pinocchio::getJointJacobian(model, data, j1_id, pinocchio::LOCAL, J1);
  pinocchio::getJointJacobian(model, data, j2_id, pinocchio::LOCAL, J2);
  ddist.setZero();
  if (j1_id != 0) {
    temp_ddist.noalias() =
        workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
            .ddist_dM1.transpose() *
        workspace.get_coll_pos(coll_a, batch_id).toActionMatrixInverse();
    ddist.noalias() += temp_ddist * J1;
  }
  if (j2_id != 0) {
    temp_ddist.noalias() =
        workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time]
            .ddist_dM2.transpose() *
        workspace.get_coll_pos(coll_b, batch_id).toActionMatrixInverse();
    ddist.noalias() += temp_ddist * J2;
  }
}

void compute_d_dist_and_d_Jcoll(QP_pass_workspace2 &workspace,
                                const pinocchio::Model &model,
                                pinocchio::Data &data, size_t j1_id,
                                size_t j2_id, size_t batch_id, size_t time,
                                size_t thread_id, Eigen::Ref<Eigen::VectorXd> q,
                                size_t n_coll) {
  ZoneScopedN("compute ddist/dq & dJcoll/dq");

  auto [coll_a, coll_b] = workspace.pairs[n_coll];
  coal::CollisionResult &cres =
      workspace.cres[n_coll][batch_id * workspace.seq_len_ + time];
  diffcoal::ContactDerivative &cdres =
      workspace.cdres[n_coll][batch_id * workspace.seq_len_ + time];

  auto &dJcoll_dq = workspace.dJcoll_dq[thread_id];
  dJcoll_dq.setZero();

  auto &term_1_A = workspace.term_1_A[thread_id];
  auto &term_1_B = workspace.term_1_B[thread_id];

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &term_2_B =
      workspace.term_2_B[thread_id];
  Eigen::Ref<Eigen::MatrixXd> term_2_A = workspace.term_2_A[thread_id];

  Eigen::Tensor<double, 3> &H1 =
      workspace.Hessian[workspace.num_thread_ + thread_id];
  Eigen::Tensor<double, 3> &H2 = workspace.Hessian[thread_id];

  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::updateFramePlacement(model, data, workspace.tool_id);
  pinocchio::updateGeometryPlacements(model, data, workspace.gmodel[thread_id],
                                      workspace.gdata[thread_id]);
  pinocchio::computeForwardKinematicsDerivatives(model, data, q, q, q);
  pinocchio::computeJointKinematicHessians(model, data);

  compute_dn_dq(workspace, model, data, j1_id, j2_id, batch_id, time, thread_id,
                n_coll);

  compute_ddist(workspace, model, data, j1_id, j2_id, batch_id, time, n_coll,
                coll_a, coll_b, thread_id);

  H1.setZero();
  H2.setZero();
  pinocchio::getJointKinematicHessian(model, data, j1_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H1);
  pinocchio::getJointKinematicHessian(model, data, j2_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, H2);
  dJ_coll_first_term(workspace, model, data, cres, thread_id, j1_id, j2_id);

  dJ_coll_second_term(workspace, model, data, j1_id, j2_id, batch_id, thread_id,
                      coll_a, coll_b, cdres);

  dJcoll_dq = term_1_A;
  dJcoll_dq += term_1_B;
  dJcoll_dq += term_2_A;
  dJcoll_dq += term_2_B;
}

void computePoseLossGradient(QP_pass_workspace2 &workspace, size_t thread_id,
                             const pinocchio::Model &model,
                             pinocchio::Data &data,
                             const pinocchio::SE3 &T_current,
                             const Eigen::VectorXd &q,
                             const pinocchio::Motion &log_T, double rot_w,
                             const std::size_t tool_id) {
  ZoneScopedN("compute pose loss gradient");
  auto &Adj = workspace.Adj_vec[thread_id];
  auto &Jlog = workspace.Jlog_vec[thread_id];
  auto &J_frame = workspace.J_frame_vec[thread_id];
  auto &e = workspace.e_vec[thread_id];

  pinocchio::framesForwardKinematics(model, data, q);

  e = log_T.toVector();
  Vector6d w;
  w << 1, 1, 1, rot_w, rot_w, rot_w;
  Vector6d e_scaled = w.array() * e.array();

  Vector6d grad_e = 2.0 * e_scaled.array() * w.array();
  Vector6d sign_e_scaled = e_scaled.unaryExpr(
      [](double x) { return static_cast<double>((x > 0) - (x < 0)); });
  grad_e += (workspace.lambda_L1 * sign_e_scaled.array() * w.array()).matrix();

  Adj.setZero();
  Jlog.setZero();
  J_frame.setZero();
  Adj = T_current.toActionMatrixInverse();
  Jlog = pinocchio::Jlog6(T_current);
  pinocchio::computeFrameJacobian(model, data, q, tool_id, pinocchio::LOCAL,
                                  J_frame);

  workspace.dloss_dq_tmp1[thread_id].noalias() =
      (-Adj.transpose()) * Jlog.transpose();
  workspace.dloss_dq_tmp2[thread_id].noalias() =
      J_frame.transpose() * workspace.dloss_dq_tmp1[thread_id];
  workspace.dloss_dq_tmp3[thread_id].noalias() =
      workspace.dloss_dq_tmp2[thread_id] * grad_e;
}

void single_backward_pass(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    size_t thread_id, size_t batch_id, size_t seq_len, size_t tool_id,
    double dt, Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output) {
  if (workspace.discarded[batch_id]) {
    // element is discarded so gradient stays 0.
  } else {
    ZoneScopedN("single backward pass");

#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
    pinocchio::Data &data = workspace.data_vec_[thread_id];
    auto &dloss_dq = workspace.dloss_dq[batch_id];
    auto &dloss_dq_diff = workspace.dloss_dq_diff[batch_id];
    const auto &goals = workspace.intermediate_goals[batch_id];
    const auto &geom_goals = workspace.intermediate_geom_goals[batch_id];
    Eigen::Index time_goal_idx = static_cast<Eigen::Index>(goals.size()) - 1;
    Eigen::Index time_geom_goal_idx =
        static_cast<Eigen::Index>(geom_goals.size()) - 1;

    computePoseLossGradient(
        workspace, thread_id, model, data, workspace.last_T[batch_id],
        workspace.last_q[batch_id], workspace.last_logT[batch_id],
        workspace.rot_w, tool_id);
    dloss_dq.noalias() =
        (workspace.end_loss_w * dt) * workspace.dloss_dq_tmp3[thread_id];
    dloss_dq_diff.setZero();

    for (Eigen::Index time =
             static_cast<Eigen::Index>(workspace.steps_per_batch[batch_id]);
         time >= 0; time--) {
      size_t idx = batch_id * seq_len + time;
      double lambda = workspace.lambda;
      size_t grad_dim = static_cast<size_t>(grad_output.dimension(2));
      const size_t nv = static_cast<size_t>(model.nv);

      auto &grad_target = workspace.grad_target_vec[thread_id];
      auto &J = workspace.jacobians_[idx];
      auto &padded = workspace.padded[thread_id];
      padded.setZero();
      auto &KKT_grad = workspace.workspace_.grad_KKT_mem_[idx];
      auto &rhs_grad = workspace.workspace_.grad_rhs_mem_[idx];
      auto &grad_log_target = workspace.grad_log_target_[idx];

      double *q_ptr = workspace.positions_.data() +
                      batch_id * (seq_len + 1) * model.nv + (time)*model.nv;
      const Eigen::Map<Eigen::VectorXd> q(q_ptr, model.nv);

      Eigen::Map<Eigen::VectorXd> grad_vec(
          grad_output.data() + batch_id * seq_len * grad_dim + time * grad_dim,
          static_cast<Eigen::Index>(grad_dim));

      if (time_goal_idx >= 0 &&
          goals[time_goal_idx].second == static_cast<size_t>(time)) {
        auto T = workspace.current_placement_vec[idx].actInv(
            goals[time_goal_idx].first);
        computePoseLossGradient(workspace, thread_id, model, data, T, q,
                                pinocchio::log6(T), workspace.i_rot_w, tool_id);
        dloss_dq.noalias() += (workspace.intermediate_loss_w * dt) *
                              workspace.dloss_dq_tmp3[thread_id];
        --time_goal_idx;
      }

      if (time_geom_goal_idx >= 0 &&
          std::get<2>(geom_goals[time_geom_goal_idx]) ==
              static_cast<size_t>(time)) {
        auto [coll_a, coll_b, n_coll] = workspace.get_coll_data(
            std::get<0>(geom_goals[time_geom_goal_idx]),
            std::get<1>(geom_goals[time_geom_goal_idx]));
        auto &ub = workspace.ub[thread_id];
        auto &lb = workspace.lb[thread_id];
        auto &G = workspace.G[thread_id];
        coal::CollisionResult &res = workspace.cres[n_coll][idx];
        compute_jcoll(workspace, model, data, thread_id, n_coll, idx, coll_a,
                      coll_b, batch_id, time, ub, lb, G, q, true);
        double x =
            res.getContact(0).penetration_depth - workspace.safety_margin;
        double grad_factor;

        if (std::abs(x) < 1.0)
          grad_factor = x * x;
        else
          grad_factor = 2 * x;

        dloss_dq.noalias() -= grad_factor *
                              (workspace.intermediate_geom_loss_w * dt) *
                              G.row(n_coll);
        --time_geom_goal_idx;
      }

      padded.head(nv) = dloss_dq + dloss_dq_diff;
      QP_backward(workspace.workspace_, padded, idx);

      workspace.grad_err_[idx] =
          -workspace.jacobians_[idx] * rhs_grad.head(model.nv) * lambda;
      pinocchio::SE3 &diff = workspace.diff[idx];
      grad_target =
          pinocchio::Jlog6(diff).transpose() * workspace.grad_err_[idx];
      grad_log_target.noalias() =
          pinocchio::Jexp6(workspace.target[idx]).transpose() * grad_target;

      workspace.grad_J_[thread_id].noalias() =
          2 * J * KKT_grad.block(0, 0, model.nv, model.nv);

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
              thread_id, q, n_coll);
          backpropagateThroughCollisions(dloss_dq_diff, workspace, time,
                                         batch_id, seq_len, n_coll, thread_id);
        }
      }
    }
  }
#ifdef EIGEN_RUNTIME_NO_MALLOC
  Eigen::internal::set_is_malloc_allowed(true);
#endif
}

void backward_pass2(QP_pass_workspace2 &workspace,
                    const pinocchio::Model &model,
                    Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output,
                    size_t num_thread, size_t batch_size) {
  ZoneScopedN("backward pass");
  size_t seq_len = static_cast<size_t>(workspace.b_.dimension(1));
  size_t tool_id = static_cast<size_t>(workspace.tool_id);
  double dt = workspace.dt;

  for (auto vec = workspace.grad_log_target_.begin();
       vec != workspace.grad_log_target_.end(); ++vec) {
    vec->setZero();
  }

  omp_set_num_threads(static_cast<int>(num_thread));
  // #pragma omp parallel for schedule(static, 1)
  for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
    size_t thread_id = static_cast<size_t>(omp_get_thread_num());
    single_backward_pass(workspace, model, thread_id, batch_id, seq_len,
                         tool_id, dt, grad_output);
  }
}
