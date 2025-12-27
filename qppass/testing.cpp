#include "dik_cols.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/model.hpp"
#include <cmath>
#include <csignal>
#include <pinocchio/parsers/urdf.hpp>

double constexpr eps = 1e-5;
constexpr double tol_abs = 1e-4;
constexpr double tol_rel = 1e-4;

Eigen::VectorXd compute_r1(QP_pass_workspace2 &workspace,
                           const pinocchio::Model &model, pinocchio::Data &data,
                           size_t thread_id, size_t n_coll, size_t idx,
                           size_t coll_a, size_t coll_b, size_t batch_id,
                           Eigen::Ref<Eigen::VectorXd> q) {
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::updateFramePlacement(model, data, workspace.tool_id);
  pinocchio::updateGeometryPlacements(model, data, workspace.gmodel[thread_id],
                                      workspace.gdata[thread_id]);
  pinocchio::updateGlobalPlacements(model, data);

  auto &w1 = workspace.w1[thread_id];
  auto &r1 = workspace.r1[thread_id];
  coal::CollisionResult &res = workspace.cres[n_coll][idx];
  coal::CollisionRequest &req = workspace.creq[n_coll][idx];
  res.clear();

  req.epa_max_iterations = 1000;
  req.gjk_convergence_criterion_type =
      coal::GJKConvergenceCriterionType::Absolute;
  req.gjk_tolerance = 1e-10;
  req.epa_tolerance = 1e-10;
  req.epa_max_iterations = 1000;
  res.clear();
  pinocchio::updateGlobalPlacements(model, data);

  coal::collide(
      &workspace.get_coal_obj(coll_a, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
      &workspace.get_coal_obj(coll_b, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]), req,
      res);

  size_t j1_id = workspace.get_geom(coll_a, batch_id).parentJoint;
  w1 = res.getContact(0).nearest_points[0];
  r1.noalias() = w1 - data.oMi[j1_id].translation();
  return r1;
}

Eigen::VectorXd compute_c(QP_pass_workspace2 &workspace,
                          const pinocchio::Model &model, pinocchio::Data &data,
                          size_t thread_id, size_t n_coll, size_t idx,
                          size_t coll_a, size_t coll_b, size_t batch_id,
                          Eigen::Ref<Eigen::VectorXd> q) {
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::updateFramePlacement(model, data, workspace.tool_id);
  pinocchio::updateGeometryPlacements(model, data, workspace.gmodel[thread_id],
                                      workspace.gdata[thread_id]);
  pinocchio::updateGlobalPlacements(model, data);

  auto &w1 = workspace.w1[thread_id];
  auto &w2 = workspace.w2[thread_id];
  auto &r1 = workspace.r1[thread_id];
  coal::CollisionResult &res = workspace.cres[n_coll][idx];
  coal::CollisionRequest &req = workspace.creq[n_coll][idx];
  res.clear();

  req.epa_max_iterations = 1000;
  req.gjk_convergence_criterion_type =
      coal::GJKConvergenceCriterionType::Absolute;
  req.gjk_tolerance = 1e-10;
  req.epa_tolerance = 1e-10;
  req.epa_max_iterations = 1000;
  res.clear();
  pinocchio::updateGlobalPlacements(model, data);

  coal::collide(
      &workspace.get_coal_obj(coll_a, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
      &workspace.get_coal_obj(coll_b, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]), req,
      res);

  size_t j1_id = workspace.get_geom(coll_a, batch_id).parentJoint;
  w1 = res.getContact(0).nearest_points[0];
  w2 = res.getContact(0).nearest_points[1];
  r1.noalias() = w1 - data.oMi[j1_id].translation();
  Eigen::Vector<double, 3> n = w2 - w1;
  n = n.normalized();
  return r1.cross(n);
}

Eigen::VectorXd compute_n(QP_pass_workspace2 &workspace,
                          const pinocchio::Model &model, pinocchio::Data &data,
                          size_t thread_id, size_t n_coll, size_t idx,
                          size_t coll_a, size_t coll_b, size_t batch_id,
                          Eigen::Ref<Eigen::VectorXd> q) {
  pinocchio::framesForwardKinematics(model, data, q);
  pinocchio::updateFramePlacement(model, data, workspace.tool_id);
  pinocchio::updateGeometryPlacements(model, data, workspace.gmodel[thread_id],
                                      workspace.gdata[thread_id]);
  pinocchio::updateGlobalPlacements(model, data);

  auto &w1 = workspace.w1[thread_id];
  auto &w2 = workspace.w2[thread_id];
  coal::CollisionResult &res = workspace.cres[n_coll][idx];
  coal::CollisionRequest &req = workspace.creq[n_coll][idx];
  res.clear();

  req.epa_max_iterations = 1000;
  req.gjk_convergence_criterion_type =
      coal::GJKConvergenceCriterionType::Absolute;
  req.gjk_tolerance = 1e-10;
  req.epa_tolerance = 1e-10;
  req.epa_max_iterations = 1000;
  res.clear();
  pinocchio::updateGlobalPlacements(model, data);

  coal::collide(
      &workspace.get_coal_obj(coll_a, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_a]),
      &workspace.get_coal_obj(coll_b, batch_id),
      pinocchio::toFclTransform3f(workspace.gdata[thread_id].oMg[coll_b]), req,
      res);

  size_t j1_id = workspace.get_geom(coll_a, batch_id).parentJoint;
  w1 = res.getContact(0).nearest_points[0];
  w2 = res.getContact(0).nearest_points[1];
  return (w2 - w1).normalized();
}

Eigen::Matrix3d rotation_x(double theta) {
  double c = std::cos(theta);
  double s = std::sin(theta);
  Eigen::Matrix3d R;
  R << 1, 0, 0, 0, c, -s, 0, s, c;
  return R;
}

Eigen::Matrix3d rotation_z(double theta) {
  double c = std::cos(theta);
  double s = std::sin(theta);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}

Eigen::Matrix3d rotation_y(double theta) {
  double c = std::cos(theta);
  double s = std::sin(theta);
  Eigen::Matrix3d R;
  R << c, 0, s, 0, 1, 0, -s, 0, c;
  return R;
}

template <bool compute_first_term = true, bool compute_second_term = true>
Eigen::MatrixXd
fd_dJcoll_dq_(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
              pinocchio::Data &data, size_t n_coll, size_t thread_id,
              size_t idx, size_t coll_a, size_t coll_b, size_t batch_id,
              size_t time, Eigen::Ref<Eigen::VectorXd> q) {
  Eigen::VectorXd q_copy = q;
  Eigen::MatrixXd G(1, model.nv);
  Eigen::VectorXd lb(1);
  Eigen::VectorXd ub(1);
  Eigen::MatrixXd dJcoll_dq(model.nv, model.nv);
  dJcoll_dq.setZero();
  Eigen::MatrixXd Jcoll_plus(1, model.nv);
  Eigen::MatrixXd Jcoll_minus(1, model.nv);
  for (int i = 0; i < model.nv; ++i) {
    Eigen::VectorXd q_plus = q_copy;
    Eigen::VectorXd q_minus = q_copy;
    q_plus(i) += eps;
    q_minus(i) -= eps;
    compute_jcoll<compute_first_term, compute_second_term>(
        workspace, model, data, thread_id, n_coll, idx, coll_a, coll_b,
        batch_id, time, ub, lb, Jcoll_plus, q_plus, true);
    compute_jcoll<compute_first_term, compute_second_term>(
        workspace, model, data, thread_id, n_coll, idx, coll_a, coll_b,
        batch_id, time, ub, lb, Jcoll_minus, q_minus, true);
    dJcoll_dq.col(i) = ((Jcoll_plus - Jcoll_minus) / (2 * eps)).transpose();
  }
  return -dJcoll_dq * workspace.dt;
}

Eigen::MatrixXd fd_dr1_dq(QP_pass_workspace2 &workspace,
                          const pinocchio::Model &model, pinocchio::Data &data,
                          size_t thread_id, size_t n_coll, size_t idx,
                          size_t coll_a, size_t coll_b, size_t batch_id,
                          Eigen::Ref<Eigen::VectorXd> q) {
  const int nv = model.nv;
  Eigen::MatrixXd dr1_dq(3, nv);
  dr1_dq.setZero();
  for (int i = 0; i < nv; ++i) {
    Eigen::VectorXd q_plus = q;
    Eigen::VectorXd q_minus = q;

    q_plus(i) += eps;
    q_minus(i) -= eps;
    Eigen::Vector3d r1_plus =
        compute_r1(workspace, model, data, thread_id, n_coll, idx, coll_a,
                   coll_b, batch_id, q_plus);
    Eigen::Vector3d r1_minus =
        compute_r1(workspace, model, data, thread_id, n_coll, idx, coll_a,
                   coll_b, batch_id, q_minus);
    dr1_dq.col(i) = (r1_plus - r1_minus) / (2.0 * eps);
  }

  return dr1_dq;
}

Eigen::MatrixXd fd_dn_dq(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, pinocchio::Data &data,
                         size_t thread_id, size_t n_coll, size_t idx,
                         size_t coll_a, size_t coll_b, size_t batch_id,
                         Eigen::Ref<Eigen::VectorXd> q) {
  const int nv = model.nv;
  Eigen::MatrixXd dr1_dq(3, nv);
  dr1_dq.setZero();
  for (int i = 0; i < nv; ++i) {
    Eigen::VectorXd q_plus = q;
    Eigen::VectorXd q_minus = q;

    q_plus(i) += eps;
    q_minus(i) -= eps;
    Eigen::Vector3d r1_plus =
        compute_n(workspace, model, data, thread_id, n_coll, idx, coll_a,
                  coll_b, batch_id, q_plus);
    Eigen::Vector3d r1_minus =
        compute_n(workspace, model, data, thread_id, n_coll, idx, coll_a,
                  coll_b, batch_id, q_minus);
    dr1_dq.col(i) = (r1_plus - r1_minus) / (2.0 * eps);
  }

  return dr1_dq;
}

Eigen::MatrixXd fd_dc_dq(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, pinocchio::Data &data,
                         size_t thread_id, size_t n_coll, size_t idx,
                         size_t coll_a, size_t coll_b, size_t batch_id,
                         Eigen::Ref<Eigen::VectorXd> q) {
  const int nv = model.nv;
  Eigen::MatrixXd dc_dq(3, nv);
  dc_dq.setZero();

  for (int i = 0; i < nv; ++i) {
    Eigen::VectorXd q_plus = q;
    Eigen::VectorXd q_minus = q;
    q_plus(i) += eps;
    q_minus(i) -= eps;

    Eigen::Vector3d c_plus =
        compute_c(workspace, model, data, thread_id, n_coll, idx, coll_a,
                  coll_b, batch_id, q_plus);
    Eigen::Vector3d c_minus =
        compute_c(workspace, model, data, thread_id, n_coll, idx, coll_a,
                  coll_b, batch_id, q_minus);

    dc_dq.col(i) = (c_plus - c_minus) / (2.0 * eps);
  }

  return dc_dq;
}

bool TEST(pinocchio::Model &rmodel) {
  size_t batch_size = 1;
  double q_req = 1e-2;
  double dt = 0.005;
  size_t eq_dim = 0;
  size_t n_thread = 1;
  size_t seq_len = 5;
  double bound = -1000;

  QP_pass_workspace2 workspace;
  workspace.set_echo(true);
  workspace.set_allow_collisions(true);
  workspace.pre_allocate(batch_size);
  workspace.set_q_reg(q_req);
  workspace.set_bound(bound);
  workspace.set_lambda(-2);
  workspace.set_collisions_safety_margin(0.02);
  workspace.set_collisions_strength(50);
  workspace.set_L1_weight(0);
  workspace.set_rot_weight(1e-4);

  workspace.add_pair(0, 5);
  workspace.add_pair(5, 0);
  workspace.add_pair(2, 5);
  workspace.add_pair(5, 2);
  workspace.add_pair(0, 2);
  // workspace.add_pair(1, 5);
  // workspace.add_pair(0, 7);
  // workspace.add_pair(0, 8);
  // workspace.add_pair(0, 9);
  // workspace.add_pair(0, 10);
  // workspace.add_pair(0, 11);

  // workspace.add_pair(1, 5);
  // workspace.add_pair(1, 8);
  // workspace.add_pair(1, 9);
  // workspace.add_pair(1, 10);
  // workspace.add_pair(1, 11);

  // workspace.add_pair(6, 8);
  // workspace.add_pair(6, 9);
  // workspace.add_pair(6, 10);
  // workspace.add_pair(6, 11);
  // workspace.add_pair(2, 5);
  // workspace.add_pair(2, 9);
  // workspace.add_pair(3, 9);
  // workspace.add_pair(4, 9);
  // workspace.add_pair(2, 5);
  // workspace.add_pair(3, 5);
  // workspace.add_pair(4, 5);

  // workspace.add_pair(12, 8);
  // workspace.add_pair(12, 9);
  // workspace.add_pair(12, 10);
  // workspace.add_pair(12, 11);

  // workspace.add_pair(13, 8);
  // workspace.add_pair(13, 9);
  // workspace.add_pair(13, 10);
  // workspace.add_pair(13, 11);
  double deg2rad = M_PI / 180.0;
  double ball_radius = 0.1;
  Eigen::Vector3d b1(0.35, 0.55, 0.04);
  Eigen::Vector3d b2(0.35, 0.35, 0.04);
  Eigen::Vector3d b3(0.35, 0.35, 0.04);
  Eigen::Vector3d b4(0.35, 0.60, 0.04);

  Eigen::Matrix3d Ry = rotation_y(90.0 * deg2rad);

  size_t tool_id = 257;
  workspace.set_tool_id(tool_id);
  pinocchio::Data data = pinocchio::Data(rmodel);

  Eigen::Vector3d eff_pos(.1, .18, -.1);
  Eigen::Matrix3d eff_rot = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(0, 0, eff_pos, eff_rot);

  Eigen::Vector3d arm_pos(-0.2, 0.0, 0.02);
  Eigen::Matrix3d arm_rot = Ry;
  workspace.set_coll_pos(1, 0, arm_pos, arm_rot);

  Eigen::Vector3d arm_pos1(-0.4, 0.0, 0.02);
  Eigen::Matrix3d arm_rot1 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(2, 0, arm_pos1, arm_rot1);

  Eigen::Vector3d arm_pos2(-0.2, 0.0, 0.02);
  Eigen::Matrix3d arm_rot2 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(3, 0, arm_pos2, arm_rot2);

  Eigen::Vector3d arm_pos3(-0.0, 0.0, 0.02);
  Eigen::Matrix3d arm_rot3 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(4, 0, arm_pos3, arm_rot3);

  Eigen::Vector3d arm_pos4(-0.4, 0.0, 0.15);
  Eigen::Matrix3d arm_rot4 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(12, 0, arm_pos4, arm_rot4);

  Eigen::Vector3d arm_pos5(-0.3, 0.0, 0.15);
  Eigen::Matrix3d arm_rot5 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(13, 0, arm_pos5, arm_rot5);

  Eigen::Vector3d plane_pos(0.0, 0.0, -5.0);
  Eigen::Matrix3d plane_rot = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(5, 0, plane_pos, plane_rot);

  Eigen::Vector3d caps_pos(-0.25, 0.0, +0.15);
  Eigen::Matrix3d caps_rot = Ry;
  workspace.set_coll_pos(6, 0, caps_pos, caps_rot);

  Eigen::Vector3d ball_pos(0.1, 0.1, 10.3);
  Eigen::Matrix3d ball_rot = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(7, 0, ball_pos, ball_rot);
  workspace.set_ball_size(Eigen::VectorXd::Constant(1, ball_radius));

  Eigen::Vector3d box_pos1(0.3, 0.5, 0.35);
  Eigen::Matrix3d box_rot1 = Eigen::Matrix3d::Identity();
  workspace.set_coll_pos(8, 0, box_pos1, box_rot1);
  workspace.set_box_size(Eigen::VectorXd::Constant(1, b1[0]),
                         Eigen::VectorXd::Constant(1, b1[1]),
                         Eigen::VectorXd::Constant(1, b1[2]), 1);

  Eigen::Vector3d box_pos2(0.3, 0.5 - b1[1] / 2.0, 0.35 / 2.0);
  Eigen::Matrix3d box_rot2 = rotation_x(90.0 * deg2rad);
  workspace.set_coll_pos(9, 0, box_pos2, box_rot2);
  workspace.set_box_size(Eigen::VectorXd::Constant(1, b2[0]),
                         Eigen::VectorXd::Constant(1, b2[1]),
                         Eigen::VectorXd::Constant(1, b2[2]), 2);

  Eigen::Vector3d box_pos3(0.3, 0.5 + b1[1] / 2.0, 0.35 / 2.0);
  Eigen::Matrix3d box_rot3 = rotation_x(90.0 * deg2rad);
  workspace.set_coll_pos(10, 0, box_pos3, box_rot3);
  workspace.set_box_size(Eigen::VectorXd::Constant(1, b3[0]),
                         Eigen::VectorXd::Constant(1, b3[1]),
                         Eigen::VectorXd::Constant(1, b3[2]), 3);

  Eigen::Vector3d box_pos4(0.3 + b1[0] / 2.0, box_pos1[1], box_pos1[2] / 2.0);
  Eigen::Matrix3d box_rot4 = Ry;
  workspace.set_coll_pos(11, 0, box_pos4, box_rot4);
  workspace.set_box_size(Eigen::VectorXd::Constant(1, b4[0]),
                         Eigen::VectorXd::Constant(1, b4[1]),
                         Eigen::VectorXd::Constant(1, b4[2]), 4);

  workspace.allocate(rmodel, batch_size, seq_len, rmodel.nv, eq_dim, n_thread);
  workspace.init_geometry(rmodel, batch_size);

  Eigen::VectorXd q_start(6);
  q_start << -1.4835299, -1.6755161, -2.2165682, -1.5707963, 0.2094395,
      -0.5759587;

  pinocchio::framesForwardKinematics(rmodel, data, q_start);
  Eigen::Matrix3d R_target = data.oMf[tool_id].rotation();
  Eigen::Matrix3d R = Ry;
  Eigen::Vector3d v(0.4, 0.35, 0.5);
  pinocchio::SE3 end_SE3(R_target, v);
  Eigen::VectorXd end_log = pinocchio::log6(end_SE3).toVector();
  Eigen::MatrixXd states_init = q_start.transpose();

  Eigen::Matrix<double, 6, 1> p_0;
  p_0.setRandom();
  Eigen::Matrix<double, 6, 1> p_1;
  p_1.setRandom();

  p_1 = pinocchio::log6(end_SE3).toVector();
  pinocchio::SE3 pos = data.oMf[tool_id];
  pos.translation() -= Eigen::Vector3d(1.0, 0.0, 1.0);
  p_0 = pinocchio::log6(pos).toVector();

  pos = end_SE3;
  pos.translation() -= Eigen::Vector3d(0.3, 0.0, -0.4);
  pos.rotation() = R_target;

  p_1 = pinocchio::log6(pos).toVector();
  p_0 = pinocchio::log6(pos).toVector();

  Eigen::Tensor<double, 3, Eigen::RowMajor> p_np(1, seq_len, 6);
  for (std::size_t t = 0; t < seq_len; ++t) {
    for (int j = 0; j < 6; ++j) {
      if (t < seq_len / 2)
        p_np(0, t, j) = p_0(j);
      else
        p_np(0, t, j) = p_1(j);
    }
  }
  pinocchio::container::aligned_vector<pinocchio::SE3> targets = {end_SE3};
  forward_pass2(workspace, p_np, p_np, p_np, states_init, rmodel, n_thread,
                targets, dt);

  Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output(1, seq_len, rmodel.nv);
  // backward_pass2(workspace, rmodel, grad_output, 1, 1);

  auto get_qp =
      [&](size_t t) -> std::optional<proxsuite::proxqp::dense::QP<double>> & {
    return workspace.workspace_.qp[t];
  };

  for (size_t n_coll = 0; n_coll < workspace.pairs.size(); ++n_coll) {
    auto [coll_a, coll_b] = workspace.pairs[n_coll];
    for (size_t time = 0; time < seq_len; ++time) {
      size_t j1_id = workspace.get_geom(coll_a, 0).parentJoint;
      size_t j2_id = workspace.get_geom(coll_b, 0).parentJoint;
      Eigen::MatrixXd fd_dJcoll_dq(rmodel.nv, rmodel.nv);
      auto &ana_dJcoll_dq = workspace.dJcoll_dq[0];
      Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(
          workspace.positions_.data() + time * rmodel.nv,
          static_cast<Eigen::Index>(rmodel.nv));
      diffcoal::ContactDerivative &cdres = workspace.cdres[n_coll][time];
      compute_d_dist_and_d_Jcoll(workspace, rmodel, data, j1_id, j2_id, 0, time,
                                 0, q, n_coll);
      fd_dJcoll_dq = fd_dJcoll_dq_(workspace, rmodel, data, n_coll, 0, time,
                                   coll_a, coll_b, 0, time, q);
      Eigen::MatrixXd diff = fd_dJcoll_dq - ana_dJcoll_dq;

      double norm_inf_abs = diff.cwiseAbs().maxCoeff();
      double norm_ana_inf = ana_dJcoll_dq.cwiseAbs().maxCoeff();
      double norm_inf_rel = norm_inf_abs / (norm_ana_inf + 1e-5);
      if (fd_dJcoll_dq.isApprox(ana_dJcoll_dq, sqrt(eps)) &&
          norm_inf_abs < tol_abs && norm_inf_rel < tol_rel) {
        std::cout << "✅ Matrices are approximately equal within tolerance "
                  << eps << std::endl;
        std::cout << "‣ abs_inf_norm: " << norm_inf_abs
                  << "  | rel_inf_norm: " << norm_inf_rel
                  << "  | coll_a: " << coll_a << "  | coll_b: " << coll_b
                  << std::endl;
      } else {
        auto &n = workspace.n[0];
        std::cout << "normal" << n << std::endl;
        std::cout << "❌ Matrices differ beyond tolerance " << eps << std::endl;
        std::cout << "‣ abs_inf_norm: " << norm_inf_abs
                  << "  | rel_inf_norm: " << norm_inf_rel
                  << "  | coll_a: " << coll_a << "  | coll_b: " << coll_b
                  << std::endl;

        std::cout << "\n🔹 Finite-difference Jacobian:\n"
                  << fd_dJcoll_dq << std::endl;
        std::cout << "\n🔹 Analytic Jacobian:\n" << ana_dJcoll_dq << std::endl;

        auto &term_1_A = workspace.term_1_A[0];
        auto &term_1_B = workspace.term_1_B[0];
        // coal::CollisionResult &cres = workspace.cres[n_coll][time];
        // dJ_coll_first_term(workspace, rmodel, data, cres, 0, j1_id, j2_id);
        Eigen::MatrixXd fd_dJcoll_dq_1(rmodel.nv, rmodel.nv);
        Eigen::MatrixXd ana_dJcoll_dq_1(rmodel.nv, rmodel.nv);
        std::cout << "term_1_A " << term_1_A << std::endl;
        std::cout << "term_1_B " << term_1_B << std::endl;
        auto &dn_dq = workspace.dn_dq[0];

        ana_dJcoll_dq_1 = term_1_A + term_1_B;
        fd_dJcoll_dq_1 =
            fd_dJcoll_dq_<true, false>(workspace, rmodel, data, n_coll, 0, time,
                                       coll_a, coll_b, 0, time, q);
        double norm_inf_abs_1 =
            (ana_dJcoll_dq_1 - fd_dJcoll_dq_1).cwiseAbs().maxCoeff();

        double norm_inf_rel_1 =
            (ana_dJcoll_dq_1 - fd_dJcoll_dq_1)
                .cwiseAbs()
                .cwiseQuotient((ana_dJcoll_dq_1.cwiseAbs().array() +
                                fd_dJcoll_dq_1.cwiseAbs().array() + 1e-5)
                                   .matrix())
                .maxCoeff();

        bool approx_ok = ana_dJcoll_dq_1.isApprox(fd_dJcoll_dq_1, tol_rel);

        if (approx_ok && norm_inf_abs_1 < tol_abs && norm_inf_rel_1 < tol_rel) {
          std::cout << "✅ dJcoll/dq_1 matches FD (abs < " << tol_abs
                    << ", rel < " << tol_rel << ")\n";
        } else {
          std::cout << "❌ dJcoll/dq_1 mismatch\n";
          std::cout << "  max abs error = " << norm_inf_abs_1 << "\n";
          std::cout << "  max rel error = " << norm_inf_rel_1 << "\n";
          std::cout << "Analytic:\n"
                    << ana_dJcoll_dq_1 << "\nFD:\n"
                    << fd_dJcoll_dq_1 << "\n";
        }

        Eigen::MatrixXd &term_2_A = workspace.term_2_A[0];
        Eigen::Matrix<double, 6, Eigen::Dynamic> &term_2_B =
            workspace.term_2_B[0];
        Eigen::MatrixXd fd_dJcoll_dq_2(rmodel.nv, rmodel.nv);
        Eigen::MatrixXd ana_dJcoll_dq_2(rmodel.nv, rmodel.nv);

        // dJ_coll_second_term(workspace, rmodel, data, j1_id, j2_id, 0, 0,
        // coll_a,
        //                     coll_b, cdres);
        ana_dJcoll_dq_2 = term_2_A + term_2_B;
        fd_dJcoll_dq_2 = fd_dJcoll_dq_(workspace, rmodel, data, n_coll, 0, time,
                                       coll_a, coll_b, 0, time, q);
        double norm_inf_abs_2 =
            (ana_dJcoll_dq_2 - fd_dJcoll_dq_2).cwiseAbs().maxCoeff();

        double norm_inf_rel_2 =
            (ana_dJcoll_dq_2 - fd_dJcoll_dq_2)
                .cwiseAbs()
                .cwiseQuotient((ana_dJcoll_dq_2.cwiseAbs().array() +
                                fd_dJcoll_dq_2.cwiseAbs().array() + 1e-5)
                                   .matrix())
                .maxCoeff();

        bool approx_ok_2 = ana_dJcoll_dq_2.isApprox(fd_dJcoll_dq_2, tol_rel);

        if (approx_ok_2 && norm_inf_abs_2 < tol_abs &&
            norm_inf_rel_2 < tol_rel) {
          std::cout << "✅ dJcoll/dq_2 matches FD (abs < " << tol_abs
                    << ", rel < " << tol_rel << ")\n";
        } else {
          std::cout << "❌ dJcoll/dq_2 mismatch\n";
          std::cout << "  max abs error = " << norm_inf_abs_2 << "\n";
          std::cout << "  max rel error = " << norm_inf_rel_2 << "\n";
          std::cout << "Analytic:\n"
                    << ana_dJcoll_dq_2 << "\nFD:\n"
                    << fd_dJcoll_dq_2 << "\n";
        }
        std::cout << "\nPress Enter to continue..." << std::endl;

        Eigen::Matrix<double, 3, Eigen::Dynamic> &dr1_dq = workspace.dr1_dq[0];
        Eigen::Matrix<double, 3, Eigen::Dynamic> &dr2_dq = workspace.dr2_dq[0];

        Eigen::MatrixXd fd_dr1 = fd_dr1_dq(workspace, rmodel, data, 0, n_coll,
                                           time, coll_a, coll_b, 0, q);
        Eigen::MatrixXd fd_dr2 = fd_dr1_dq(workspace, rmodel, data, 0, n_coll,
                                           time, coll_b, coll_a, 0, q);

        Eigen::MatrixXd diff1 = fd_dr1 - dr1_dq;
        Eigen::MatrixXd diff2 = fd_dr2 - dr2_dq;

        double abs1 = diff1.cwiseAbs().maxCoeff();
        double abs2 = diff2.cwiseAbs().maxCoeff();

        double rel1 = abs1 / (dr1_dq.cwiseAbs().maxCoeff() + 1e-8);
        double rel2 = abs2 / (dr2_dq.cwiseAbs().maxCoeff() + 1e-8);

        std::cout << "\n🔸 Test dr1_dq:\n";
        if (abs1 < tol_abs && rel1 < tol_rel) {
          std::cout << "✅ OK  abs_inf=" << abs1 << "  rel_inf=" << rel1
                    << std::endl;
        } else {
          std::cout << "❌ dr1_dq mismatch  abs_inf=" << abs1
                    << "  rel_inf=" << rel1 << std::endl;
          std::cout << "Analytic:\n"
                    << dr1_dq << "\nFD:\n"
                    << fd_dr1 << std::endl;
        }
        std::cout << "Analytic:\n"
                  << dr1_dq << "\nFD:\n"
                  << fd_dr1 << std::endl;

        std::cout << "\n🔸 Test dr2_dq:\n";
        if (abs2 < tol_abs && rel2 < tol_rel) {
          std::cout << "✅ OK  abs_inf=" << abs2 << "  rel_inf=" << rel2
                    << std::endl;
        } else {
          std::cout << "❌ dr2_dq mismatch  abs_inf=" << abs2
                    << "  rel_inf=" << rel2 << std::endl;
          std::cout << "Analytic:\n"
                    << dr2_dq << "\nFD:\n"
                    << fd_dr2 << std::endl;
        }
        std::cout << "Analytic:\n"
                  << dr2_dq << "\nFD:\n"
                  << fd_dr2 << std::endl;

        Eigen::MatrixXd fd_dn = fd_dn_dq(workspace, rmodel, data, 0, n_coll,
                                         time, coll_a, coll_b, 0, q);

        Eigen::MatrixXd diff = fd_dn - dn_dq;

        double abs_norm = diff.cwiseAbs().maxCoeff();
        double ref_norm = dn_dq.cwiseAbs().maxCoeff();
        double rel_norm = abs_norm / (ref_norm + 1e-8);

        std::cout << "\n🔸 Test dn_dq:\n";
        if (abs_norm < tol_abs && rel_norm < tol_rel) {
          std::cout << "✅ dn_dq matches FD (abs < " << tol_abs << ", rel < "
                    << tol_rel << ")\n";
          std::cout << "‣ abs_inf_norm: " << abs_norm
                    << "  | rel_inf_norm: " << rel_norm << std::endl;
        } else {
          std::cout << "❌ dn_dq mismatch\n";
          std::cout << "  max abs error = " << abs_norm
                    << "\n  max rel error = " << rel_norm << "\n";
          std::cout << "Analytic dn_dq:\n"
                    << dn_dq << "\nFinite‑difference dn_dq:\n"
                    << fd_dn << std::endl;
        }

        std::cout << fd_dc_dq(workspace, rmodel, data, 0, n_coll, time, coll_a,
                              coll_b, 0, q)
                  << std::endl;
        std::cout << -fd_dc_dq(workspace, rmodel, data, 0, n_coll, time, coll_b,
                               coll_a, 0, q)
                  << std::endl;

        std::cin.get();
      }
    }
  }

  return false;
}