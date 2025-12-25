#include "dik_cols.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/model.hpp"
#include <cmath>
#include <csignal>
#include <pinocchio/parsers/urdf.hpp>

double constexpr eps = 1e-5;
constexpr double tol_abs = 1e-4;
constexpr double tol_rel = 1e-4;

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
    std::cout << "jplus " << Jcoll_plus << std::endl;
    std::cout << "jminus " << Jcoll_minus << std::endl;
    std::cout << "((Jcoll_plus - Jcoll_minus) / (2 * eps)).transpose();"
              << -((Jcoll_plus - Jcoll_minus) / (2 * eps)).transpose() *
                     workspace.dt
              << std::endl;
    dJcoll_dq.col(i) = ((Jcoll_plus - Jcoll_minus) / (2 * eps)).transpose();
  }
  return -dJcoll_dq * workspace.dt;
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

  Eigen::Vector3d eff_pos(.1, .1, .15);
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
  backward_pass2(workspace, rmodel, grad_output, 1, 1);

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
        std::cout << "dn_dq " << dn_dq << std::endl;

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
        // diffcoal::ContactDerivative &cdres = workspace.cdres[n_coll][time];
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
        std::cin.get();
      }
    }
  }

  return false;
}