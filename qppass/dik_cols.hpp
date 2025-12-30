#pragma once
#include "qp.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <coal/collision.h>
#include <coal/shape/geometric_shapes.h>
#include <diffcoal/contact_derivative.hpp>
#include <diffcoal/contact_derivative_data.hpp>
#include <diffcoal/spatial.hpp>
#include <omp.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/multibody/fcl.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/geometry-object.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <pinocchio/spatial/log.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <ranges>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef COLLISIONS_SUPPORT
inline constexpr bool collisions = true;
#else
inline constexpr bool collisions = false;
#endif

#ifdef DISCARD_NON_CONVERGENT_IK
inline constexpr bool discard_non_convergent_ik = true;
#else
inline constexpr bool discard_non_convergent_ik = false;
#endif

using Vector6d = Eigen::Vector<double, 6>;
using Matrix66d = Eigen::Matrix<double, 6, 6>;
using Matrix3xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

struct QP_pass_workspace2 {
  double lambda = -1;
  size_t batch_size_ = 0;
  size_t seq_len_ = 0;
  size_t cost_dim_ = 0;
  size_t num_thread_ = 0;
  double dt = 1;
  size_t tool_id = 21;
  double lambda_L1 = 0;
  double rot_w = 1;
  double q_reg = 1e-5;
  double safety_margin = 0.01;
  double collision_strength = 20.0;
  bool allow_collisions = false;

  void set_allow_collisions(bool allow_) { allow_collisions = allow_; }

  Eigen::Tensor<double, 3, Eigen::RowMajor> log_target;
  Eigen::Tensor<double, 3, Eigen::RowMajor> A_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> b_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> positions_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> articular_speed_;

  std::vector<Eigen::Tensor<double, 3, Eigen::ColMajor>> Hessian;
  Eigen::MatrixXd mem;
  Eigen::MatrixXd mem2;
  std::vector<pinocchio::Data> data_vec_;
  std::vector<Matrix6xd> jacobians_;
  std::vector<Matrix6xd> grad_J_;
  std::vector<Eigen::MatrixXd> Q_vec_;
  std::vector<Eigen::MatrixXd> J_vec_;
  std::vector<Eigen::MatrixXd> A_thread_mem;
  std::vector<Eigen::MatrixXd> J_frame;
  std::vector<Eigen::MatrixXd> Adj_backward;
  std::vector<Eigen::MatrixXd> J_log;
  std::vector<Matrix66d> adj_diff;
  std::vector<Matrix66d> Adj_vec;
  std::vector<Matrix66d> Jlog_vec;
  std::vector<Matrix6xd> J_frame_vec;
  std::vector<Matrix66d> Jlog_v4;
  std::vector<Eigen::MatrixXd> dJcoll_dq;
  std::vector<Eigen::MatrixXd> term_1_A;
  std::vector<Eigen::MatrixXd> term_1_B;
  std::vector<Matrix6xd> J1;
  std::vector<Matrix6xd> J_1;
  std::vector<Matrix6xd> J2;
  std::vector<Matrix6xd> J_2;
  std::vector<Eigen::RowVectorXd> J_coll;
  std::vector<Vector6d> grad_err_;
  std::vector<Eigen::VectorXd> ddist;
  std::vector<Vector6d> grad_log_target_;
  std::vector<Eigen::VectorXd> temp;
  std::vector<Eigen::VectorXd> last_q;
  std::vector<Eigen::VectorXd> log_diff;
  std::vector<Eigen::VectorXd> grad_target;
  std::vector<Matrix66d> dloss_dq_tmp1;
  std::vector<Eigen::MatrixXd> dloss_dq_tmp2;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> term_2_B;
  std::vector<Eigen::VectorXd> dloss_dq_tmp3;
  std::vector<Eigen::VectorXd> e;
  std::vector<Vector6d> err_vec;
  std::vector<Eigen::VectorXd> padded;
  std::vector<Eigen::VectorXd> ub;
  std::vector<Eigen::VectorXd> lb;
  std::vector<Eigen::MatrixXd> G;
  std::vector<Eigen::Vector3d> r1;
  std::vector<Eigen::Vector3d> r2;
  std::vector<Eigen::Vector3d> w1;
  std::vector<Eigen::Vector3d> w2;
  std::vector<Eigen::Vector3d> w_diff;
  std::vector<Eigen::MatrixXd> term_2_A;
  std::vector<Eigen::Vector3d> c;
  std::vector<Eigen::RowVector<double, 6>> temp_ddist;
  std::vector<Eigen::Vector3d> dcj;
  std::vector<Eigen::RowVectorXd> term1;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> J_diff;
  std::vector<Eigen::Vector3d> n;
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> dr1_dq;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> dr2_dq;
  std::vector<Eigen::Matrix<double, 3, 6>> tmp1_dr1_dq, tmp2_dr1_dq;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> tmp3_dr1_dq,
      tmp4_dr1_dq;
  std::vector<Vector6d> v1;
  std::vector<Vector6d> v2;
  std::vector<Vector6d> v3;
  std::vector<Vector6d> target_vec;
  std::vector<Vector6d> temp_direct;
  std::vector<Vector6d> last_log_vec;
  std::vector<Vector6d> log_indirect_1_vec;
  std::vector<Vector6d> w_vec;
  std::vector<Vector6d> e_vec;
  std::vector<Vector6d> e_scaled_vec;
  std::vector<Vector6d> grad_e_vec;
  std::vector<Vector6d> grad_target_vec;
  std::vector<Vector6d> sign_e_scaled_vec;
  std::vector<Eigen::VectorXd> dloss_dq;
  std::vector<Eigen::VectorXd> dloss_dq_diff;
  std::vector<pinocchio::SE3> last_T;
  std::vector<pinocchio::SE3> target_placement_vec;
  std::vector<pinocchio::SE3> current_placement_vec;
  std::vector<pinocchio::SE3> diff;
  std::vector<pinocchio::Motion> last_logT;
  std::vector<pinocchio::Motion> target;
  std::vector<size_t> steps_per_batch;
  std::vector<bool> discarded;
  std::vector<Matrix66d> joint_to_frame_action;
  Eigen::VectorXd losses;
  std::vector<Matrix3xd> dn_dq;
  std::vector<Matrix3xd> dw_dq;
  std::vector<Matrix3xd> dw2_dq;

  void set_L1_weight(double L1_w);
  void set_collisions_safety_margin(double margin);
  void set_collisions_strength(double margin);
  void set_rot_weight(double L1_w);
  void set_q_reg(double q_reg);
  void set_lambda(double lambda);
  void set_tool_id(size_t id);
  void init_geometry(pinocchio::Model model, size_t batch_size);

  Qp_Workspace workspace_;

  void allocate(const pinocchio::Model &model, size_t batch_size,
                size_t seq_len, size_t cost_dim, size_t eq_dim,
                size_t num_thread);

  void pre_allocate(size_t batch_size);
  void reset();

  Eigen::Tensor<double, 3, Eigen::RowMajor> Get_positions_();
  std::vector<Eigen::VectorXd> get_last_q();
  std::vector<Vector6d> grad_log_target();
  std::vector<Eigen::VectorXd> grad_b();

  std::vector<pinocchio::GeometryModel> gmodel;
  std::vector<pinocchio::GeometryData> gdata;

  pinocchio::GeometryModel get_gmodel(size_t i) { return gmodel[i]; }

  std::vector<std::pair<int, int>> pairs;
  void add_pair(int a, int b) {
    // if (a > b)
    //   std::swap(a, b);
    auto it = std::find(pairs.begin(), pairs.end(), std::make_pair(a, b));
    if (it != pairs.end())
      throw std::runtime_error("Pair (" + std::to_string(a) + "," +
                               std::to_string(b) + ") already exists");

    pairs.emplace_back(a, b);
  }

  coal::CollisionGeometry &get_coal_obj(size_t idx, size_t batch_id) {
    switch (idx) {
    case 0:
      return effector_ball[batch_id];
    case 1:
      return arm[batch_id];
    case 2:
      return arm_1[batch_id];
    case 3:
      return arm_2[batch_id];
    case 4:
      return arm_3[batch_id];
    case 5:
      return plane[batch_id];
    case 6:
      return cylinder[batch_id];
    case 7:
      return ball[batch_id];
    case 8:
      return box1[batch_id];
    case 9:
      return box2[batch_id];
    case 10:
      return box3[batch_id];
    case 11:
      return box4[batch_id];
    case 12:
      return arm_4[batch_id];
    case 13:
      return arm_5[batch_id];

    default:
      throw std::out_of_range("Invalid object index");
    }
  }
  std::vector<coal::Sphere> effector_ball;
  std::vector<coal::Cylinder> arm;
  std::vector<coal::Sphere> arm_1;
  std::vector<coal::Sphere> arm_2;
  std::vector<coal::Sphere> arm_3;
  std::vector<coal::Sphere> arm_4;
  std::vector<coal::Sphere> arm_5;
  std::vector<coal::Box> plane;
  std::vector<coal::Capsule> cylinder;
  std::vector<coal::Sphere> ball;
  std::vector<coal::Box> box1;
  std::vector<coal::Box> box2;
  std::vector<coal::Box> box3;
  std::vector<coal::Box> box4;

  std::vector<std::optional<pinocchio::GeometryObject>> geom_end_eff;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm_1;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm_2;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm_3;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm_4;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_arm_5;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_plane;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_cylinder;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_ball;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_box1;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_box2;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_box3;
  std::vector<std::optional<pinocchio::GeometryObject>> geom_box4;

  std::vector<std::vector<coal::CollisionRequest>> creq;
  std::vector<std::vector<coal::CollisionResult>> cres;
  std::vector<std::vector<coal::CollisionResult>> cres2;

  std::vector<std::vector<diffcoal::ContactDerivativeRequest>> cdreq;
  std::vector<std::vector<diffcoal::ContactDerivative>> cdres;
  std::vector<std::vector<diffcoal::ContactDerivative>> cdres2;

  std::vector<Eigen::Vector3d> end_eff_pos;
  std::vector<Eigen::Vector3d> arm_pos;
  std::vector<Eigen::Vector3d> arm_1_pos;
  std::vector<Eigen::Vector3d> arm_2_pos;
  std::vector<Eigen::Vector3d> arm_3_pos;
  std::vector<Eigen::Vector3d> arm_4_pos;
  std::vector<Eigen::Vector3d> arm_5_pos;
  std::vector<Eigen::Vector3d> plane_pos;
  std::vector<Eigen::Vector3d> cylinder_pos;
  std::vector<Eigen::Vector3d> ball_pos;
  std::vector<Eigen::Vector3d> box_pos1;
  std::vector<Eigen::Vector3d> box_pos2;
  std::vector<Eigen::Vector3d> box_pos3;
  std::vector<Eigen::Vector3d> box_pos4;

  std::vector<Eigen::Matrix<double, 3, 3>> end_eff_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_1_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_2_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_3_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_4_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> arm_5_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> plane_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> cylinder_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> ball_rot;
  std::vector<Eigen::Matrix<double, 3, 3>> box_rot1;
  std::vector<Eigen::Matrix<double, 3, 3>> box_rot2;
  std::vector<Eigen::Matrix<double, 3, 3>> box_rot3;
  std::vector<Eigen::Matrix<double, 3, 3>> box_rot4;

  void view_geom_objects() const {
    spdlog::info(fmt::runtime(
        "\n================= COLLISION OBJECTS =================\n"
        "  [0]  End effector (ball)\n"
        "  [1]  Arm\n"
        "  [2]  Arm segment 1\n"
        "  [3]  Arm segment 2\n"
        "  [4]  Arm segment 3\n"
        "  [5]  Ground plane\n"
        "  [6]  Collision cylinder\n"
        "  [7]  Collision ball\n"
        "  [8]  Collision box 1\n"
        "  [9]  Collision box 2\n"
        "  [10] Collision box 3\n"
        "  [11] Collision box 4\n"
        "  [12] Arm segment 4\n"
        "  [13] Arm segment 5\n"
        "=====================================================\n"));
  }

  Eigen::Ref<Eigen::VectorXd> dloss_dqf(size_t i);

  pinocchio::GeometryObject get_geom(size_t idx, size_t batch_id) const {
    const std::optional<pinocchio::GeometryObject> *opt_ptr = nullptr;

    switch (idx) {
    case 0:
      opt_ptr = &geom_end_eff[batch_id];
      break;
    case 1:
      opt_ptr = &geom_arm[batch_id];
      break;
    case 2:
      opt_ptr = &geom_arm_1[batch_id];
      break;
    case 3:
      opt_ptr = &geom_arm_2[batch_id];
      break;
    case 4:
      opt_ptr = &geom_arm_3[batch_id];
      break;
    case 5:
      opt_ptr = &geom_plane[batch_id];
      break;
    case 6:
      opt_ptr = &geom_cylinder[batch_id];
      break;
    case 7:
      opt_ptr = &geom_ball[batch_id];
      break;
    case 8:
      opt_ptr = &geom_box1[batch_id];
      break;
    case 9:
      opt_ptr = &geom_box2[batch_id];
      break;
    case 10:
      opt_ptr = &geom_box3[batch_id];
      break;
    case 11:
      opt_ptr = &geom_box4[batch_id];
      break;
    case 12:
      opt_ptr = &geom_arm_4[batch_id];
      break;
    case 13:
      opt_ptr = &geom_arm_5[batch_id];
      break;
    default:
      throw std::out_of_range("Invalid object index");
    }

    if (!opt_ptr->has_value())
      throw std::runtime_error("Geometry object " + std::to_string(idx) +
                               " is empty");

    return opt_ptr->value();
  }

  pinocchio::SE3 get_coll_pos(int idx, size_t batch_id) {
    switch (idx) {
    case 0:
      return pinocchio::SE3(end_eff_rot[batch_id], end_eff_pos[batch_id]);

    case 1:
      return pinocchio::SE3(arm_rot[batch_id], arm_pos[batch_id]);

    case 2:
      return pinocchio::SE3(arm_1_rot[batch_id], arm_1_pos[batch_id]);

    case 3:
      return pinocchio::SE3(arm_2_rot[batch_id], arm_2_pos[batch_id]);

    case 4:
      return pinocchio::SE3(arm_3_rot[batch_id], arm_3_pos[batch_id]);

    case 5:
      return pinocchio::SE3(plane_rot[batch_id], plane_pos[batch_id]);

    case 6:
      return pinocchio::SE3(cylinder_rot[batch_id], cylinder_pos[batch_id]);

    case 7:
      return pinocchio::SE3(ball_rot[batch_id], ball_pos[batch_id]);

    case 8:
      return pinocchio::SE3(box_rot1[batch_id], box_pos1[batch_id]);

    case 9:
      return pinocchio::SE3(box_rot2[batch_id], box_pos2[batch_id]);

    case 10:
      return pinocchio::SE3(box_rot3[batch_id], box_pos3[batch_id]);

    case 11:
      return pinocchio::SE3(box_rot4[batch_id], box_pos4[batch_id]);

    case 12:
      return pinocchio::SE3(arm_4_rot[batch_id], arm_4_pos[batch_id]);

    case 13:
      return pinocchio::SE3(arm_5_rot[batch_id], arm_5_pos[batch_id]);

    default:
      throw std::runtime_error("wrong idx");
    }
  }

  void set_coll_pos(size_t idx, size_t batch_id, Eigen::Vector3d pos,
                    Eigen::Matrix<double, 3, 3> rot) {
    switch (idx) {
    case 0:
      end_eff_pos[batch_id] = pos;
      end_eff_rot[batch_id] = rot;
      break;
    case 1:
      arm_pos[batch_id] = pos;
      arm_rot[batch_id] = rot;
      break;
    case 2:
      arm_1_pos[batch_id] = pos;
      arm_1_rot[batch_id] = rot;
      break;
    case 3:
      arm_2_pos[batch_id] = pos;
      arm_2_rot[batch_id] = rot;
      break;
    case 4:
      arm_3_pos[batch_id] = pos;
      arm_3_rot[batch_id] = rot;
      break;
    case 5:
      plane_pos[batch_id] = pos;
      plane_rot[batch_id] = rot;
      break;
    case 6:
      cylinder_pos[batch_id] = pos;
      cylinder_rot[batch_id] = rot;
      break;
    case 7:
      ball_pos[batch_id] = pos;
      ball_rot[batch_id] = rot;
      break;
    case 8:
      box_pos1[batch_id] = pos;
      box_rot1[batch_id] = rot;
      break;
    case 9:
      box_pos2[batch_id] = pos;
      box_rot2[batch_id] = rot;
      break;
    case 10:
      box_pos3[batch_id] = pos;
      box_rot3[batch_id] = rot;
      break;
    case 11:
      box_pos4[batch_id] = pos;
      box_rot4[batch_id] = rot;
      break;
    case 12:
      arm_4_pos[batch_id] = pos;
      arm_4_rot[batch_id] = rot;
      break;
    case 13:
      arm_5_pos[batch_id] = pos;
      arm_5_rot[batch_id] = rot;
      break;
    default:
      throw std::runtime_error("wrong idx");
    }
  }
  void set_all_coll_pos(size_t idx,
                        const Eigen::Ref<const Eigen::MatrixXd> &pos,
                        const Eigen::Ref<const Eigen::MatrixXd> &rot) {
    if (pos.cols() != 3)
      throw std::runtime_error("pos must have shape (N,3)");
    if (rot.cols() != 3 || rot.rows() % 3 != 0)
      throw std::runtime_error("rot must have shape (3*N,3)");

    const int N = static_cast<int>(pos.rows());
    if (rot.rows() != 3 * N)
      throw std::runtime_error("rot.rows() != 3*N");

    std::vector<Eigen::Vector3d> *p_pos = nullptr;
    std::vector<Eigen::Matrix3d> *p_rot = nullptr;

    switch (idx) {
    case 0:
      p_pos = &end_eff_pos;
      p_rot = &end_eff_rot;
      break;
    case 1:
      p_pos = &arm_pos;
      p_rot = &arm_rot;
      break;
    case 2:
      p_pos = &arm_1_pos;
      p_rot = &arm_1_rot;
      break;
    case 3:
      p_pos = &arm_2_pos;
      p_rot = &arm_2_rot;
      break;
    case 4:
      p_pos = &arm_3_pos;
      p_rot = &arm_3_rot;
      break;
    case 5:
      p_pos = &plane_pos;
      p_rot = &plane_rot;
      break;
    case 6:
      p_pos = &cylinder_pos;
      p_rot = &cylinder_rot;
      break;
    case 7:
      p_pos = &ball_pos;
      p_rot = &ball_rot;
      break;
    case 8:
      p_pos = &box_pos1;
      p_rot = &box_rot1;
      break;
    case 9:
      p_pos = &box_pos2;
      p_rot = &box_rot2;
      break;
    case 10:
      p_pos = &box_pos3;
      p_rot = &box_rot3;
      break;
    case 11:
      p_pos = &box_pos4;
      p_rot = &box_rot4;
      break;
    case 12:
      p_pos = &arm_4_pos;
      p_rot = &arm_4_rot;
      break;
    case 13:
      p_pos = &arm_5_pos;
      p_rot = &arm_5_rot;
      break;
    default:
      throw std::runtime_error("wrong idx");
    }

    p_pos->resize(N);
    p_rot->resize(N);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3>> pos_map(
        pos.data(), N, 3);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3>> rot_map(
        rot.data(), 3 * N, 3);

    for (int i = 0; i < N; ++i) {
      (*p_pos)[i] = pos_map.row(i);
      (*p_rot)[i] = rot_map.middleRows<3>(3 * i);
    }
  }
  void set_ball_size(const Eigen::VectorXd &radius) {
    assert(radius.size() == ball.size());
    auto spheres = ball.begin();

    for (auto r : radius | std::views::all) {
      *spheres++ = coal::Sphere(r);
    }
  }
  void set_capsule_size(const Eigen::VectorXd &radius,
                        const Eigen::VectorXd &size) {
    assert(radius.size() == cylinder.size() && cylinder.size() == size.size());
    auto it = cylinder.begin();
    for (auto [r, s] : std::views::zip(radius, size)) {
      *it++ = coal::Capsule(r, s);
    }
  }

  void set_box_size(const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                    const Eigen::VectorXd &z, std::size_t idx) {

    assert(x.size() == y.size() && y.size() == z.size());
    std::vector<coal::Box> *target_ = nullptr;
    switch (idx) {
    case 1:
      target_ = &box1;
      break;
    case 2:
      target_ = &box2;
      break;
    case 3:
      target_ = &box3;
      break;
    case 4:
      target_ = &box4;
      break;
    default:
      throw std::runtime_error("idx must be 1–4");
    }

    auto &box = *target_;
    box.clear();
    box.reserve(static_cast<std::size_t>(x.size()));
    for (auto [xi, yi, zi] : std::views::zip(x, y, z)) {
      box.emplace_back(xi, yi, zi);
    }
  }

  std::vector<bool> get_discarded() { return discarded; }
  bool echo = true;
  void set_echo(bool echo_) { echo = echo_; }
  QP_pass_workspace2() {}
};

void backward_pass2(QP_pass_workspace2 &workspace,
                    const pinocchio::Model &model,
                    Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output,
                    size_t num_thread, size_t batch_size);

Eigen::VectorXd
forward_pass2(QP_pass_workspace2 &workspace,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &log_target,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, size_t num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
              double dt);

bool compute_coll_matrix(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, size_t thread_id,
                         size_t batch_id, size_t tool_id, unsigned int time,
                         size_t idx, Eigen::Ref<Eigen::VectorXd> q,
                         pinocchio::Data &data, Eigen::Ref<Eigen::VectorXd> ub,
                         Eigen::Ref<Eigen::VectorXd> lb,
                         Eigen::Ref<Eigen::MatrixXd> G);

void pre_allocate_qp(QP_pass_workspace2 &workspace, unsigned int &time,
                     size_t &idx);

void forward_pass_final_computation(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    size_t thread_id, size_t batch_id, size_t seq_len, size_t tool_id,
    const pinocchio::SE3 &T_star, unsigned int time,
    const Eigen::Ref<const Eigen::VectorXd> q_next, pinocchio::Data &data);

void compute_cost(QP_pass_workspace2 &workspace, size_t thread_id, size_t idx);

void compute_target(QP_pass_workspace2 &workspace, const pinocchio::Data &data,
                    const Eigen::Map<const Eigen::VectorXd> p, size_t thread_id,
                    size_t idx, size_t tool_id);

void single_forward_pass(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, size_t thread_id,
                         size_t batch_id, size_t seq_len, size_t cost_dim,
                         size_t eq_dim, size_t tool_id, pinocchio::SE3 T_star);

void compute_frame_hessian(QP_pass_workspace2 &workspace,
                           const pinocchio::Model &model, size_t thread_id,
                           size_t tool_id, pinocchio::Data &data,
                           const Eigen::Ref<Eigen::VectorXd> q);

void backpropagateThroughQ(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                           QP_pass_workspace2 &workspace, size_t thread_id);

void backpropagateThroughJ0(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                            const pinocchio::Model &model,
                            const pinocchio::SE3 &diff,
                            Eigen::Ref<const Eigen::VectorXd> rhs_grad,
                            double lambda, QP_pass_workspace2 &workspace,
                            size_t thread_id);

void backpropagateThroughT(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                           const pinocchio::Model &model, pinocchio::SE3 &diff,
                           Eigen::Ref<Eigen::VectorXd> rhs_grad, double lambda,
                           QP_pass_workspace2 &workspace, size_t thread_id,
                           size_t batch_id, size_t time);

void backpropagateThroughCollisions(Eigen::Ref<Eigen::VectorXd> grad_vec_local,
                                    QP_pass_workspace2 &workspace, size_t time,
                                    size_t batch_id, size_t seq_len,
                                    size_t n_coll, size_t thread_id);

void compute_dn_dq(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t j1_id, size_t j2_id,
                   size_t batch_id, size_t time, size_t thread_id,
                   size_t n_coll);

void compute_dr1_dq(const pinocchio::Model &model, pinocchio::Data &data,
                    size_t j1_id, size_t j2_id,
                    diffcoal::ContactDerivative &cdres, size_t coll,
                    size_t coll_2, size_t batch_id,
                    QP_pass_workspace2 &workspace, size_t thread_id);

void compute_dr2_dq(const pinocchio::Model &model, pinocchio::Data &data,
                    size_t j1_id, size_t j2_id,
                    diffcoal::ContactDerivative &cdres, size_t coll,
                    size_t coll_2, size_t batch_id,
                    QP_pass_workspace2 &workspace, size_t thread_id);

void dJ_coll_first_term(QP_pass_workspace2 &workspace,
                        const pinocchio::Model &model, pinocchio::Data &data,
                        coal::CollisionResult &cres, size_t thread_id,
                        size_t j1_id, size_t j2_id);

void dJ_coll_second_term(QP_pass_workspace2 &workspace,
                         const pinocchio::Model &model, pinocchio::Data &data,
                         size_t j1_id, size_t j2_id, size_t batch_id,
                         size_t thread_id, size_t coll_a, size_t coll_b,
                         diffcoal::ContactDerivative &cdres);

void compute_ddist(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t j1_id, size_t j2_id,
                   size_t batch_id, size_t time, size_t n_coll, size_t coll_a,
                   size_t coll_b, size_t thread_id);

void compute_d_dist_and_d_Jcoll(QP_pass_workspace2 &workspace,
                                const pinocchio::Model &model,
                                pinocchio::Data &data, size_t j1_id,
                                size_t j2_id, size_t batch_id, size_t time,
                                size_t thread_id, Eigen::Ref<Eigen::VectorXd> q,
                                size_t n_coll);

void single_backward_pass(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    size_t thread_id, size_t batch_id, size_t seq_len, size_t cost_dim,
    size_t tool_id, double dt,
    Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output);

template <bool compute_first_term = true, bool compute_second_term = true>
void compute_jcoll(QP_pass_workspace2 &workspace, const pinocchio::Model &model,
                   pinocchio::Data &data, size_t thread_id, size_t n_coll,
                   size_t idx, size_t coll_a, size_t coll_b, size_t batch_id,
                   size_t time, Eigen::Ref<Eigen::VectorXd> ub,
                   Eigen::Ref<Eigen::VectorXd> lb,
                   Eigen::Ref<Eigen::MatrixXd> G, Eigen::Ref<Eigen::VectorXd> q,
                   bool compute_kine);

constexpr bool isBox(int val) {
  constexpr std::array<int, 5> specialVals{8, 9, 10, 11, 5};
  return std::find(specialVals.begin(), specialVals.end(), val) !=
         specialVals.end();
}

constexpr bool isCapsule(int val) {
  constexpr std::array<int, 2> specialVals{1, 6};
  return std::find(specialVals.begin(), specialVals.end(), val) !=
         specialVals.end();
}