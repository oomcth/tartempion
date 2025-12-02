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
#include <shared_mutex>
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

class AtomicSphere {
public:
  explicit AtomicSphere(double radius = 0.1)
      : ptr_(std::make_shared<const coal::Sphere>(radius)) {}

  operator coal::Sphere() const {
    auto tmp = std::atomic_load(&ptr_);
    return *tmp;
  }

  AtomicSphere &operator=(const coal::Sphere &s) {
    auto new_ptr = std::make_shared<const coal::Sphere>(s);
    std::atomic_store(&ptr_, new_ptr);
    return *this;
  }

  std::shared_ptr<const coal::Sphere> getPtr() const {
    return std::atomic_load(&ptr_);
  }

private:
  std::shared_ptr<const coal::Sphere> ptr_;
};

class AtomicCapsule {
public:
  explicit AtomicCapsule(double radius = 0.05, double length = 0.5)
      : ptr_(std::make_shared<const coal::Capsule>(radius, length)) {}

  operator coal::Capsule() const {
    auto tmp = std::atomic_load(&ptr_);
    return *tmp;
  }

  AtomicCapsule &operator=(const coal::Capsule &c) {
    auto new_ptr = std::make_shared<const coal::Capsule>(c);
    std::atomic_store(&ptr_, new_ptr);
    return *this;
  }

  std::shared_ptr<const coal::Capsule> getPtr() const {
    return std::atomic_load(&ptr_);
  }

private:
  std::shared_ptr<const coal::Capsule> ptr_;
};

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

  Eigen::Tensor<double, 3, Eigen::RowMajor> p_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> A_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> b_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> positions_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> articular_speed_;

  std::vector<Eigen::Tensor<double, 3, Eigen::ColMajor>> Hessian;

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
  std::vector<Eigen::MatrixXd> term_A;
  std::vector<Eigen::MatrixXd> term_B;
  std::vector<Matrix6xd> J1;
  std::vector<Matrix6xd> J_1;
  std::vector<Matrix6xd> J2;
  std::vector<Matrix6xd> J_2;
  std::vector<Eigen::RowVectorXd> J_coll;
  std::vector<Vector6d> grad_err_;
  std::vector<Eigen::VectorXd> ddist;
  std::vector<Vector6d> grad_p_;
  std::vector<Eigen::VectorXd> temp;
  std::vector<Eigen::VectorXd> last_q;
  std::vector<Eigen::VectorXd> log_diff;
  std::vector<Eigen::VectorXd> grad_target;
  std::vector<Matrix66d> dloss_dq_tmp1;
  std::vector<Eigen::MatrixXd> dloss_dq_tmp2;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::Tensor<double, 2>> temp_tensor;
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
  std::vector<Eigen::MatrixXd> dout;
  std::vector<Eigen::Vector3d> c;
  std::vector<Eigen::Vector3d> dcj;
  std::vector<Eigen::RowVectorXd> term1;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> J_diff;
  std::vector<Eigen::Vector3d> n;
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> dr1_dq;
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

  void set_L1_weight(double L1_w);
  void set_collisions_safety_margin(double margin);
  void set_collisions_strength(double margin);
  void set_rot_weight(double L1_w);
  void set_q_reg(double q_reg);
  void set_lambda(double lambda);
  void set_tool_id(size_t id);
  void set_bound(double bound);
  void init_geometry(pinocchio::Model model);

  Qp_Workspace workspace_;

  void allocate(const pinocchio::Model &model, size_t batch_size,
                size_t seq_len, size_t cost_dim, size_t eq_dim,
                size_t num_thread);
  void reset();

  Eigen::Tensor<double, 3, Eigen::RowMajor> Get_positions_();
  std::vector<Eigen::VectorXd> get_last_q();
  std::vector<Vector6d> grad_p();
  std::vector<Eigen::VectorXd> grad_b();

  std::vector<pinocchio::GeometryModel> gmodel;
  std::vector<pinocchio::GeometryData> gdata;

  std::vector<std::pair<int, int>> pairs;
  void add_pair(int a, int b) {
    if (a > b)
      std::swap(a, b);
    if (a == 0 && b == 1)
      throw std::runtime_error("Pair (0,1) is not allowed");
    auto it = std::find(pairs.begin(), pairs.end(), std::make_pair(a, b));
    if (it != pairs.end())
      throw std::runtime_error("Pair (" + std::to_string(a) + "," +
                               std::to_string(b) + ") already exists");

    pairs.emplace_back(a, b);
  }

  const coal::CollisionGeometry &get_coal_obj(size_t idx) {
    switch (idx) {
    case 0:
      return effector_ball;
    case 1:
      return arm_cylinder;
    case 2:
      return plane;
    case 3:
      return *cylinder.getPtr();
    case 4:
      return *ball.getPtr();

    default:
      throw std::out_of_range("Invalid object index");
    }
  }
  const coal::Sphere effector_ball = coal::Sphere(0.1);
  const coal::Capsule arm_cylinder = coal::Capsule(0.05, 0.5);
  const coal::Box plane = coal::Box(1e6, 1e6, 10);
  AtomicCapsule cylinder = AtomicCapsule();
  AtomicSphere ball = AtomicSphere();
  const pinocchio::GeometryObject &get_geom(size_t idx) {
    const std::optional<pinocchio::GeometryObject> *opt_ptr = nullptr;

    switch (idx) {
    case 0:
      opt_ptr = &geom_end_eff;
      break;
    case 1:
      opt_ptr = &geom_arm_cylinder;
      break;
    case 2:
      opt_ptr = &geom_plane;
      break;
    case 3:
      opt_ptr = &geom_cylinder;
      break;
    case 4:
      opt_ptr = &geom_ball;
      break;
    default:
      throw std::out_of_range("Invalid object index");
    }
    if (!opt_ptr->has_value())
      throw std::runtime_error("Geometry object index " + std::to_string(idx) +
                               " has no value");
    return opt_ptr->value();
  }
  std::optional<pinocchio::GeometryObject> geom_end_eff;
  std::optional<pinocchio::GeometryObject> geom_arm_cylinder;
  std::optional<pinocchio::GeometryObject> geom_plane;
  std::optional<pinocchio::GeometryObject> geom_cylinder;
  std::optional<pinocchio::GeometryObject> geom_ball;

  std::vector<std::vector<coal::CollisionRequest>> creq;
  std::vector<std::vector<coal::CollisionResult>> cres;
  std::vector<std::vector<coal::CollisionResult>> cres2;

  std::vector<std::vector<diffcoal::ContactDerivativeRequest>> cdreq;
  std::vector<std::vector<diffcoal::ContactDerivative>> cdres;
  std::vector<std::vector<diffcoal::ContactDerivative>> cdres2;

  std::vector<Matrix3xd> dn_dq;
  std::vector<Matrix3xd> dw_dq;
  std::vector<Matrix3xd> dw2_dq;

  Eigen::Ref<Eigen::VectorXd> dloss_dqf(size_t i);

  void view_geom_objects() const {
    using std::cout;
    cout << "\n================= GEOMETRIC OBJECTS =================\n";
    cout << "  [0]  End effector ball\n";
    cout << "  [1]  Arm cylinder\n";
    cout << "  [2]  Ground\n";
    cout << "  [3]  Collision cylinder\n";
    cout << "  [4]  Collisions ball\n";
    cout << "====================================================="
         << std::endl;
  }

  pinocchio::SE3 get_coll_pos(int idx) {
    switch (idx) {
    case 0:
      return pinocchio::SE3(Eigen::Matrix<double, 3, 3>::Identity(),
                            end_eff_pos);

    case 1:
      return pinocchio::SE3(Eigen::Matrix<double, 3, 3>::Identity(),
                            arm_cylinder_pos);

    case 2:
      return pinocchio::SE3(Eigen::Matrix<double, 3, 3>::Identity(), plane_pos);

    case 3:
      return pinocchio::SE3(Eigen::Matrix<double, 3, 3>::Identity(),
                            cylinder_pos);

    case 4:
      return pinocchio::SE3(Eigen::Matrix<double, 3, 3>::Identity(), ball_pos);

    default:
      throw "wrong idx";
    }
  }

  void set_coll_pos(int idx, Eigen::Vector3d pos,
                    Eigen::Matrix<double, 3, 3> rot) {
    switch (idx) {
    case 0:
      end_eff_pos = pos;
      enf_eff_rot = rot;
      break;
    case 1:
      arm_cylinder_pos = pos;
      arm_cylinder_rot = rot;
      break;
    case 2:
      plane_pos = pos;
      plane_rot = rot;
      break;
    case 3:
      cylinder_pos = pos;
      cylinder_rot = rot;
      break;
    case 4:
      ball_pos = pos;
      ball_rot = rot;
      break;
    default:
      throw "wrong idx";
    }
  }

  void set_ball_size(double radius) { ball = coal::Sphere(radius); }
  void set_capsule_size(double radius, double size) {
    cylinder = coal::Capsule(radius, size);
  }

  Eigen::Vector3d end_eff_pos;
  Eigen::Vector3d arm_cylinder_pos;
  Eigen::Vector3d plane_pos;
  Eigen::Vector3d cylinder_pos;
  Eigen::Vector3d ball_pos;

  Eigen::Matrix<double, 3, 3> enf_eff_rot = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 3> arm_cylinder_rot = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 3> plane_rot = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 3> cylinder_rot = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double, 3, 3> ball_rot = Eigen::Matrix3d::Identity();
  QP_pass_workspace2()
      : end_eff_pos(0, 0, 0.), arm_cylinder_pos(0, 0, 0.2), plane_pos(0, 0, -5),
        cylinder_pos(0.25, 0.25, 0.25), ball_pos(0, 0, 0) {}
};

void backward_pass2(
    QP_pass_workspace2 &workspace, const pinocchio::Model &model,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
    size_t num_thread, size_t batch_size);

Eigen::VectorXd
forward_pass2(QP_pass_workspace2 &workspace,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, size_t num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
              double dt);
