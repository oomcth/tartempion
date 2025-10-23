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
#include <unsupported/Eigen/CXX11/Tensor>

using Vector6d = Eigen::Vector<double, 6>;
using Matrix66d = Eigen::Matrix<double, 6, 6>;
using Matrix3xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Matrix6xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;

struct QP_pass_workspace2 {
  double lambda = -1;
  int batch_size_ = -1;
  int seq_len_ = -1;
  int cost_dim_ = -1;
  int num_thread_ = -1;
  double mu = 1e-8;
  double bias = 1e-5;
  int n_iter = 1000;
  double dt = 1;
  int tool_id = 21;
  double lambda_L1 = 0;
  double rot_w = 1;
  double q_reg = 1e-5;

  Eigen::Tensor<double, 3, Eigen::RowMajor> p_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> A_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> b_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> positions_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> articular_speed_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> grad_output_;

  std::vector<Eigen::Tensor<double, 3, Eigen::ColMajor>> Hessian;

  std::vector<pinocchio::Data> data_vec_;

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      grad_Q_;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      grad_A_;
  std::vector<Matrix6xd> jacobians_;
  std::vector<Matrix6xd> grad_J_;
  std::vector<Matrix6xd> dJdvq_vec;
  std::vector<Matrix6xd> dJdaq_vec;
  std::vector<Eigen::MatrixXd> Q_vec_;
  std::vector<Eigen::MatrixXd> J_vec_;
  std::vector<Eigen::MatrixXd> A_thread_mem;
  std::vector<Eigen::MatrixXd> grad_AJ;
  std::vector<Matrix6xd> grad_Jeq;
  std::vector<Matrix6xd> gradJ_Q;
  std::vector<Matrix66d> adj;
  std::vector<Eigen::MatrixXd> J_frame;
  std::vector<Eigen::MatrixXd> Adj_backward;
  std::vector<Eigen::MatrixXd> J_log;
  std::vector<Matrix66d> adj_diff;
  std::vector<Matrix66d> Adj_vec;
  std::vector<Matrix66d> Jlog_vec;
  std::vector<Matrix6xd> J_frame_vec;
  std::vector<Matrix66d> Jlog_v4;
  std::vector<Eigen::MatrixXd> dJcoll_dq;
  std::vector<MatrixXd> term_A;
  std::vector<MatrixXd> term_B;
  std::vector<Eigen::VectorXd> localPosition;
  std::vector<Matrix6xd> J1;
  std::vector<Matrix6xd> J_1;
  std::vector<Matrix6xd> J2;
  std::vector<Matrix6xd> J_2;
  std::vector<RowVectorXd> J_coll;
  std::vector<Matrix3d> skew_r1;
  std::vector<Matrix3d> skew_r2;
  std::vector<Vector6d> grad_err_;
  std::vector<Eigen::VectorXd> ddist;
  std::vector<Vector6d> grad_p_;
  std::vector<Eigen::VectorXd> grad_b_;
  std::vector<Eigen::VectorXd> v_vec;
  std::vector<Eigen::VectorXd> a_vec;
  std::vector<Eigen::VectorXd> p_thread_mem;
  std::vector<Eigen::VectorXd> temp;
  std::vector<Eigen::VectorXd> last_q;
  std::vector<Eigen::VectorXd> log_diff;
  std::vector<Eigen::VectorXd> grad_target;
  std::vector<Eigen::VectorXd> e;
  std::vector<Vector6d> err_vec;
  std::vector<Eigen::VectorXd> padded;
  std::vector<Eigen::Vector<double, 1>> ub;
  std::vector<Eigen::Vector<double, 1>> lb;
  std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> G;
  std::vector<Vector3d> r1;
  std::vector<Vector3d> r2;
  std::vector<Vector3d> w1;
  std::vector<Vector3d> w2;
  std::vector<Vector3d> w_diff;
  std::vector<Vector3d> n;
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
  std::vector<int> steps_per_batch;
  std::vector<double> errors_per_batch;
  Eigen::VectorXd losses;

  void set_L1_weight(double L1_w);
  void set_rot_weight(double L1_w);
  void set_q_reg(double q_reg);
  void set_lambda(double lambda);
  void set_tool_id(int id);
  void set_bound(double bound);
  void init_geometry(pinocchio::Model rmodel);

  Qp_Workspace workspace_;

  void allocate(const pinocchio::Model &model, int batch_size, int seq_len,
                int cost_dim, int eq_dim, int num_thread);
  void reset();

  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  grad_A();
  Eigen::Tensor<double, 3, Eigen::RowMajor> Get_positions_();
  std::vector<Eigen::VectorXd> get_last_q();
  std::vector<Vector6d> grad_p();
  std::vector<Eigen::VectorXd> grad_b();

  const double effector_ball_radius = 0.1;
  const double base_ball_radius = 0.25;
  const double elbow_ball_radius = 0.1;
  const int elbow_id = 10;
  const coal::Sphere effector_ball = coal::Sphere(effector_ball_radius);
  const coal::Sphere base_ball = coal::Sphere(base_ball_radius);
  const coal::Sphere elbow_ball = coal::Sphere(elbow_ball_radius);
  const coal::Box plane = coal::Box(10, 10, 0.01);

  std::vector<pinocchio::GeometryModel> gmodel;
  std::vector<pinocchio::GeometryData> gdata;
  std::vector<pinocchio::SE3> end_eff_placement;
  std::vector<pinocchio::SE3> base_placement;
  std::vector<pinocchio::SE3> elbow_placement;
  std::vector<pinocchio::SE3> plane_placement;
  std::optional<pinocchio::GeometryObject> geom_end_eff;
  std::optional<pinocchio::GeometryObject> geom_base;
  std::optional<pinocchio::GeometryObject> geom_elbow;
  std::optional<pinocchio::GeometryObject> geom_plane;
  std::vector<coal::CollisionRequest> creq;
  std::vector<coal::CollisionResult> cres;
  std::vector<coal::CollisionResult> cres2;
  std::vector<diffcoal::ContactDerivativeRequest> cdreq;
  std::vector<diffcoal::ContactDerivative> cdres;
  std::vector<diffcoal::ContactDerivative> cdres2;
  std::vector<Matrix3xd> dn_dq;
  std::vector<Matrix3xd> dw_dq;
  std::vector<Matrix3xd> dw2_dq;
  Eigen::Ref<Eigen::VectorXd> dloss_dqf(int i);
};

void backward_pass2(
    QP_pass_workspace2 &workspace, pinocchio::Model &model,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
    int num_thread, int batch_size);

Eigen::VectorXd
forward_pass2(QP_pass_workspace2 &workspace,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &p,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &A,
              const Eigen::Tensor<double, 3, Eigen::RowMajor> &b,
              const Eigen::MatrixXd &initial_position,
              const pinocchio::Model &model, int num_thread,
              const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star,
              double dt);
