

#include "simpleSE3loss.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <coal/collision.h>
#include <diffcoal/spatial.hpp>
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

std::vector<double> SE3_loss_struct::get_trans_error() const {
  return trans_error;
}

std::vector<double> SE3_loss_struct::get_rot_error() const { return rot_error; }

Eigen::VectorXd SE3_loss_struct::SE3_loss(Eigen::MatrixXd updated,
                                          Eigen::MatrixXd frozen) {
  batch_size = static_cast<int>(frozen.rows());
  Eigen::VectorXd losses(batch_size);
  grad = Eigen::MatrixXd(batch_size, 6);
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    pinocchio::SE3 updated_ =
        pinocchio::exp6(pinocchio::Motion(updated.row(batch_id)));
    pinocchio::SE3 frozen_ =
        pinocchio::exp6(pinocchio::Motion(frozen.row(batch_id)));
    pinocchio::SE3 rel = frozen_.actInv(updated_);
    pinocchio::Motion xi = pinocchio::log6(rel);
    pinocchio::SE3 rel_clamped = pinocchio::exp6(xi);

    Eigen::Matrix<double, 6, 6> Jlog;
    Eigen::Matrix<double, 6, 6> Jexp;
    Jlog.setZero();
    Jexp.setZero();
    pinocchio::Jexp6(pinocchio::Motion(updated.row(batch_id)), Jexp);
    pinocchio::Jlog6(frozen_.actInv(updated_), Jlog);
    grad.row(batch_id) =
        2.0 * pinocchio::log6(rel_clamped).toVector().eval().transpose();
    grad.row(batch_id).tail<3>() *= lambda;
    grad.row(batch_id) *= Jlog * Jexp;
    losses[batch_id] = pinocchio::log6(rel).toVector().squaredNorm();
  }
  return losses;
}
Eigen::MatrixXd SE3_loss_struct::d_SE3_loss() { return grad; }

Eigen::VectorXd SE3_loss_struct::SE3_loss_2(
    const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_pred,
    const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star) {

  batch_size = static_cast<int>(T_pred.size());
  Eigen::VectorXd losses(batch_size);
  grad = Eigen::MatrixXd(batch_size, 6);
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    pinocchio::SE3 rel = T_star[batch_id].actInv(T_pred[batch_id]);
    Eigen::Matrix<double, 6, 6> Jr;
    Jr.setZero();
    Jr = pinocchio::Jlog6(rel.inverse()).transpose();
    grad.row(batch_id) =
        (2.0 * Jr * pinocchio::log6(rel).toVector()).transpose();
    grad.row(batch_id).tail<3>() *= lambda;
    losses[batch_id] = pinocchio::log6(rel).toVector().squaredNorm();
  }
  return losses;
}
Eigen::MatrixXd SE3_loss_struct::d_SE3_loss_2() { return grad; }

Eigen::VectorXd SE3_loss_struct::SE3_loss_3(
    const Eigen::MatrixXd &trans, const Eigen::MatrixXd &a1,
    const Eigen::MatrixXd &a2,
    const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & T_star) {
  batch_size = static_cast<int>(trans.rows());
  Eigen::VectorXd losses(batch_size);

  grad_t = Eigen::MatrixXd(batch_size, 3);
  grad_a1 = Eigen::MatrixXd(batch_size, 3);
  grad_a2 = Eigen::MatrixXd(batch_size, 3);

  const double eps = 1e-6;

  auto make_rotation = [](const Eigen::Vector3d &v1,
                          const Eigen::Vector3d &v2) {
    Eigen::Vector3d e1 = v1.normalized();
    Eigen::Vector3d tmp = v2 - e1.dot(v2) * e1;
    Eigen::Vector3d e2 = tmp.normalized();
    Eigen::Vector3d e3 = e1.cross(e2);
    Eigen::Matrix3d R;
    R.col(0) = e1;
    R.col(1) = e2;
    R.col(2) = e3;
    return R;
  };
  trans_error.resize(batch_size);
  rot_error.resize(batch_size);
  for (int b = 0; b < batch_size; ++b) {
    Eigen::Vector3d t = trans.row(b);
    Eigen::Vector3d v1 = a1.row(b);
    Eigen::Vector3d v2 = a2.row(b);
    Eigen::Matrix3d R = make_rotation(v1, v2);

    pinocchio::SE3 T_pred(R, t);

    pinocchio::SE3 rel = T_star[b].actInv(T_pred);
    Eigen::VectorXd xi = pinocchio::log6(rel).toVector();
    losses[b] = xi.squaredNorm();
    trans_error[b] = rel.translation().norm();
    rot_error[b] = xi.tail(3).norm();

    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d t_plus = t;
      Eigen::Vector3d t_minus = t;
      t_plus[i] += eps;
      t_minus[i] -= eps;

      pinocchio::SE3 Tp_plus(R, t_plus);
      pinocchio::SE3 Tp_minus(R, t_minus);

      double Lp =
          pinocchio::log6(T_star[b].actInv(Tp_plus)).toVector().squaredNorm();
      double Lm =
          pinocchio::log6(T_star[b].actInv(Tp_minus)).toVector().squaredNorm();
      grad_t(b, i) = (Lp - Lm) / (2.0 * eps);
    }

    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d v1_plus = v1;
      v1_plus[i] += eps;
      Eigen::Vector3d v1_minus = v1;
      v1_minus[i] -= eps;

      Eigen::Matrix3d R_plus = make_rotation(v1_plus, v2);
      Eigen::Matrix3d R_minus = make_rotation(v1_minus, v2);

      pinocchio::SE3 Tp_plus(R_plus, t);
      pinocchio::SE3 Tp_minus(R_minus, t);

      double Lp =
          pinocchio::log6(T_star[b].actInv(Tp_plus)).toVector().squaredNorm();
      double Lm =
          pinocchio::log6(T_star[b].actInv(Tp_minus)).toVector().squaredNorm();
      grad_a1(b, i) = (Lp - Lm) / (2.0 * eps);
    }

    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3d v2_plus = v2;
      v2_plus[i] += eps;
      Eigen::Vector3d v2_minus = v2;
      v2_minus[i] -= eps;

      Eigen::Matrix3d R_plus = make_rotation(v1, v2_plus);
      Eigen::Matrix3d R_minus = make_rotation(v1, v2_minus);

      pinocchio::SE3 Tp_plus(R_plus, t);
      pinocchio::SE3 Tp_minus(R_minus, t);

      double Lp =
          pinocchio::log6(T_star[b].actInv(Tp_plus)).toVector().squaredNorm();
      double Lm =
          pinocchio::log6(T_star[b].actInv(Tp_minus)).toVector().squaredNorm();
      grad_a2(b, i) = (Lp - Lm) / (2.0 * eps);
    }
  }

  return losses;
}

Eigen::MatrixXd SE3_loss_struct::d_t_loss() { return grad_t; }
Eigen::MatrixXd SE3_loss_struct::d_a1_loss() { return grad_a1; }
Eigen::MatrixXd SE3_loss_struct::d_a2_loss() { return grad_a2; }