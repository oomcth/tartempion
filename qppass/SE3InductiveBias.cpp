#include "SE3InductiveBias.hpp"

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
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

Eigen::MatrixXd SE3InductiveBias::compute_T_target(
    const Eigen::MatrixXd &batched_q,
    const PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::SE3) & start_positions) {
  int batch_size = static_cast<int>(batched_q.rows());

  Eigen::Matrix<double, 6, 6> Jexp(6, 6);
  Eigen::Matrix<double, 6, 6> Jlog(6, 6);
  Eigen::Matrix<double, 6, 6> Adj(6, 6);
  if (target_placement.rows() != batch_size) {
    grad_propagation.resize(batch_size);
    target_placement = Eigen::Matrix<double, Eigen::Dynamic, 6>(batch_size, 6);
  }
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    Jlog.setZero();
    Jexp.setZero();
    Adj.setZero();
    pinocchio::Jexp6(pinocchio::Motion(batched_q.row(batch_id)), Jexp);
    pinocchio::Jlog6(
        pinocchio::exp6(pinocchio::Motion(batched_q.row(batch_id))) *
            start_positions[batch_id],
        Jlog);
    Adj = start_positions[batch_id].toActionMatrixInverse();
    grad_propagation[batch_id] = Jlog * Adj * Jexp;
    target_placement.row(batch_id) =
        pinocchio::log6(
            pinocchio::exp6(pinocchio::Motion(batched_q.row(batch_id))) *
            start_positions[batch_id])
            .toVector()
            .transpose();
  }
  return target_placement;
}

Eigen::MatrixXd
SE3InductiveBias::d_compute_T_target(const Eigen::MatrixXd &batched_grads) {
  int batch_size = static_cast<int>(target_placement.rows());
  grad_out = Eigen::Matrix<double, Eigen::Dynamic, 6>::Zero(batch_size, 6);

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    grad_out.row(batch_id) =
        batched_grads.row(batch_id) * grad_propagation[batch_id];
  }
  return grad_out;
}