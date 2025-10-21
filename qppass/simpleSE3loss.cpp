

#include "simpleSE3loss.hpp"
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
    Eigen::Matrix<double, 6, 6> Jlog;
    Eigen::Matrix<double, 6, 6> Jexp;
    Jlog.setZero();
    Jexp.setZero();
    pinocchio::Jexp6(pinocchio::Motion(updated.row(batch_id)), Jexp);
    pinocchio::Jlog6(frozen_.actInv(updated_), Jlog);
    grad.row(batch_id) =
        2 * pinocchio::log6(frozen_.actInv(updated_)).toVector().transpose() *
        Jlog * Jexp;
    losses[batch_id] =
        pinocchio::log6(frozen_.actInv(updated_)).toVector().squaredNorm();
  }
  return losses;
}
Eigen::MatrixXd SE3_loss_struct::d_SE3_loss() { return grad; }