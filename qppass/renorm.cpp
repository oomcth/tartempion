#include "renorm.hpp"
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

Eigen::MatrixXd Normalizer::normalize(const Eigen::MatrixXd &batched_q,
                                      double scale, double min_scale) {
  scale_ = scale;
  min_scale_ = min_scale;
  batched_q_ = batched_q;
  int batch_size = batched_q.rows();
  batch_size_ = batch_size;
  changed.resize(batch_size, false);
  output.resize(batched_q.rows(), 6);
  output.setZero();
  pinocchio::SE3 working_pos;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    working_pos = pinocchio::exp6(pinocchio::Motion(batched_q.row(batch_id)));
    double norm = working_pos.translation().norm();
    if (norm > scale_) {
      changed[batch_id] = true;
      working_pos.translation() = working_pos.translation() * scale_ / norm;
    } else if (norm < min_scale) {
      changed[batch_id] = true;
      working_pos.translation() = working_pos.translation() * min_scale_ / norm;
    }
    output.row(batch_id) = pinocchio::log6(working_pos).toVector();
  }
  return output;
};

Eigen::MatrixXd Normalizer::d_normalize(const Eigen::MatrixXd &batched_grads) {
  Eigen::MatrixXd grad_output(batch_size_, 6);
  grad_output.setZero();
  Eigen::Matrix<double, 6, 6> Jexp;
  Eigen::Matrix<double, 6, 6> Jlog;
  Eigen::Matrix<double, 6, 6> J_rescale;

  pinocchio::SE3 working_pos;
  for (int i = 0; i < batch_size_; ++i) {
    if (changed[i]) {
      pinocchio::Jexp6(pinocchio::Motion(batched_q_.row(i)), Jexp);
      pinocchio::Jlog6(pinocchio::exp6(pinocchio::Motion(output.row(i))), Jlog);
      working_pos = pinocchio::exp6(pinocchio::Motion(batched_q_.row(i)));
      double norm = working_pos.translation().norm();
      Eigen::Vector3d t = working_pos.translation();
      double effective_scale = 0;
      if (norm >
          scale_) { // we need to kill the er component of grad if it is pos.
        effective_scale = scale_;
      } else { // we need to kill the er component of grad if it is neg.
        effective_scale = min_scale_;
      }
      J_rescale.setIdentity();
      J_rescale.block<3, 3>(0, 0) = (1 / norm) *
                                        (Eigen::MatrixXd::Identity(3, 3) -
                                         t * t.transpose() / (norm * norm)) *
                                        effective_scale +
                                    (t / norm) * (t / norm).transpose();
      J_rescale.block<3, 3>(0, 0) = working_pos.rotation().transpose() *
                                    J_rescale.block<3, 3>(0, 0) *
                                    working_pos.rotation();
      grad_output.row(i) = batched_grads.row(i) * Jlog * J_rescale * Jexp;

    } else {
      grad_output.row(i) = batched_grads.row(i);
    }
  }
  return grad_output;
};