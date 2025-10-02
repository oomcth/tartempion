#include "renorm.hpp"
#include "forward_pass.hpp"
#include "qp.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cassert>
#include <coal/collision.h>
#include <diffcoal/spatial.hpp>
#include <iostream>
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
                                      double scale) {
  scale_ = scale;
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
    }
    output.row(batch_id) = pinocchio::log6(working_pos).toVector();
  }
  return output;
};

Eigen::MatrixXd Normalizer::d_normalize(const Eigen::MatrixXd &batched_grads) {
  Eigen::MatrixXd grad_output(batch_size_, 6);
  grad_output.setZero();
  Eigen::MatrixXd Jexp(6, 6);
  Eigen::MatrixXd Jlog(6, 6);
  Eigen::MatrixXd J_rescale(6, 6);

  pinocchio::SE3 working_pos;
  for (int i = 0; i < batch_size_; ++i) {
    if (changed[i]) {
      pinocchio::Jexp6(pinocchio::Motion(batched_q_.row(i)), Jexp);
      pinocchio::Jlog6(pinocchio::exp6(pinocchio::Motion(output.row(i))), Jlog);
      working_pos = pinocchio::exp6(pinocchio::Motion(batched_q_.row(i)));
      double norm = working_pos.translation().norm();
      Eigen::Vector3d t = working_pos.translation();
      double norm_inv = 1.0 / norm;
      double scale_over_norm = scale_ / norm;
      J_rescale.setIdentity();
      J_rescale.block<3, 3>(0, 0) = (1 / norm) *
                                        (Eigen::MatrixXd::Identity(3, 3) -
                                         t * t.transpose() / (norm * norm)) *
                                        scale_ +
                                    (t / norm) * (t / norm).transpose();
      J_rescale.block<3, 3>(0, 0) = working_pos.rotation().transpose() *
                                    J_rescale.block<3, 3>(0, 0) *
                                    working_pos.rotation();
      //   std::cout << J_rescale << std::endl;
      //   std::cout << Jexp << std::endl;
      //   std::cout << Jlog << std::endl;
      grad_output.row(i) = batched_grads.row(i) * Jlog * J_rescale * Jexp;

    } else {
      grad_output.row(i) = batched_grads.row(i);
    }
  }
  return grad_output;
};