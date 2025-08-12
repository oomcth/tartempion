#include "fonction.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <random>
#include <thread>
#include <typeinfo>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

void printTensor5D(const Eigen::Tensor<double, 5, Eigen::RowMajor> &tensor) {
  // Récupérer les dimensions du tenseur
  const auto &dims = tensor.dimensions();
  if (dims.size() != 5) {
    std::cerr << "Erreur : Le tenseur n'est pas 5D !" << std::endl;
    return;
  }

  int d0 = dims[0];
  int d1 = dims[1];
  int d2 = dims[2];
  int d3 = dims[3];
  int d4 = dims[4];

  // Vérifier si le tenseur est vide
  if (d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0 || d4 == 0) {
    std::cout << "Tenseur 5D vide : dimensions [" << d0 << ", " << d1 << ", "
              << d2 << ", " << d3 << ", " << d4 << "]" << std::endl;
    return;
  }

  // Configurer la précision pour l'affichage des nombres
  std::cout << std::fixed << std::setprecision(4);

  // Parcourir les dimensions
  for (int i0 = 0; i0 < d0; ++i0) {
    std::cout << "Tensor[:, :, :, :, :] at i0 = " << i0 << ":" << std::endl;
    for (int i1 = 0; i1 < d1; ++i1) {
      std::cout << "  Tensor[:, :, :, :] at i1 = " << i1 << ":" << std::endl;
      for (int i2 = 0; i2 < d2; ++i2) {
        std::cout << "    Tensor[:, :, :] at i2 = " << i2 << ":" << std::endl;
        // Afficher la matrice 2D pour les dimensions i3 et i4
        std::cout << "      [" << std::endl;
        for (int i3 = 0; i3 < d3; ++i3) {
          std::cout << "        ";
          for (int i4 = 0; i4 < d4; ++i4) {
            std::cout << std::setw(10) << tensor(i0, i1, i2, i3, i4);
            if (i4 < d4 - 1)
              std::cout << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << "      ]" << std::endl;
        if (i2 < d2 - 1)
          std::cout << "      ----" << std::endl;
      }
      if (i1 < d1 - 1)
        std::cout << "    ====" << std::endl;
    }
    if (i0 < d0 - 1)
      std::cout << "  ****" << std::endl;
  }
}

void print_tensor3d(const Eigen::Tensor<double, 3, Eigen::RowMajor> &tensor) {
  for (int i = 0; i < tensor.dimension(0); ++i) {
    std::cout << "Slice [" << i << "]:\n";
    for (int j = 0; j < tensor.dimension(1); ++j) {
      std::cout << "  Row " << j << ": ";
      for (int k = 0; k < tensor.dimension(2); ++k) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                  << tensor(i, j, k) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

void print_tensor4d(const Eigen::Tensor<double, 4, Eigen::RowMajor> &tensor) {
  for (int i = 0; i < tensor.dimension(0); ++i) {
    std::cout << "Block [" << i << "]:" << std::endl;
    for (int j = 0; j < tensor.dimension(1); ++j) {
      std::cout << "  Slice [" << j << "]:" << std::endl;
      for (int k = 0; k < tensor.dimension(2); ++k) {
        std::cout << "    Row " << k << ": ";
        for (int l = 0; l < tensor.dimension(3); ++l) {
          std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                    << tensor(i, j, k, l) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void print_tensor3d_col(
    const Eigen::Tensor<double, 3, Eigen::ColMajor> &tensor) {
  for (int i = 0; i < tensor.dimension(0); ++i) {
    std::cout << "Slice [" << i << "]:\n";
    for (int j = 0; j < tensor.dimension(1); ++j) {
      std::cout << "  Row " << j << ": ";
      for (int k = 0; k < tensor.dimension(2); ++k) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(7)
                  << tensor(i, j, k) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

Eigen::Map<const Eigen::VectorXd>
extract_slice_map_row(const Eigen::Tensor<double, 3, Eigen::RowMajor> &tenseur,
                      int b, int i) {
  int seqlen = tenseur.dimension(1);
  int dof = tenseur.dimension(2);
  const double *ptr = &tenseur.data()[b * seqlen * dof + i * dof];
  return Eigen::Map<const Eigen::VectorXd>(ptr, dof);
}

void KinematicsWorkspace::save_speed(
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds) {
  speeds_ = speeds;
}

void KinematicsWorkspace::GetPositions() {
  std::cout << positions << std::endl;
}

std::vector<Eigen::VectorXd> KinematicsWorkspace::get_last_q() {
  return last_q;
};

void KinematicsWorkspace::maybe_allocate(const int batch_size, const int dof,
                                         const int num_threads,
                                         const int num_timesteps,
                                         const pinocchio::Model &model) {
  const int total_jacobians = batch_size * num_timesteps;
  last_q.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    last_q[i] = Eigen::VectorXd::Zero(dof);
  }
  if (true ||
      ((int)jacobians_.size() != total_jacobians || dof != allocated_dof_)) {
    contrib = Eigen::Tensor<double, 5, Eigen::RowMajor>(
        batch_size, num_timesteps, dof, num_timesteps, 6);
    contrib.setZero();
    result_buffer.resize(batch_size, num_timesteps, dof);
    result_buffer.setZero();
    localPosition.resize(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      localPosition[i] = Eigen::VectorXd::Zero(dof);
    }
    result = Eigen::Tensor<double, 3, Eigen::RowMajor>(batch_size,
                                                       num_timesteps, dof);
    result.setZero();
    jacobians_.clear();
    jacobians_.reserve(total_jacobians);
    for (int i = 0; i < total_jacobians; ++i) {
      jacobians_.emplace_back(6, dof);
      jacobians_[i].setZero();
    }
    allocated_dof_ = dof;
  }

  if (true || ((int)data_vec_.size() != num_threads)) {
    dJdvq_vec.clear();
    dJdaq_vec.clear();
    dtdJdvq_vec.clear();
    acc_vec.clear();

    dJdvq_vec.reserve(num_threads);
    dJdaq_vec.reserve(num_threads);
    dtdJdvq_vec.reserve(num_threads);
    acc_vec.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
      // data_vec_.emplace_back(model);
      dJdvq_vec.emplace_back(6, model.nv);
      dJdvq_vec[i].setZero();
      dJdaq_vec.emplace_back(6, model.nv);
      dJdaq_vec[i].setZero();
      dtdJdvq_vec.emplace_back(6, model.nv);
      dtdJdvq_vec[i].setZero();
      acc_vec.emplace_back(Eigen::VectorXd::Zero(dof));
      acc_vec[i].setZero();
    }
    data_vec_.clear();
    data_vec_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      data_vec_.emplace_back(pinocchio::Data(model));
    }
    allocated_num_threads_ = num_threads;
  }

  if (true || (derivatives_.dimension(0) != batch_size ||
               derivatives_.dimension(1) != num_timesteps ||
               derivatives_.dimension(2) != 6)) {
    derivatives_ =
        Eigen::Tensor<double, 3, Eigen::RowMajor>(batch_size, num_timesteps, 6);
    derivatives_.setZero();
    positions = Eigen::Tensor<double, 3, Eigen::RowMajor>(batch_size,
                                                          num_timesteps, dof);
    positions.setZero();
    speeds_ = Eigen::Tensor<double, 3, Eigen::RowMajor>(batch_size,
                                                        num_timesteps, dof);
    speeds_.setZero();
    allocated_num_timesteps_ = num_timesteps;
    allocated_batch_size_ = batch_size;
    allocated_num_timesteps_ = num_timesteps;
  }
}

void KinematicsWorkspace::GetJacobians() {
  for (size_t i = 0; i < jacobians_.size(); ++i) {
    std::cout << "Jacobian " << i << ":\n" << jacobians_[i] << "\n";
  }
}
void KinematicsWorkspace::GetSpeeds() {
  std::cout << "speeds : " << speeds_ << std::endl;
}
void KinematicsWorkspace::Getdjdqq() {
  for (size_t i = 0; i < jacobians_.size(); ++i) {
    std::cout << "dj " << i << ":\n" << dJdvq_vec[i] << "\n";
  }
}

Eigen::Tensor<double, 3, Eigen::RowMajor>
ComputeForwardKinematics( // Unit test passed
    const size_t num_threads, const pinocchio::Model &model,
    const pinocchio::Data &template_data,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds,
    Eigen::Ref<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Positions,
    const int tool_id, const double dt, KinematicsWorkspace &workspace) {
  omp_set_num_threads(num_threads);
  const auto &dims = speeds.dimensions();
  const int batch_size = dims[0];
  const int num_timesteps = dims[1];
  const int dof = dims[2];

  workspace.maybe_allocate(batch_size, dof, num_threads, num_timesteps, model);
  workspace.save_speed(speeds);
  auto &jacobians_vec = workspace.jacobians_;
  auto &data_vec = workspace.data_vec_;
  auto &derivatives = workspace.derivatives_;

  Eigen::internal::set_is_malloc_allowed(true);
  // #pragma omp parallel for schedule(dynamic)
  for (int batch = 0; batch < batch_size; ++batch) {
    const int thread_id = omp_get_thread_num();
    pinocchio::Data &data_ref = data_vec[thread_id];
    for (int timestep = 0; timestep < num_timesteps; ++timestep) {
      Eigen::Matrix<double, 6, Eigen::Dynamic> &jacobian_ref =
          jacobians_vec[batch * num_timesteps + timestep];
      for (int i = 0; i < dof; ++i) {
        workspace.localPosition[thread_id](i) = Positions(batch, i);
      }
      pinocchio::framesForwardKinematics(model, data_ref,
                                         workspace.localPosition[thread_id]);
      pinocchio::computeFrameJacobian(model, data_ref,
                                      workspace.localPosition[thread_id],
                                      tool_id, pinocchio::WORLD, jacobian_ref);

      const double *speeds_ptr =
          speeds.data() + (batch * num_timesteps + timestep) * dof;
      double *result_ptr =
          derivatives.data() + (batch * num_timesteps + timestep) * 6;

      Eigen::Map<const Eigen::VectorXd> speeds_vec(speeds_ptr, dof);
      Eigen::Map<Eigen::Matrix<double, 6, 1>> result_vec(result_ptr);
      // regarder si juste faire forward kine marche mieux pour pas faire le
      // matmul

      result_vec.noalias() = jacobian_ref * speeds_vec;
      std::memcpy(&workspace.positions(batch, timestep, 0),
                  &Positions(batch, 0), dof * sizeof(double));
      Positions.row(batch).noalias() += dt * speeds_vec;
      if (timestep == num_timesteps - 1) {
        workspace.last_q[batch] = Positions.row(batch);
        // std::cout << "last q kine" << Positions.row(batch) << std::endl;
        // std::cout << "first Jacobian" << jacobians_vec[batch * num_timesteps]
        //           << std::endl;
      }
    }
  }
  Eigen::internal::set_is_malloc_allowed(true);
  return derivatives;
}

Eigen::Tensor<double, 3, Eigen::RowMajor> ComputeForwardKinematicsDerivatives(
    const size_t num_threads, const pinocchio::Model &model,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
    const int tool_id, const double dt, KinematicsWorkspace &workspace) {
  using namespace Eigen;
  using namespace pinocchio;

  omp_set_num_threads(num_threads);

  const auto &dims = grad_output.dimensions();
  const int batch_size = dims[0];
  const int seq_len = dims[1];
  const int dof = model.nq;

  //   workspace.maybe_allocate(batch_size, dof, num_threads, seq_len, model);
  auto &data_vec = workspace.data_vec_;
  Eigen::Tensor<double, 5, Eigen::RowMajor> &contrib = workspace.contrib;
  contrib.setZero();

  Eigen::internal::set_is_malloc_allowed(false);
  // #pragma omp parallel for schedule(static)
  for (int batch = 0; batch < batch_size; batch++) {
    int thread_id = omp_get_thread_num();
    Data &data_ref = workspace.data_vec_[thread_id];
    for (int time = 0; time < seq_len; time++) {
      Eigen::Matrix<double, 6, Eigen::Dynamic> &dJdqv =
          workspace.dJdvq_vec[thread_id];
      Eigen::Matrix<double, 6, Eigen::Dynamic> &dJdqa =
          workspace.dJdaq_vec[thread_id];

      Eigen::Map<const Eigen::VectorXd> q =
          extract_slice_map_row(workspace.positions, batch, time);
      Eigen::Map<const Eigen::VectorXd> v =
          extract_slice_map_row(workspace.speeds_, batch, time);
      computeForwardKinematicsDerivatives(model, data_ref, q, v, v);
      getFrameVelocityDerivatives(model, data_ref, tool_id, WORLD, dJdqv,
                                  dJdqa);
      for (int j = 0; j < time; ++j) {
        for (int col = 0; col < dJdqv.cols(); ++col) {
          for (int row = 0; row < dJdqv.rows(); ++row) {
            contrib(batch, j, col, time, row) += dt * dJdqv(row, col);
          }
        }
      }
      Eigen::Matrix<double, 6, Eigen::Dynamic> &J_ref =
          workspace.jacobians_[batch * seq_len + time];
      for (int col = 0; col < J_ref.cols(); ++col) {
        for (int row = 0; row < J_ref.rows(); ++row) {
          contrib(batch, time, col, time, row) = J_ref(row, col);
        }
      }
    }
  }
  Eigen::internal::set_is_malloc_allowed(true);
  Eigen::Tensor<double, 3, Eigen::RowMajor> &result = workspace.result_buffer;
  result.setZero();

  // #pragma omp parallel for collapse(1)
  for (int b = 0; b < batch_size; ++b) {
    for (int k = 0; k < seq_len; ++k) {
      for (int l = 0; l < dof; ++l) {
        double sum = 0.0;
        // #pragma omp simd reduction(+ : sum)
        for (int i = 0; i < seq_len; ++i) {
          for (int j = 0; j < 6; ++j) {
            sum += grad_output(b, i, j) * contrib(b, k, l, i, j);
          }
        }
        result(b, k, l) = sum;
      }
    }
  }
  return result;
}

Eigen::MatrixXd get_final_pos(Eigen::MatrixXd start,
                              Eigen::Tensor<double, 3, Eigen::RowMajor> speeds,
                              double dt, int num_thread) {
  using namespace pinocchio;
  int batch = start.rows();
  auto seq_len = static_cast<int>(speeds.dimension(1));
  omp_set_num_threads(num_thread);
  Eigen::MatrixXd out(batch, 6);

  for (int b = 0; b < batch; ++b) {
    Eigen::Vector3d p = start.row(b).head<3>();
    Eigen::Quaterniond q(start(b, 6), start(b, 3), start(b, 4), start(b, 5));
    SE3 T(q.normalized(), p);

    for (int t = 0; t < seq_len; ++t) {
      Eigen::Matrix<double, 6, 1> twist;
      for (int i = 0; i < 6; ++i)
        twist(i) = speeds(b, t, i);
      Motion m(twist);
      SE3 dT = exp6(m.toVector() * dt);
      T = dT * T;
    }

    Motion final_twist = log6(T);
    out.row(b) = final_twist.toVector();
  }

  return out;
}

Eigen::MatrixXd
backward_wrt_twists(const Eigen::MatrixXd &start,
                    const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds,
                    const Eigen::MatrixXd &dL_dout, double dt, int num_thread) {
  using namespace pinocchio;

  int batch = start.rows();
  int seq_len = static_cast<int>(speeds.dimension(1));
  Eigen::MatrixXd dL_dtwists(batch * seq_len, 6);
  dL_dtwists.setZero();

  omp_set_num_threads(num_thread);

#pragma omp parallel for
  for (int b = 0; b < batch; ++b) {
    // Initial pose
    Eigen::Vector3d p = start.row(b).head<3>();
    Eigen::Quaterniond q(start(b, 6), start(b, 3), start(b, 4),
                         start(b, 5)); // qw, qx, qy, qz
    SE3 T = SE3(q.normalized(), p);

    // Forward pass: store SE3 poses and twists
    std::vector<SE3> T_list;
    std::vector<Motion> twist_list;
    T_list.reserve(seq_len + 1);
    twist_list.reserve(seq_len);
    T_list.push_back(T);

    for (int t = 0; t < seq_len; ++t) {
      Eigen::Matrix<double, 6, 1> xi;
      for (int i = 0; i < 6; ++i)
        xi(i) = speeds(b, t, i);
      twist_list.emplace_back(xi);

      SE3 dT = exp6(xi * dt);
      T = dT * T;
      T_list.push_back(T);
    }

    // Compute initial gradient from loss
    Eigen::Matrix<double, 6, 6> Jlog = Jlog6(T_list.back());
    Eigen::VectorXd current_grad =
        Jlog.transpose() * dL_dout.row(b).transpose();

    // Backward pass without updating current_grad
    for (int t = seq_len - 1; t >= 0; --t) {
      const SE3 &T_current = T_list[t];
      const Motion &xi = twist_list[t];

      Eigen::Matrix<double, 6, 6> Jexp = Jexp6(xi * dt);
      Eigen::Matrix<double, 6, 6> Ad_T_inv =
          T_current.inverse().toActionMatrix();

      Eigen::Matrix<double, 6, 1> dL_dxi =
          dt * Jexp.transpose() * Ad_T_inv.transpose() * current_grad;
      dL_dtwists.row(b * seq_len + t) = dL_dxi;
    }
  }

  return dL_dtwists;
}
