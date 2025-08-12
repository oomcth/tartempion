#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <omp.h>
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

void print_tensor3d(const Eigen::Tensor<double, 3, Eigen::RowMajor> &tensor);
void print_tensor3d_col(
    const Eigen::Tensor<double, 3, Eigen::ColMajor> &tensor);
Eigen::Map<const Eigen::VectorXd>
extract_slice_map_row(const Eigen::Tensor<double, 3, Eigen::RowMajor> &tenseur,
                      int b, int i);

struct KinematicsWorkspace {
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> jacobians_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> result_buffer;
  std::vector<pinocchio::Data> data_vec_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> derivatives_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> positions;
  Eigen::Tensor<double, 3, Eigen::RowMajor> speeds_;
  Eigen::Tensor<double, 3, Eigen::RowMajor> result;
  std::vector<Eigen::VectorXd> localPosition;
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> dJdvq_vec;
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> dJdaq_vec;
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> dtdJdvq_vec;
  std::vector<Eigen::VectorXd> acc_vec;
  Eigen::Tensor<double, 5, Eigen::RowMajor> contrib;
  int allocated_batch_size_ = -1;
  int allocated_dof_ = -1;
  int allocated_num_threads_ = -1;
  int allocated_num_timesteps_ = -1;
  std::vector<Eigen::VectorXd> get_last_q();
  std::vector<Eigen::VectorXd> last_q;

  void GetJacobians();
  void GetPositions();
  void GetSpeeds();
  void Getdjdqq();
  void save_speed(const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds);
  void maybe_allocate(const int batch_size, const int dof,
                      const int num_threads, const int num_timesteps,
                      const pinocchio::Model &model);
};

Eigen::Tensor<double, 3, Eigen::RowMajor> ComputeForwardKinematics(
    const size_t num_threads, const pinocchio::Model &model,
    const pinocchio::Data &template_data,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds,
    Eigen::Ref<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Positions,
    const int tool_id, const double dt, KinematicsWorkspace &workspace);

Eigen::Tensor<double, 3, Eigen::RowMajor> ComputeForwardKinematicsDerivatives(
    const size_t num_threads, const pinocchio::Model &model,
    const Eigen::Tensor<double, 3, Eigen::RowMajor> &grad_output,
    const int tool_id, const double dt, KinematicsWorkspace &workspace);

Eigen::MatrixXd get_final_pos(Eigen::MatrixXd start,
                              Eigen::Tensor<double, 3, Eigen::RowMajor> speeds,
                              double dt, int num_thread);

Eigen::MatrixXd
backward_wrt_twists(const Eigen::MatrixXd &start,
                    const Eigen::Tensor<double, 3, Eigen::RowMajor> &speeds,
                    const Eigen::MatrixXd &dL_dout, double dt, int num_thread);