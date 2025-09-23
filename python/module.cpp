#include <Eigen/Dense>
#include <eigenpy/eigenpy.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>

#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/sample-models.hpp>

#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

#include "fonction.hpp"
// #include "qppass/forward_pass.hpp"
#include "qppass/dik_cols.hpp"
#include "qppass/qp.hpp"

#include <pinocchio/container/aligned-vector.hpp>

namespace bp = boost::python;

using namespace bp;

template <typename VecType>
struct StdVectorPythonVisitor
    : public boost::python::def_visitor<StdVectorPythonVisitor<VecType>> {

  using Container = std::vector<VecType>;

  template <class Class> void visit(Class &cl) const {
    cl.def("__len__", &StdVectorPythonVisitor::len)
        .def("__getitem__", &StdVectorPythonVisitor::get_item,
             return_value_policy<copy_const_reference>())
        .def("__setitem__", &StdVectorPythonVisitor::set_item)
        .def("append", &StdVectorPythonVisitor::append);
  }

  static typename Container::size_type len(Container const &v) {
    return v.size();
  }

  static const VecType &get_item(Container const &v, std::size_t i) {
    if (i >= v.size())
      throw std::out_of_range("Index out of range");
    return v[i];
  }

  static void set_item(Container &v, std::size_t i, const VecType &val) {
    if (i >= v.size())
      throw std::out_of_range("Index out of range");
    v[i] = val;
  }

  static void append(Container &v, const VecType &val) { v.push_back(val); }

  static void expose(const std::string &class_name) {
    class_<Container>(class_name.c_str())
        .def(StdVectorPythonVisitor<VecType>());
  }
};

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

MatrixXd q_to_xyz(MatrixXd batched_position, pinocchio::Model &model) {
  pinocchio::Data data(model);
  Eigen::MatrixXd output(batched_position.rows(), 3);
  for (int i = 0; i < batched_position.rows(); ++i) {
    Eigen::VectorXd q = batched_position.row(i);
    pinocchio::framesForwardKinematics(model, data, q);
    output.row(i) = data.oMf[15].translation();
  }
  return output;
}

Eigen::Tensor<double, 3, Eigen::RowMajor>
Hessian(pinocchio::Model &model, pinocchio::Data &data,
        Eigen::Vector<double, Eigen::Dynamic> q, int tool_id) {
  Eigen::Tensor<double, 3> Hess(6, model.nq, model.nq);
  Hess.setZero();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::computeJointKinematicHessians(model, data, q);
  Hess.setZero();
  pinocchio::getJointKinematicHessian(model, data, tool_id, pinocchio::LOCAL,
                                      Hess);
  Eigen::Tensor<double, 3, Eigen::RowMajor> Hess_row(6, model.nq, model.nq);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < model.nq; ++j) {
      for (int k = 0; k < model.nq; ++k) {
        Hess_row(i, j, k) = Hess(i, j, k);
      }
    }
  }

  return Hess_row;
}

Eigen::MatrixXd GetdJdqv(pinocchio::Model &model, pinocchio::Data &data,
                         Eigen::Vector<double, Eigen::Dynamic> q,
                         Eigen::Vector<double, Eigen::Dynamic> v,
                         Eigen::Vector<double, Eigen::Dynamic> a, int tool_id) {

  int dof = q.size();
  Eigen::MatrixXd dJdqv(6, dof);
  Eigen::MatrixXd dJdqa(6, dof);

  dJdqv.setZero();
  dJdqa.setZero();

  using namespace pinocchio;
  computeForwardKinematicsDerivatives(model, data, q, v, a);
  getJointVelocityDerivatives(model, data, tool_id, LOCAL, dJdqv, dJdqa);

  return dJdqv;
}

Eigen::MatrixXd get_Quat(pinocchio::Model model, pinocchio::Data data,
                         Eigen::MatrixXd &batched_position, int frame_id) {
  auto batch_size = static_cast<int>(batched_position.rows());
  Eigen::MatrixXd quaternions(batch_size, 4);
  quaternions.setZero();
  for (int i = 0; i < batch_size; ++i) {
    pinocchio::framesForwardKinematics(model, data,
                                       batched_position.row(i).transpose());
    pinocchio::updateFramePlacement(model, data, frame_id);
    Eigen::Quaterniond q(data.oMf[frame_id].rotation());
    quaternions.row(i) = q.coeffs().transpose();
  }
  return quaternions;
}

void check() { std::cout << "Tartempion import success" << std::endl; }

Eigen::Tensor<double, 3, Eigen::RowMajor>
RowMajor(Eigen::Tensor<double, 3, Eigen::RowMajor> rowmajor) {
  std::cout << "coord 1, 1, 0 : " << rowmajor(1, 1, 0) << std::endl;
  print_tensor3d(rowmajor);
  return rowmajor;
}
Eigen::Tensor<double, 3, Eigen::ColMajor>
ColMajor(Eigen::Tensor<double, 3, Eigen::ColMajor> colmajor) {
  std::cout << "coord 1, 1, 0 : " << colmajor(1, 1, 0) << std::endl;
  print_tensor3d_col(colmajor);
  return colmajor;
}

Eigen::VectorXd pos_from_se3(SE3 se3) {
  Eigen::Vector<double, 7> pos;
  pos.setZero();
  pos[0] = se3.translation()[0];
  pos[1] = se3.translation()[1];
  pos[2] = se3.translation()[2];
  Eigen::Quaterniond quat(se3.rotation());
  pos[3] = quat.x();
  pos[4] = quat.y();
  pos[5] = quat.z();
  pos[6] = quat.w();

  return pos;
}

Eigen::MatrixXd q_to_pos(pinocchio::Model model, Eigen::MatrixXd q,
                         int frame_id) {
  int batch_size = q.rows();
  Data data;
  data = Data(model);
  Eigen::Matrix<double, Eigen::Dynamic, 7> positions(batch_size, 7);
  positions.setZero();
  for (int i = 0; i < batch_size; ++i) {
    pinocchio::framesForwardKinematics(model, data, q.row(i).transpose());
    pinocchio::updateFramePlacement(model, data, frame_id);
    SE3 position = data.oMf[frame_id];
    positions.row(i) = pos_from_se3(position);
  }
  return positions;
}

BOOST_PYTHON_MODULE(tartempion) {
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<Eigen::Tensor<double, 3, Eigen::RowMajor>>();
  eigenpy::enableEigenPySpecific<Eigen::Tensor<double, 3, Eigen::ColMajor>>();
  eigenpy::enableEigenPySpecific<Eigen::VectorXd>();
  eigenpy::enableEigenPySpecific<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>();
  StdVectorPythonVisitor<Eigen::VectorXd>::expose("StdVecVectorXd");
  StdVectorPythonVisitor<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>>::expose("StdVecMAtrixRowMajor");
  bp::scope().attr("__version__") = "1.0";

  bp::import("pinocchio");
  bp::class_<KinematicsWorkspace>("KinematicsWorkspace", bp::init<>())
      .def_readwrite("save_speed", &KinematicsWorkspace::save_speed)
      .def_readwrite("maybe_allocate", &KinematicsWorkspace::maybe_allocate)
      .def("GetJacobians", &KinematicsWorkspace::GetJacobians)
      .def("GetSpeeds", &KinematicsWorkspace::GetSpeeds)
      .def("GetPositions", &KinematicsWorkspace::GetPositions)
      .def("last_q", &KinematicsWorkspace::get_last_q)
      .def("GetdJ", &KinematicsWorkspace::Getdjdqq);

  bp::def("q_to_xyz", &q_to_xyz);
  bp::def("q_to_pos", &q_to_pos);
  bp::def("batched_quaternions", &get_Quat);
  bp::def("Hess", &Hessian);
  bp::def("dJdqv", &GetdJdqv);
  bp::def("ComputeForwardKinematics", &ComputeForwardKinematics);
  bp::def("get_final_pos", &get_final_pos);
  bp::def("backward_prod_exp", &backward_wrt_twists);
  bp::def("ComputeForwardKinematicsDerivatives",
          &ComputeForwardKinematicsDerivatives);
  bp::def("check", &check);
  bp::def("backward_pass", &backward_pass2);
  bp::def("forward_pass", &forward_pass2);
  bp::def("check_qp", &test_qp);
  bp::class_<QP_pass_workspace2>("QPworkspace", bp::init<>())
      .def("grad_p", &QP_pass_workspace2::grad_p)
      .def("grad_A", &QP_pass_workspace2::grad_A)
      .def("last_q", &QP_pass_workspace2::get_last_q)
      .def("get_q", &QP_pass_workspace2::Get_positions_)
      .def("set_L1", &QP_pass_workspace2::set_L1_weight)
      .def("set_rot_w", &QP_pass_workspace2::set_rot_weight)
      .def("set_q_reg", &QP_pass_workspace2::set_q_reg)
      .def("set_lambda", &QP_pass_workspace2::set_lambda)
      .def("dldq", &QP_pass_workspace2::dloss_dqf)
      .def("set_tool_id", &QP_pass_workspace2::set_tool_id)
      .def("set_bound", &QP_pass_workspace2::set_bound)
      .def("grad_b", &QP_pass_workspace2::grad_b);
}
