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

#include "fonction.hpp"
#include "qppass/renorm.hpp"
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
// #include "qppass/forward_pass.hpp"
#include "qppass/dik_cols.hpp"
#include "qppass/qp.hpp"

#include "qppass/SE3InductiveBias.hpp"
#include "qppass/simpleSE3loss.hpp"
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

Eigen::MatrixXd q_to_xyz(Eigen::MatrixXd batched_position,
                         pinocchio::Model &model) {
  pinocchio::Data data(model);
  Eigen::MatrixXd output(batched_position.rows(), 3);
  for (int i = 0; i < batched_position.rows(); ++i) {
    Eigen::VectorXd q = batched_position.row(i);
    pinocchio::framesForwardKinematics(model, data, q);
    output.row(i) = data.oMf[15].translation();
  }
  return output;
}

void check() { std::cout << "Tartempion import success" << std::endl; }

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
  // bp::class_<KinematicsWorkspace>("KinematicsWorkspace", bp::init<>())
  //     .def_readwrite("save_speed", &KinematicsWorkspace::save_speed)
  //     .def_readwrite("maybe_allocate", &KinematicsWorkspace::maybe_allocate)
  //     .def("GetJacobians", &KinematicsWorkspace::GetJacobians)
  //     .def("GetSpeeds", &KinematicsWorkspace::GetSpeeds)
  //     .def("GetPositions", &KinematicsWorkspace::GetPositions)
  //     .def("last_q", &KinematicsWorkspace::get_last_q)
  //     .def("GetdJ", &KinematicsWorkspace::Getdjdqq);

  // bp::def("q_to_xyz", &q_to_xyz);
  // bp::def("ComputeForwardKinematics", &ComputeForwardKinematics);
  // bp::def("get_final_pos", &get_final_pos);
  // bp::def("backward_prod_exp", &backward_wrt_twists);
  // bp::def("ComputeForwardKinematicsDerivatives",
  //         &ComputeForwardKinematicsDerivatives);
  bp::def("check", &check);
  // bp::def("check_qp", &test_qp);
  bp::def("backward_pass", &backward_pass2);
  bp::def("forward_pass", &forward_pass2);
  bp::class_<QP_pass_workspace2>("QPworkspace", bp::init<>())
      .def("allocate", &QP_pass_workspace2::allocate)
      .def("grad_p", &QP_pass_workspace2::grad_p)
      .def("last_q", &QP_pass_workspace2::get_last_q)
      .def("get_q", &QP_pass_workspace2::Get_positions_)
      .def("set_L1", &QP_pass_workspace2::set_L1_weight)
      .def("set_rot_w", &QP_pass_workspace2::set_rot_weight)
      .def("set_q_reg", &QP_pass_workspace2::set_q_reg)
      .def("set_lambda", &QP_pass_workspace2::set_lambda)
      .def("set_collisions_strength",
           &QP_pass_workspace2::set_collisions_strength)
      .def("set_collisions_safety_margin",
           &QP_pass_workspace2::set_collisions_safety_margin)
      .def("dldq", &QP_pass_workspace2::dloss_dqf)
      .def("set_tool_id", &QP_pass_workspace2::set_tool_id)
      .def("set_bound", &QP_pass_workspace2::set_bound);
  bp::class_<Normalizer>("Normalizer", bp::init<>())
      .def("normalize", &Normalizer::normalize)
      .def("d_normalize", &Normalizer::d_normalize);
  bp::class_<SE3InductiveBias>("SE3_Inductive_Bias", bp::init<>())
      .def("Inductive_Bias", &SE3InductiveBias::compute_T_target)
      .def("d_Inductive_Bias", &SE3InductiveBias::d_compute_T_target);
  bp::class_<SE3_loss_struct>("SE3_loss_workspace", bp::init<>())
      .def("SE3_loss", &SE3_loss_struct::SE3_loss)
      .def("d_SE3_loss", &SE3_loss_struct::d_SE3_loss);
}
