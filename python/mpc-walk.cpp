///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/bindings/python/utils/list.hpp>
#include <pinocchio/multibody/fwd.hpp>

// Must be included first!
#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <eigenpy/eigenpy.hpp>

#include "sobec/fwd.hpp"
#include "sobec/mpc-walk.hpp"

namespace sobec {
namespace python {

using namespace crocoddyl;
namespace bp = boost::python;

void initialize(MPCWalk self, bp::object xs, bp::object us) {
  bp::extract<bp::list> xs_l(xs), us_l(us);
  std::vector<Eigen::VectorXd> xs_v, us_v;
  pinocchio::python::extract(xs_l(), xs_v);
  pinocchio::python::extract(us_l(), us_v);
  self.initialize(xs_v, us_v);
}

void exposeMPCWalk() {
  bp::register_ptr_to_python<boost::shared_ptr<MPCWalk> >();

  bp::class_<MPCWalk>("MPCWalk",
                      bp::init<boost::shared_ptr<ShootingProblem> >(
                          bp::args("self"), "Initialize the MPC (empty init)"))
      .def("calc", &MPCWalk::calc, bp::args("self", "x", "t"))
      .def("initialize", &initialize, bp::args("self", "xs", "us"))
      .add_property("Tmpc", &MPCWalk::get_Tmpc, &MPCWalk::set_Tmpc);
}

}  // namespace python
}  // namespace sobec
