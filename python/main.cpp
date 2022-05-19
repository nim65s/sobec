#include <eigenpy/eigenpy.hpp>

#include "sobec/python.hpp"

BOOST_PYTHON_MODULE(sobec_pywrap) {
  sobec::python::exposeResidualCoMVelocity();
  sobec::python::exposeActivationQuadRef();
}