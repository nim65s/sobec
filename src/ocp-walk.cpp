///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022 LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "sobec/ocp-walk.hpp"
// #include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "sobec/contact/contact-force.hpp"
#include "sobec/contact/multiple-contacts.hpp"

namespace sobec {

OCPRobotWrapper::OCPRobotWrapper(boost::shared_ptr<pinocchio::Model> model_,
                                 const std::string& contactKey,
                                 const std::string& referencePosture) {
  this->model = model_;

  // Search contact Ids using key name (eg all frames containing "sole_link")
  for (pinocchio::FrameIndex idx = 0; idx < model->frames.size(); ++idx) {
    if (model->frames[idx].name.find(contactKey) != std::string::npos) {
      std::cout << "Found contact " << idx << std::endl;
      contactIds.push_back(idx);
    }
  }

  // Add tow and heel frames ... TODO
  for (pinocchio::FrameIndex cid : contactIds) {
    towIds[cid] = cid;
    heelIds[cid] = cid;
  }

  //   boost::shared_ptr<pinocchio::Data> data;
  data = boost::make_shared<pinocchio::Data>(*model);

  // Get ref config
  Eigen::VectorXd q0 = model->referenceConfigurations[referencePosture];

  // eval x0
  x0.resize(model->nq + model->nv);
  x0.head(model->nq) = q0;
  x0.tail(model->nv).fill(0);

  // eval COM
  com0 = pinocchio::centerOfMass(*model, *data, q0, false);

  // eval mass
  robotGravityForce =
      pinocchio::computeTotalMass(*model) * model->gravity.linear()[2];
}

boost::shared_ptr<IntegratedActionModelEuler> OCPWalk::buildTerminalModel() {
  return NULL;
}

std::vector<boost::shared_ptr<IntegratedActionModelEuler> >
OCPWalk::buildRunningModel(
    const Eigen::Ref<const MatrixX2d>& contact_pattern,
    const std::vector<std::vector<pinocchio::Force> >& referenceForces) {
  state = boost::make_shared<StateMultibody>(robot->model);
  actuation = boost::make_shared<ActuationModelFloatingBase>(state);

  int N = contact_pattern.cols();
  std::vector<boost::shared_ptr<IntegratedActionModelEuler> > models;
  for (int i = 0; i < N; ++i) {
    // Contacts
    auto contacts =
        boost::make_shared<ContactModelMultiple>(state, actuation->get_nu());
    for (int k = 0; k < robot->contactIds.size();
         ++k) {  // k, cid in enumerate(robot.contactIds):
      if (contact_pattern(k, i) == 0.0) continue;
      int cid = robot->contactIds[k];
      auto contact = boost::make_shared<ContactModel6D>(
          state, cid, pinocchio::SE3::Identity(), actuation->get_nu(),
          params->baumgartGains);
      contacts->addContact(robot->model->frames[cid].name + "_contact",
                           contact);
    }

    // Costs
    auto costs = boost::make_shared<CostModelSum>(state, actuation->get_nu());

    auto xRegResidual = boost::make_shared<ResidualModelState>(
        state, robot->x0, actuation->get_nu());
    auto activation_state = boost::make_shared<ActivationModelWeightedQuad>(
        params->stateImportance * params->stateImportance);
    auto xRegCost = boost::make_shared<CostModelResidual>(
        state, activation_state, xRegResidual);
    costs->addCost("stateReg", xRegCost, params->refStateWeight);

    auto uResidual =
        boost::make_shared<ResidualModelControl>(state, actuation->get_nu());
    auto activation_ctrl = boost::make_shared<ActivationModelWeightedQuad>(
        params->controlImportance * params->controlImportance);
    auto uRegCost = boost::make_shared<CostModelResidual>(
        state, activation_ctrl, uResidual);
    if (params->refTorqueWeight > 0.)
      costs->addCost("ctrlReg", uRegCost, params->refTorqueWeight);

    auto comResidual = boost::make_shared<ResidualModelCoMPosition>(
        state, robot->com0, actuation->get_nu());
    Eigen::Vector3d com_weight_array;
    com_weight_array << 0., 0., 1.;
    auto comAct =
        boost::make_shared<ActivationModelWeightedQuad>(com_weight_array);
    auto comCost =
        boost::make_shared<CostModelResidual>(state, comAct, comResidual);
    if (params->comWeight > 0.)
      costs->addCost("com", comCost, params->comWeight);

    auto comVelResidual = boost::make_shared<ResidualModelCoMVelocity>(
        state, params->vcomRef, actuation->get_nu());
    auto comVelAct =
        boost::make_shared<ActivationModelWeightedQuad>(params->vcomImportance);
    auto comVelCost =
        boost::make_shared<CostModelResidual>(state, comVelAct, comVelResidual);
    costs->addCost("comVelCost", comVelCost, params->vcomWeight);

    // Contact costs
    for (int k = 0; k < robot->contactIds.size();
         ++k)  // k, cid in enumerate(robot.contactIds):
    {
      if (contact_pattern(k, i) == 0.) continue;

      int cid = robot->contactIds[k];

      Eigen::Vector2d w_cop;
      double value = 1.0 / (params->footSize * params->footSize);
      w_cop << value, value;
      auto copResidual = boost::make_shared<ResidualModelCenterOfPressure>(
          state, cid, actuation->get_nu());
      auto copAct = boost::make_shared<ActivationModelWeightedQuad>(w_cop);
      auto copCost =
          boost::make_shared<CostModelResidual>(state, copAct, copResidual);
      costs->addCost(robot->model->frames[cid].name + "_cop", copCost,
                     params->copWeight);

      // # Cone with enormous friction (Assuming the robot will barely ever
      // slide). # p.footSize is the allowed area size, while cone expects the
      // corner # coordinates => x2
      Eigen::Vector2d corners;
      value = params->footSize * 2;
      corners << value, value;
      const auto cone = WrenchCone(Eigen::Matrix3d::Identity(), 1000, corners,
                                   4, true, 1, 10000);
      auto coneCost = boost::make_shared<ResidualModelContactWrenchCone>(
          state, cid, cone, actuation->get_nu());
      VectorXd ub = cone.get_ub();
      for (int j = 0; j < 4; ++j) ub[j] = 1e100;  // ub[:4] = np.inf;
      // # ub[5:] = np.inf  ### DEBUG
      for (int j = ub.size() - 1; j >= ub.size() - 8; --j)
        ub[j] = 1e100;  // ub[-8:] = np.inf;
      auto coneAct = boost::make_shared<ActivationModelQuadraticBarrier>(
          ActivationBounds(cone.get_lb(), ub));
      auto coneCostRes =
          boost::make_shared<CostModelResidual>(state, coneAct, coneCost);
      costs->addCost(robot->model->frames[cid].name + "_cone", coneCostRes,
                     params->conePenaltyWeight);

      // # Penalize the distance to the central axis of the cone ...
      // #  ... using normalization weights depending on the axis.
      // # The weights are squared to match the tuning of the CASADI
      // formulation.
      auto coneAxisResidual = boost::make_shared<ResidualModelContactForce>(
          state, cid, pinocchio::Force::Zero(), 6, actuation->get_nu());
      Eigen::VectorXd w = params->forceImportance * params->forceImportance;
      w[2] = 0.0;
      auto coneAxisAct = boost::make_shared<ActivationModelWeightedQuad>(w);
      auto coneAxisCost = boost::make_shared<CostModelResidual>(
          state, coneAxisAct, coneAxisResidual);
      costs->addCost(robot->model->frames[cid].name + "_coneaxis", coneAxisCost,
                     params->coneAxisWeight);

      // # Follow reference (smooth) contact forces
      auto forceRefResidual = boost::make_shared<ResidualModelContactForce>(
          state, cid, pinocchio::Force(referenceForces[i][k]), 6,
          actuation->get_nu());
      auto forceRefCost =
          boost::make_shared<CostModelResidual>(state, forceRefResidual);
      costs->addCost(robot->model->frames[cid].name + "_forceref", forceRefCost,
                     params->refForceWeight /
                         (robot->robotGravityForce * robot->robotGravityForce));
    }

    // Contact costs

    // Flying foot

    // Action
  }

  return models;
}

}  // namespace sobec
