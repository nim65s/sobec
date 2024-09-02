import pinocchio as pin
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401

np.set_printoptions(precision=6, linewidth=350, suppress=True,threshold=1e6)

try:
      import example_parallel_robots as epr
      from toolbox_parallel_robots.freeze_joints import freezeJoints
      from toolbox_parallel_robots.projections import configurationProjection
except ImportError as e:
      print(e)
      print(
      "Please install the `toolbox_parallel_robots` and `example_parallel_robots` packages to run this model"
      )
(
      model,
      robot_constraint_models,
      actuation_model,
      visual_model,
      collision_model,
) = epr.load("digit_2legs", free_flyer=True)

joints_lock_names = [
      # Right
      # "motor_knee",
      "spherical_free_foot1_X",
      "spherical_free_foot1_Y",
      "spherical_free_foot2_X",
      "spherical_free_foot2_Y",
      "motor_shin1",
      "spherical_free_shin2",
      "motor_shin2",
      "spherical_free_shin1",
      "spherical_free_hip",
      "spherical_free_knee_X",
      "spherical_free_knee_Y",
      # Left
      # "motor_knee_left",
      "spherical_free_foot1_left_X",
      "spherical_free_foot1_left_Y",
      "spherical_free_foot2_left_X",
      "spherical_free_foot2_left_Y",
      "motor_shin1_left",
      "spherical_free_shin2_left",
      "motor_shin2_left",
      "spherical_free_shin1_left",
      "spherical_free_hip_left",
      "spherical_free_knee_left_X",
      "spherical_free_knee_left_Y",
]
jointToLockIds = [i for (i, n) in enumerate(model.names) if n in joints_lock_names]
(
      model,
      _,  # Constraints models and actuation models
      _,  # are changed ad hoc for Digit
      visual_model,
      collision_model,
) = freezeJoints(
      model,
      robot_constraint_models,
      actuation_model,
      visual_model,
      collision_model,
      jointToLockIds,
      pin.neutral(model),
)

data=model.createData()
q_ref = pin.neutral(model)

import meshcat
from pinocchio.visualize import MeshcatVisualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
server = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.initViewer(loadModel=True, viewer=server)

def generateContactModels(model, contactIds, q_ref):
      data = model.createData()
      contact_constraints_models = []
      for cId in contactIds:
            pin.framesForwardKinematics(model, data, q_ref)
            floorContactPositionLeft = data.oMf[cId].translation
            floorContactPositionLeft[0] = 0
            floorContactPositionLeft[2] = 0
            MContactPlacement = pin.SE3(
            pin.utils.rotate("x", 0.0), floorContactPositionLeft
            )  # SE3 position of the contact
            footFloorConstraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            model,
            model.frames[cId].parentJoint,
            model.frames[cId].placement,
            0,  # To the world
            MContactPlacement,
            pin.ReferenceFrame.LOCAL,
            )
            contact_constraints_models.append(footFloorConstraint)
      return(contact_constraints_models)
contactIds = [
      i for i, f in enumerate(model.frames) if "foot" in f.name and "frame" in f.name
]
contact_constraints_models = generateContactModels(model, contactIds, q_ref)
baseId = model.getFrameId("torso")
q_ref = pin.neutral(model)

MBasePlacement = pin.SE3.Identity()
MBasePlacement.translation = np.array([0, 0, 0.65])
base_cstr_model = pin.RigidConstraintModel(
      pin.ContactType.CONTACT_6D,
      model,
      model.frames[baseId].parentJoint,
      model.frames[baseId].placement,
      0,  # To the world
      MBasePlacement,
      pin.ReferenceFrame.LOCAL,
      )
contact_constraints_datas = [cm.createData() for cm in contact_constraints_models]
base_cstr_data = base_cstr_model.createData()

q0 = configurationProjection(
      model,
      data,
      contact_constraints_models + [base_cstr_model],
      contact_constraints_datas + [base_cstr_data],
      q_prec=q_ref,
)
viz.display(q0)
print(q0)