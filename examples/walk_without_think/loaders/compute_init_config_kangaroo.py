import pinocchio as pin
import numpy as np

np.set_printoptions(precision=6, linewidth=350, suppress=True, threshold=1e6)

try:
    from example_parallel_robots import load
    from example_parallel_robots.loader_tools import completeRobotLoader
    from example_parallel_robots.freeze_joints import freezeJoints
    from toolbox_parallel_robots.projections import configurationProjection
except ImportError:
    print(
        "Please install the `toolbox_parallel_robots` and `example_parallel_robots` packages to run this model"
    )
(
    model,
    robot_constraint_models,
    actuation_model,
    visual_model,
    collision_model,
) = load("kangaroo_2legs", closed_loop=True, free_flyer=True)

import meshcat
from pinocchio.visualize import MeshcatVisualizer

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
model.frames[96].name = "foot_frame_left"
model.frames[244].name = "foot_frame_right"

baseId = model.getFrameId("hanche")
contactIds = [model.getFrameId("foot_frame_right"), model.getFrameId("foot_frame_left")]

data = model.createData()
contact_constraints_models = []
for cId in contactIds:
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    floorContactPositionLeft = data.oMf[cId].translation
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

## Define base constraint for mounting
MBasePlacement = pin.SE3.Identity()
MBasePlacement.translation = np.array([0.0, 0, 0.60])
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
robot_constraint_datas = [cm.createData() for cm in robot_constraint_models]
base_cstr_data = base_cstr_model.createData()

from vizutils import visualizeConstraints
visualizeConstraints(viz, model, data, contact_constraints_models, pin.neutral(model))
# visualizeConstraints(viz, model, data, contact_constraints_models + robot_constraint_models + [base_cstr_model], pin.neutral(model))

# * Get the projected configuration and corresponding controls
q0 = configurationProjection(
    model,
    data,
    contact_constraints_models + robot_constraint_models + [base_cstr_model],
    contact_constraints_datas + robot_constraint_datas + [base_cstr_data],
    q_prec=pin.neutral(model),
)

viz.display(q0) 
print(q0)