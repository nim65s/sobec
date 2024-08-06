import pinocchio as pin
import numpy as np
import sobec


def load_simplified():
    urdffile = "robot.urdf"
    urdfpath = "../model_robot_virgile/model_simplified"
    urdf = pin.RobotWrapper.BuildFromURDF(
        urdfpath + "/" + urdffile, urdfpath, root_joint=pin.JointModelFreeFlyer()
    )
    # Optimized with compute_init_config_virgile (pin3) so that:
    # - both foot flat on the ground
    # - COM in between the two feet
    # - torso at 0 orientation
    urdf.q0 = np.array([ 0.177111, -0.117173,  0.692838,  0.      ,  0.      , -0.      ,  1.      ,  0.      , -0.000377,  1.203784,  0.807122, -0.396662,  0.000377,  0.      , -0.000377, -1.204769, -0.81081 ,  0.39396 ,  0.000377])
    urdf.model.referenceConfigurations["half_sitting"] = urdf.q0.copy()
    robot = sobec.wwt.RobotWrapper(urdf.model, contactKey="foot_frame")
    robot.collision_model = urdf.collision_model
    robot.visual_model = urdf.visual_model
    assert len(robot.contactIds) == 2
    return robot

def load_complete_open():
    try:
        from example_parallel_robots.loader_tools import completeRobotLoader
        from example_parallel_robots.freeze_joints import freezeJoints
        from toolbox_parallel_robots.projections import configurationProjection
    except ImportError:
        print(
            "Please install the `toolbox_parallel_robots` and `example_parallel_robots` packages to run this model"
        )
        return
    urdffile = "robot.urdf"
    yamlfile = "robot.yaml"
    urdfpath = "../model_robot_virgile/model_6d"
    (
        model,
        robot_constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    ) = completeRobotLoader(
        urdfpath, urdffile, yamlfile, freeflyer=True
    )
    missing_inertia = [8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36]
    for i in missing_inertia:
        model.inertias[i].inertia += np.eye(3) * 1e-3

    joints_lock_names = [
        # Right
        "left_spherical_foot_1",
        "left_spherical_foot_2",
        "free_knee_left_Y",
        "free_knee_left_Z",
        "motor_ankle1_left",
        "left_spherical_ankle_1_Y",
        "left_spherical_ankle_1_Z",
        "motor_ankle2_left",
        "left_spherical_ankle_2_Y",
        "left_spherical_ankle_2_Z",
        "motor_knee_left",
        "transmission_knee_left",
        # Right
        "right_spherical_foot_1",
        "right_spherical_foot_2",
        "free_knee_right_Y",
        "free_knee_right_Z",
        "motor_ankle1_right",
        "right_spherical_ankle_1_Y",
        "right_spherical_ankle_1_Z",
        "motor_ankle2_right",
        "right_spherical_ankle_2_Y",
        "right_spherical_ankle_2_Z",
        "motor_knee_right",
        "transmission_knee_right",
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
    robot_constraint_models = []

    model.referenceConfigurations["half_sitting"] = np.array([ 0.177111, -0.117173,  0.692838,  0.      ,  0.      , -0.      ,  1.      ,  0.      , -0.000377,  1.203784,  0.807122, -0.396662,  0.000377,  0.      , -0.000377, -1.204769, -0.81081 ,  0.39396 ,  0.000377])

    model.frames[38].name = "foot_frame_right"
    model.frames[76].name = "foot_frame_left"
    robot = sobec.wwt.RobotWrapper(model, contactKey="foot_frame")
    robot.collision_model = collision_model
    robot.visual_model = visual_model
    assert len(robot.contactIds) == 2
    return robot

def load_complete_closed():
    try:
        from example_parallel_robots.loader_tools import completeRobotLoader
        from example_parallel_robots.freeze_joints import freezeJoints
        from toolbox_parallel_robots.projections import configurationProjection
    except ImportError:
        print(
            "Please install the `toolbox_parallel_robots` and `example_parallel_robots` packages to run this model"
        )
        return
    urdffile = "robot.urdf"
    yamlfile = "robot.yaml"
    urdfpath = "../model_robot_virgile/model_6d"
    (
        model,
        robot_constraint_models,
        actuation_model,
        visual_model,
        collision_model,
    ) = completeRobotLoader(
        urdfpath, urdffile, yamlfile, freeflyer=True
    )

    joints_lock_names = [
        # Right
        "left_spherical_foot_1",
        "left_spherical_foot_2",
        "free_knee_left_Y",
        "free_knee_left_Z",
        "motor_ankle1_left",
        "left_spherical_ankle_1_Y",
        "left_spherical_ankle_1_Z",
        "motor_ankle2_left",
        "left_spherical_ankle_2_Y",
        "left_spherical_ankle_2_Z",
        "motor_knee_left",
        "transmission_knee_left",
        # Right
        "right_spherical_foot_1",
        "right_spherical_foot_2",
        "free_knee_right_Y",
        "free_knee_right_Z",
        "motor_ankle1_right",
        "right_spherical_ankle_1_Y",
        "right_spherical_ankle_1_Z",
        "motor_ankle2_right",
        "right_spherical_ankle_2_Y",
        "right_spherical_ankle_2_Z",
        "motor_knee_right",
        "transmission_knee_right",
    ]

    LOOP_JOINT_IDS_Q = []
    LOOP_JOINT_IDS_V = []
    for i, name in enumerate(joints_lock_names):
        jId = model.getJointId(name)
        for niq in range(model.joints[jId].nq):
            LOOP_JOINT_IDS_Q.append(model.joints[jId].idx_q + niq)
        for niv in range(model.joints[jId].nv):
            LOOP_JOINT_IDS_V.append(model.joints[jId].idx_v + niv)
    SERIAL_JOINT_IDS_Q = [i for i in range(model.nq) if i not in LOOP_JOINT_IDS_Q]
    SERIAL_JOINT_IDS_V = [i for i in range(model.nv) if i not in LOOP_JOINT_IDS_V]
    
    q_ref = pin.neutral(model)
    q_ref[SERIAL_JOINT_IDS_Q] = np.array([ 0.177111, -0.117173,  0.692838,  0.      ,  0.      , -0.      ,  1.      ,  0.      , -0.000377,  1.203784,  0.807122, -0.396662,  0.000377,  0.      , -0.000377, -1.204769, -0.81081 ,  0.39396 ,  0.000377])
    robot_constraint_datas = [cm.createData() for cm in robot_constraint_models]
    w = np.ones(model.nv)
    w[SERIAL_JOINT_IDS_V] = 1e5
    W = np.diag(w)
    q0 = configurationProjection(model, model.createData(), robot_constraint_models, robot_constraint_datas, q_ref, W)

    model.referenceConfigurations["half_sitting"] = q0

    model.frames[14].name = "foot_frame_right"
    model.frames[58].name = "foot_frame_left"
    robot = sobec.wwt.RobotWrapper(model, contactKey="foot_frame", closed_loop=True)
    robot.collision_model = collision_model
    robot.visual_model = visual_model
    robot.actuationModel = actuation_model
    robot.loop_constraints_models = robot_constraint_models
    assert len(robot.contactIds) == 2
    return robot