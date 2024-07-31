import pinocchio as pin
import numpy as np
import sobec

def load_simplified():
    urdffile= "robot.urdf"
    urdfpath = "../model_robot_virgile/model_simplified"
    urdf = pin.RobotWrapper.BuildFromURDF(urdfpath + "/" + urdffile,urdfpath,
                                          root_joint=pin.JointModelFreeFlyer())
    # Optimized with compute_init_config_virgile (pin3) so that:
    # - both foot flat on the ground
    # - COM in between the two feet
    # - torso at 0 orientation
    urdf.q0 = np.array([ 0.085858,  0.000065,  0.570089,  0.      ,  0.      ,  1.      ,  0.      , -0.      , -0.00009 , -0.208644, -0.043389,  0.252034, -0.00009 ,  0.      , -0.00009 ,  0.208644, -0.043389,  0.252034, -0.00009 ])
    urdf.model.referenceConfigurations['half_sitting'] = urdf.q0.copy()
    robot = sobec.wwt.RobotWrapper(urdf.model, contactKey="foot_frame")
    robot.collision_model = urdf.collision_model
    robot.visual_model = urdf.visual_model
    return(robot)