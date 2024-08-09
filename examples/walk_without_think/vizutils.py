'''
https://github.com/MeMory-of-MOtion/summer-school/blob/master/tutorials/pinocchio/vizutils.py
'''

import meshcat
import numpy as np
import pinocchio as pin

# Meshcat utils

def meshcat_material(r, g, b, a):
    import meshcat

    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))


# Gepetto/meshcat abstraction

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addBox(name, sizex, sizey, sizez, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def addViewerSphere(viz, name, size, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Sphere(size),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addSphere(name, size, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.applyConfiguration(name, xyzquat)
        viz.viewer.gui.refresh()
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)

def visualizeConstraints(viz, model, data, constraint_models, q=None):
    if q is not None:
        pin.framesForwardKinematics(model, data, q)
        viz.display(q)
    for i, c in enumerate(constraint_models):
        if c.name != '':
            name = c.name
        else:
            name = f"c{i}"
        offset = pin.SE3.Identity()
        offset.translation = np.array([0, 0, 0.005])
        box = addViewerBox(viz, "Constraints/"+name+"_1", 0.03, 0.02, 0.01, [1, 0, 0, 0.5])
        applyViewerConfiguration(viz, "Constraints/"+name+"_1", pin.SE3ToXYZQUATtuple(data.oMi[c.joint1_id]*c.joint1_placement.act(offset)))
        box = addViewerBox(viz, "Constraints/"+name+"_2", 0.03, 0.02, 0.01, [0, 1, 0, 0.5])
        applyViewerConfiguration(viz, "Constraints/"+name+"_2", pin.SE3ToXYZQUATtuple(data.oMi[c.joint2_id]*c.joint2_placement.act(offset)))

def addFrames(viz, frames_list, axis_length=0.5, axis_width=5):
    viz.displayFrames(True, frames_list, axis_length=axis_length, axis_width=axis_width)

def visualizeJointsFrames(viz, model, axis_length=0.5, axis_width=5):
    viz_frames = []
    for idJoint, n in enumerate(model.names.tolist()[1:]):
        f = pin.Frame(n, idJoint, pin.SE3.Identity(), pin.OP_FRAME)
        fId = model.addFrame(f)
        viz_frames.append(fId)
    addFrames(viz, viz_frames, axis_length, axis_width)

def visualizeInertias(viz, model, q, alpha=0.1):
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    for idJoint in range(model.njoints):
        I = model.inertias[idJoint]
        oMi = data.oMi[idJoint]
        iMf = pin.SE3.Identity()
        iMf.translation = I.lever
        oMf = oMi.act(iMf)

        size = alpha * np.log(1+I.mass)
        addViewerSphere(viz, f"Inertias/I_{model.names[idJoint]}", size, [1, 0, 0, 0.5])
        applyViewerConfiguration(viz, f"Inertias/I_{model.names[idJoint]}", pin.SE3ToXYZQUAT(oMf))
    
