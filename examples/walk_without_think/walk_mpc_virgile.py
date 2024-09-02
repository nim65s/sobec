import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
import time
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401

# Local imports
import sobec
import sobec.walk_without_think.plotter
import specific_params
from loaders_virgile import load_complete_open, load_complete_closed

# #####################################################################################
# ## TUNING ###########################################################################
# #####################################################################################

# In the code, cost terms with 0 weight are commented for reducing execution cost
# An example of working weight value is then given as comment at the end of the line.
# When setting them to >0, take care to uncomment the corresponding line.
# All these lines are marked with the tag ##0##.

walkParams = specific_params.WalkBattobotParams("closed")
walkParams.saveFile = None

# #####################################################################################
# ### LOAD ROBOT ######################################################################
# #####################################################################################
# robot = load_complete_open()
robot = load_complete_closed()
assert len(walkParams.stateImportance) == robot.model.nv * 2

# #####################################################################################
# ### CONTACT PATTERN #################################################################
# #####################################################################################
try:
    # If possible, the initial state and contact pattern are taken from a file.
    ocpConfig = sobec.wwt.loadProblemConfig()
    contactPattern = ocpConfig["contactPattern"]
    robot.x0 = ocpConfig["x0"]
    stateTerminalTarget = ocpConfig["stateTerminalTarget"]
except (KeyError, FileNotFoundError):
    # When the config file is not found ...
    # Initial config, also used for warm start, both taken from robot wrapper.
    # Contact are specified with the order chosen in <contactIds>.
    cycle = ( [[1, 0]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
              + [[0, 1]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
            )
    contactPattern = (
        []
        + [[1, 1]] * walkParams.Tstart
        + (cycle * 1)
        + [[1, 1]] * walkParams.Tend
        + [[1, 1]]
    )

# #####################################################################################
# ### VIZ #############################################################################
# #####################################################################################
try:
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
except (ImportError, AttributeError):
    print("No viewer")


# #####################################################################################
# ### DDP #############################################################################
# #####################################################################################

ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver="FDDP")
problem = ddp.problem
x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
ddp.setCallbacks(
    [
        croc.CallbackVerbose(),
        croc.CallbackLogger(),
        # miscdisp.CallbackMPCWalk(robot.contactIds)
    ]
)

with open("/tmp/mpc-virgile-repr.ascii", "w") as f:
    f.write(sobec.reprProblem(ddp.problem))
    print("OCP described in /tmp/mpc-virgile-repr.ascii")

ddp.solve(x0s, u0s, 200)

# assert sobec.logs.checkGitRefs(ddp.getCallbacks()[1], "refs/mpc-logs.npy")

# ### MPC #############################################################################
# ### MPC #############################################################################
# ### MPC #############################################################################

mpcparams = sobec.MPCWalkParams()
mpcparams.solver_maxiter = 100
sobec.wwt.config_mpc.configureMPCWalk(mpcparams, walkParams)
mpc = sobec.MPCWalk(mpcparams, ddp.problem)
mpc.initialize(ddp.xs[: walkParams.Tmpc + 1], ddp.us[: walkParams.Tmpc])
# mpc.solver.setCallbacks([ croc.CallbackVerbose() ])
x = robot.x0

hist_x = [x.copy()]
hist_u = [ddp.us[0].copy()]
for t in range(walkParams.Tsimu):
    x = mpc.solver.xs[1]
    mpc.calc(x, t)

    print(
        "{:4d} {} {:4d} reg={:.3} a={:.3} ".format(
            t,
            sobec.wwt.dispocp(mpc.problem, robot.contactIds),
            mpc.solver.iter,
            mpc.solver.x_reg,
            mpc.solver.stepLength,
        )
    )

    hist_x.append(mpc.solver.xs[1].copy())
    hist_u.append(mpc.solver.us[0].copy())

    # if not t % 10:
    viz.display(x[: robot.model.nq])
    # time.sleep(walkParams.DT)

# ### PLOT ######################################################################
# ### PLOT ######################################################################
# ### PLOT ######################################################################

plotter = sobec.wwt.plotter.WalkPlotter(robot.model, robot.contactIds)
plotter.setData(contactPattern, np.array(hist_x), np.array(hist_u), None)

target = problem.terminalModel.differential.costs.costs[
    "stateReg"
].cost.residual.reference

plotter.plotBasis(target)
plotter.plotCom(robot.com0)
plotter.plotFeet()
plotter.plotFootCollision(walkParams.footMinimalDistance)

print("Run ```plt.ion(); plt.show()``` to display the plots.")
plt.ion()
plt.show()
input()

# ### SAVE #####################################################################
# ### SAVE #####################################################################
# ### SAVE #####################################################################

if walkParams.saveFile is not None:
    sobec.wwt.save_traj(np.array(hist_x), filename=walkParams.saveFile)
