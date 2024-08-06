import numpy as np
from numpy.testing import assert_almost_equal
import pinocchio as pin
import crocoddyl
import meshcat
from pinocchio.visualize import MeshcatVisualizer

from loaders_virgile import load_complete_open, load_complete_closed
import sobec
from sobec.walk_without_think.actuation_matrix import ActuationModelMatrix
import specific_params

from tqdm import tqdm

def getClosedWarmstart(ws_file):
    ws = np.load(f"{ws_file}", allow_pickle=True)[()]
    xs_open = np.array(ws["xs"])
    us_open = np.array(ws["us"])

    walkParams = specific_params.WalkBattobotParams()

    cycle = ( [[1, 0]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
              + [[0, 1]] * walkParams.Tsingle
              + [[1, 1]] * walkParams.Tdouble
             )
    contactPattern = (
        []
        + [[1, 1]] * walkParams.Tstart
        + (cycle * 3)
        + [[1, 1]] * walkParams.Tend
        + [[1, 1]]
    )

    robot_open = load_complete_open()
    robot_closed, (SER_Q, SER_V, LOOP_Q, LOOP_V) = load_complete_closed(export_joints_ids=True)

    viz = MeshcatVisualizer(robot_closed.model, robot_closed.collision_model, robot_closed.visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    
    data_closed = robot_closed.model.createData()
    data_open = robot_open.model.createData()

    act_matrix_open = np.eye(robot_open.model.nv, robot_open.model.nv - 6, -6)
    act_matrix_closed = np.zeros((robot_closed.model.nv, len(robot_closed.actuationModel.mot_ids_v)))
    for iu, iv in enumerate(robot_closed.actuationModel.mot_ids_v):
        act_matrix_closed[iv, iu] = 1

    T = np.shape(xs_open)[0]
    assert(T == len(contactPattern))
    nq_open, nv_open = robot_open.model.nq, robot_open.model.nv
    nq_closed, nv_closed = robot_closed.model.nq, robot_closed.model.nv
    nu_open = act_matrix_open.shape[1]
    nu_closed = act_matrix_closed.shape[1]
    assert nv_closed == act_matrix_closed.shape[0]
    qs_open = xs_open[:, :-nv_open]
    vs_open = xs_open[:, -nv_open:]
    q0_closed = robot_closed.x0[:nq_closed]
    
    state = crocoddyl.StateMultibody(robot_closed.model)
    nx = nq_closed + nv_closed
    ndx = 2 * nv_closed
    SER_X = SER_Q + [
        i + nq_closed for i in SER_V
    ]
    LOOP_X = [i for i in range(nq_closed + nv_closed) if i not in SER_X]

    xs_closed = np.array([np.concatenate((q0_closed, np.zeros(nv_closed))) for t in range(xs_open.shape[0])])
    xs_closed[:, SER_X] = xs_open
    us_closed = np.empty((xs_closed.shape[0] - 1, nu_closed))

    ## Time t = 0
    xs_closed[0, :nq_closed] = q0_closed
    viz.display(xs_closed[0, :nq_closed])

    actuation = ActuationModelMatrix(state, nu_closed, act_matrix_closed)
    ## For other times
    weights_state = np.zeros(ndx)
    for i in SER_V[:6]:
        weights_state[i + nv_closed] = 1 # free flyer
    for i in SER_V[6:12]:
        weights_state[i + nv_closed] = 1 # Right leg
    for i in SER_V[12:]:
        weights_state[i + nv_closed] = 1 # Left leg

    for t, pattern in tqdm(enumerate(contactPattern[:-1])):
        contact_stack = crocoddyl.ContactModelMultiple(state, actuation.nu)
        for i, cm in enumerate(robot_closed.loop_constraints_models):
            contact = crocoddyl.ContactModel6DLoop(
                state,
                cm.joint1_id,
                cm.joint1_placement,
                cm.joint2_id,
                cm.joint2_placement,
                pin.LOCAL,
                actuation.nu,
                np.array([0., 0.]),
            )
            contact_stack.addContact(f"contact_robot_{i}", contact)
        for k, cid in enumerate(robot_closed.contactIds):
            if not pattern[k]:
                continue
            contact = crocoddyl.ContactModel6D(
                state, cid, pin.SE3.Identity(), pin.WORLD, actuation.nu, walkParams.baumgartGains
            )
            contact_stack.addContact(robot_closed.model.frames[cid].name + "_contact", contact)
        
        ## No cost
        cost = crocoddyl.CostModelSum(state, actuation.nu)
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state,
            actuation,
            contact_stack,
            cost,
        )
        stm = crocoddyl.IntegratedActionModelEuler(dam, walkParams.DT)

        term_cost = crocoddyl.CostModelSum(state, actuation.nu)

        xref_term_res = crocoddyl.ResidualModelState(state, xs_closed[t+1], actuation.nu)
        xref_term_act = crocoddyl.ActivationModelWeightedQuad(weights_state)
        xref_term_cost = crocoddyl.CostModelResidual(state, xref_term_act, xref_term_res)
        term_cost.addCost("xreg", xref_term_cost, 1.0)

        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state,
            actuation,
            contact_stack.copy(),
            term_cost,
        )
        terminal_model = crocoddyl.IntegratedActionModelEuler(dam, 0.0)

        problem = crocoddyl.ShootingProblem(xs_closed[t], [stm], terminal_model)

        tol = 1e-12
        solver = crocoddyl.SolverFDDP(problem)
        solver.th_stop = tol
        solver.verbose = True
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

        solver.solve(maxiter=1000, init_xs=[xs_closed[t], xs_closed[t+1]], init_us=[us_open[t]])

        x_p = xs_closed[t+1]
        q_p, v_p = x_p[:nq_closed], x_p[nq_closed:]
        x_r = solver.xs[1]
        q_r, v_r = x_r[:nq_closed], x_r[nq_closed:]
        # assert_almost_equal(q_p[key_frames_closed["serial_q"]], q_r[key_frames_closed["serial_q"]], decimal=6)
        # assert_almost_equal(v_p[key_frames_closed["serial_v"]], v_r[key_frames_closed["serial_v"]], decimal=6)

        xs_closed[t+1][LOOP_X] = solver.xs[1][LOOP_X]
        # xs_closed[t+1] = solver.xs[1]
        assert_almost_equal(xs_closed[t], solver.xs[0])
        us_closed[t] = solver.us[0]

        viz.display(xs_closed[t+1, :nq_closed])

    while input("Press q to quit the visualisation") != "q":
        viz.play(xs_closed[:, :nq_closed], walkParams.DT)
    return(xs_closed, us_closed)

if __name__ == "__main__":
    ws_file = "/tmp/walk_virgile_open.npy"
    xs_closed, us_closed = getClosedWarmstart(ws_file)
    if input("Do you want to save the warmstart? (y/n)") == "y":
        save_file = "/tmp/walk_virgile_closed_ws.npy"
        sobec.wwt.save_traj(np.array(xs_closed), np.array(us_closed), filename=save_file)
