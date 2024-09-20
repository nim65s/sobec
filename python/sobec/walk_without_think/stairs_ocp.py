import sys

import pinocchio as pin
import crocoddyl as croc
import numpy as np


# Local imports
import sobec
from .weight_share import computeReferenceForces


# workaround python 2
if sys.version_info.major < 3:
    FileNotFoundError = IOError


def buildRunningModels(robotWrapper, contactPattern, params, with_constraints=False):
    p = params
    robot = robotWrapper

    referenceForces = computeReferenceForces(
        contactPattern,
        robot.gravForce,
        transitionDuration=params.transitionDuration,
        minimalNormalForce=params.minimalNormalForce,
    )
    models = []

    # #################################################################################
    for t, pattern in enumerate(contactPattern[:-1]):
        # print("time t=%s %s" % (t, pattern))

        # Basics
        state = croc.StateMultibody(robot.model)
        if robot.actuationModel is None:
            actuation = croc.ActuationModelFloatingBase(state)
        else:
            from .actuation_matrix import ActuationModelMatrix
            act_matrix = np.zeros((robot.model.nv, len(robot.actuationModel.mot_ids_v)))
            for iu, iv in enumerate(robot.actuationModel.mot_ids_v):
                act_matrix[iv, iu] = 1
            actuation = ActuationModelMatrix(state, np.shape(act_matrix)[1], act_matrix)

        # Contacts
        contacts = croc.ContactModelMultiple(state, actuation.nu)
        for k, cid in enumerate(robot.contactIds):
            if not pattern[k]:
                continue
            contact = croc.ContactModel6D(
                state, cid, pin.SE3.Identity(), pin.WORLD, actuation.nu, p.baumgartGains
            )
            contacts.addContact(robot.model.frames[cid].name + "_contact", contact)
        for k, cm in enumerate(robot.loop_constraints_models):
            assert cm.type == pin.ContactType.CONTACT_6D and cm.reference_frame == pin.ReferenceFrame.LOCAL
            contact = croc.ContactModel6DLoop(
                state,
                cm.joint1_id,
                cm.joint1_placement,
                cm.joint2_id,
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
                actuation.nu,
                np.array([0., 0.])
            )
            contacts.addContact(f"loop_contact_{k}", contact)

        # Costs and constraints
        costs = croc.CostModelSum(state, actuation.nu)
        constraints = croc.ConstraintModelManager(state, actuation.nu)

        if p.refStateWeight > 0:
            xRegResidual = croc.ResidualModelState(state, robot.x0, actuation.nu)
            xRegCost = croc.CostModelResidual(
                state,
                croc.ActivationModelWeightedQuad(p.stateImportance**2),
                xRegResidual,
            )
            costs.addCost("stateReg", xRegCost, p.refStateWeight)

        if p.refTorqueWeight > 0:
            uResidual = croc.ResidualModelControl(state, actuation.nu)
            uRegCost = croc.CostModelResidual(
                state,
                croc.ActivationModelWeightedQuad(np.array(p.controlImportance**2)),
                uResidual,
            )
            costs.addCost("ctrlReg", uRegCost, p.refTorqueWeight)

        if p.comWeight > 0:
            comResidual = croc.ResidualModelCoMPosition(state, robot.com0, actuation.nu)
            comAct = croc.ActivationModelWeightedQuad(np.array([0, 0, 1]))
            comCost = croc.CostModelResidual(state, comAct, comResidual)
            costs.addCost("com", comCost, p.comWeight)

        if p.vcomWeight > 0:
            comVelResidual = sobec.ResidualModelCoMVelocity(state, p.vcomRef, actuation.nu)
            comVelAct = croc.ActivationModelWeightedQuad(p.vcomImportance)
            comVelCost = croc.CostModelResidual(state, comVelAct, comVelResidual)
            costs.addCost("comVelCost", comVelCost, p.vcomWeight)

        # Contact costs
        for k, cid in enumerate(robot.contactIds):
            if not pattern[k]:
                continue

            if p.copWeight > 0:
                copResidual = sobec.ResidualModelCenterOfPressure(state, cid, actuation.nu)
                copAct = croc.ActivationModelWeightedQuad(
                    np.array([1.0 / p.footSize**2] * 2)
                )
                if with_constraints:
                    copConstraint = croc.ConstraintModelResidual(
                        state,
                        copResidual,
                        np.array([-p.footSize, -p.footSize]),
                        np.array([p.footSize, p.footSize]),
                    )
                    constraints.addConstraint(
                        "%s_cop" % robot.model.frames[cid].name, copConstraint
                    )
                else:
                    copCost = croc.CostModelResidual(state, copAct, copResidual)
                    costs.addCost(
                        "%s_cop" % robot.model.frames[cid].name, copCost, p.copWeight
                    )

            if p.centerOfFrictionWeight > 0:
                cofResidual = sobec.ResidualModelCenterOfPressure(state, cid, actuation.nu)
                cofAct = croc.ActivationModelWeightedQuad(
                    np.array([1.0 / p.footSize**2] * 2)
                )
                cofCost = croc.CostModelResidual(state, cofAct, cofResidual)
                costs.addCost(
                    "%s_centeroffriction" % robot.model.frames[cid].name,
                    cofCost,
                    p.centerOfFrictionWeight,
                )

            # Cone with enormous friction (Assuming the robot will barely ever slide).
            # p.footSize is the allowed area size, while cone expects the corner
            # coordinates => x2
            if p.conePenaltyWeight:
                fmin = p.minimalNormalForce
                wbound = p.footSize * 2 if not p.withNormalForceBoundOnly else 1000.0
                wbound = np.array([wbound] * 2)
                cone = croc.WrenchCone(np.eye(3), 1000, wbound, 4, True, fmin, 10000)
                coneCost = croc.ResidualModelContactWrenchCone(
                    state, cid, cone, actuation.nu
                )
                ub = cone.ub.copy()
                ub[:4] = np.inf
                # ub[5:] = np.inf  ### DEBUG
                ub[-8:] = np.inf
                coneAct = croc.ActivationModelQuadraticBarrier(
                    croc.ActivationBounds(cone.lb, ub)
                )
                coneCost = croc.CostModelResidual(state, coneAct, coneCost)
                costs.addCost(
                    "%s_cone" % robot.model.frames[cid].name,
                    coneCost,
                    p.conePenaltyWeight,
                )

            # Penalize the distance to the central axis of the cone ...
            #  ... using normalization weights depending on the axis.
            # The weights are squared to match the tuning of the CASADI formulation.
            if p.coneAxisWeight > 0:
                coneAxisResidual = croc.ResidualModelContactForce(
                    state, cid, pin.Force.Zero(), 6, actuation.nu
                )
                w = np.array(p.forceImportance**2)
                w[2] = 0
                coneAxisAct = croc.ActivationModelWeightedQuad(w)
                coneAxisCost = croc.CostModelResidual(state, coneAxisAct, coneAxisResidual)
                costs.addCost(
                    "%s_coneaxis" % robot.model.frames[cid].name,
                    coneAxisCost,
                    p.coneAxisWeight,
                )

            # Follow reference (smooth) contact forces
            if p.refForceWeight > 0:
                forceRefResidual = croc.ResidualModelContactForce(
                    state, cid, pin.Force(referenceForces[t][k]), 6, actuation.nu
                )
                forceRefCost = croc.CostModelResidual(state, forceRefResidual)
                costs.addCost(
                    "%s_forceref" % robot.model.frames[cid].name,
                    forceRefCost,
                    p.refForceWeight / robot.gravForce**2,
                )

        # IMPACT
        for k, cid in enumerate(robot.contactIds):
            if t > 0 and not contactPattern[t - 1][k] and pattern[k]:
                # REMEMBER TO divide the weight by p.DT, as impact should be independant
                # of the node duration (at least, that s how weights are tuned in
                # casadi).

                print("Impact %s at time %s" % (cid, t))
                if p.impactAltitudeWeight > 0:
                    impactPosRef = np.array([0, 0, p.groundHeight[t]])
                    impactResidual = croc.ResidualModelFrameTranslation(
                        state, cid, impactPosRef, actuation.nu
                    )
                    if with_constraints:
                        constraints.addConstraint(
                            "%s_altitudeimpact" % robot.model.frames[cid].name,
                            croc.ConstraintModelResidual(
                                state,
                                impactResidual,
                                np.array([-np.inf, -np.inf, -1e-6]),
                                np.array([np.inf, np.inf, 1e-6]),
                            ),
                        )
                    else:
                        impactAct = croc.ActivationModelWeightedQuad(np.array([0, 0, 1]))
                        impactCost = croc.CostModelResidual(state, impactAct, impactResidual)
                        costs.addCost(
                            "%s_altitudeimpact" % robot.model.frames[cid].name,
                            impactCost,
                            p.impactAltitudeWeight / p.DT,
                        )

                if p.impactVelocityWeight > 0:
                    impactVelResidual = croc.ResidualModelFrameVelocity(
                        state,
                        cid,
                        pin.Motion.Zero(),
                        pin.ReferenceFrame.LOCAL,
                        actuation.nu,
                    )
                    if with_constraints:
                        constraints.addConstraint(
                            "%s_velimpact" % robot.model.frames[cid].name,
                            croc.ConstraintModelResidual(
                                state,
                                impactVelResidual,
                                np.array([-1e-6]*6),
                                np.array([1e-6]*6),
                            ),
                        )
                    else:
                        impactVelCost = croc.CostModelResidual(state, impactVelResidual)
                        costs.addCost(
                            "%s_velimpact" % robot.model.frames[cid].name,
                            impactVelCost,
                            p.impactVelocityWeight / p.DT,
                        )

                if p.impactRotationWeight > 0:
                    impactRotResidual = croc.ResidualModelFrameRotation(
                        state, cid, np.eye(3), actuation.nu
                    )
                    if with_constraints:
                        constraints.addConstraint(
                            "%s_rotimpact" % robot.model.frames[cid].name,
                            croc.ConstraintModelResidual(
                                state,
                                impactRotResidual,
                                np.array([-1e-6, -1e-6, -np.inf]),
                                np.array([1e-6, 1e-6, np.inf]),
                            ),
                        )
                    else:
                        impactRotAct = croc.ActivationModelWeightedQuad(np.array([1, 1, 0]))
                        impactRotCost = croc.CostModelResidual(
                            state, impactRotAct, impactRotResidual
                        )
                        costs.addCost(
                            "%s_rotimpact" % robot.model.frames[cid].name,
                            impactRotCost,
                            p.impactRotationWeight / p.DT,
                        )

                if p.refMainJointsAtImpactWeight > 0:
                    impactRefJointsResidual = croc.ResidualModelState(
                        state, robot.x0, actuation.nu
                    )
                    jselec = np.zeros(robot.model.nv * 2)
                    jselec[
                        [
                            robot.model.joints[robot.model.getJointId(name)].idx_v
                            for name in p.mainJointIds
                        ]
                    ] = 1
                    impactRefJointsAct = croc.ActivationModelWeightedQuad(jselec)
                    impactRefJointCost = croc.CostModelResidual(
                        state, impactRefJointsAct, impactRefJointsResidual
                    )
                    costs.addCost(
                        "impactRefJoint",
                        impactRefJointCost,
                        p.refMainJointsAtImpactWeight / p.DT,
                    )

        # Flying foot
        for k, fid in enumerate(robot.contactIds):
            if pattern[k]:
                continue
            if p.verticalFootVelWeight > 0:
                verticalFootVelResidual = croc.ResidualModelFrameVelocity(
                    state,
                    fid,
                    pin.Motion.Zero(),
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    actuation.nu,
                )
                verticalFootVelAct = croc.ActivationModelWeightedQuad(
                    np.array([0, 0, 1, 0, 0, 0])
                )
                verticalFootVelCost = croc.CostModelResidual(
                    state, verticalFootVelAct, verticalFootVelResidual
                )
                costs.addCost(
                    "%s_vfoot_vel" % robot.model.frames[fid].name,
                    verticalFootVelCost,
                    p.verticalFootVelWeight,
                )

            if p.footTrajectoryWeight > 0 and (p.footTrajImportance[t][k] > 0).any():
                print("At t=%s add foot trajectory for %s" % (t, k))
                footTrajReference = p.footTrajectories[t][k]
                print("Foot trajectory reference", footTrajReference)
                footTrajResidual = croc.ResidualModelFrameTranslation(
                    state, fid, footTrajReference, actuation.nu
                )
                footTrajAct = croc.ActivationModelWeightedQuad(p.footTrajImportance[t][k])
                footTrajCost = croc.CostModelResidual(state, footTrajAct, footTrajResidual)
                costs.addCost(
                    "%s_foottraj" % robot.model.frames[fid].name,
                    footTrajCost,
                    p.footTrajectoryWeight,
                )

            if p.flyHighWeight > 0:
                groundAltitude = p.groundAltitude[t]
                flyHighResidual = sobec.ResidualModelFlyHigh(
                    state, fid, p.flyHighSlope / 2.0, groundAltitude, actuation.nu
                )
                flyHighCost = croc.CostModelResidual(state, flyHighResidual)
                costs.addCost(
                    "%s_flyhigh" % robot.model.frames[fid].name,
                    flyHighCost,
                    p.flyHighWeight,
                )

            if p.groundColWeight > 0:
                groundColRes = croc.ResidualModelFrameTranslation(
                    state, fid, np.zeros(3), actuation.nu
                )
                # groundColBounds = croc.ActivationBounds(
                # np.array([-np.inf, -np.inf, 0.01]), np.array([np.inf, np.inf, np.inf])
                # )
                # np.inf introduces an error on lb[2] ... why? TODO ... patch by replacing
                # np.inf with 1000
                groundColBounds = croc.ActivationBounds(
                    np.array([-1000, -1000, 0.0]), np.array([1000, 1000, 1000])
                )
                groundColAct = croc.ActivationModelQuadraticBarrier(groundColBounds)
                groundColCost = croc.CostModelResidual(state, groundColAct, groundColRes)
                costs.addCost(
                    "%s_groundcol" % robot.model.frames[fid].name,
                    groundColCost,
                    p.groundColWeight,
                )

            for kc, cid in enumerate(robot.contactIds):
                if not pattern[kc]:
                    continue
                assert fid != cid
                # print("At t=%s add collision between %s and %s" % (t, cid, fid))

                for id1, id2 in [
                    (i, j)
                    for i in [cid, robot.towIds[cid], robot.heelIds[cid]]
                    for j in [fid, robot.towIds[fid], robot.heelIds[fid]]
                ]:
                    if p.feetCollisionWeight > 0:
                        feetColResidual = sobec.ResidualModelFeetCollision(
                            state, id1, id2, actuation.nu
                        )
                        feetColBounds = croc.ActivationBounds(
                            np.array([p.footMinimalDistance]), np.array([1000])
                        )
                        feetColAct = croc.ActivationModelQuadraticBarrier(feetColBounds)
                        feetColCost = croc.CostModelResidual(
                            state, feetColAct, feetColResidual
                        )
                        costs.addCost(
                            (
                                "feetcol_%s_VS_%s"
                                % (
                                    robot.model.frames[id1].name,
                                    robot.model.frames[id2].name,
                                )
                            ),
                            feetColCost,
                            p.feetCollisionWeight,
                        )

        # Joint limits
        if p.jointLimitWeight > 0:
            maxfloat = sys.float_info.max
            xLimitResidual = croc.ResidualModelState(state, robot.x0, actuation.nu)
            lowerBoundsq = robot.model.lowerPositionLimit
            upperBoundsq = robot.model.upperPositionLimit

            lowerBoundsv = -np.ones(robot.model.nv)*maxfloat
            upperBoundsv = np.ones(robot.model.nv)*maxfloat

            lowerBoundsx = np.concatenate([lowerBoundsq, lowerBoundsv])
            upperBoundsx = np.concatenate([upperBoundsq, upperBoundsv])

            lowerBoundsdx = state.diff(robot.x0, lowerBoundsx)
            upperBoundsdx = state.diff(robot.x0, upperBoundsx)
            for i in range(len(lowerBoundsdx)):
                if np.isnan(lowerBoundsdx[i]) or np.isinf(lowerBoundsdx[i]):
                    lowerBoundsdx[i] = -maxfloat
                if np.isnan(upperBoundsdx[i]) or np.isinf(upperBoundsdx[i]):
                    upperBoundsdx[i] = maxfloat

            print("Lower bounds dx", lowerBoundsdx)
            print("Upper bounds dx", upperBoundsdx)
            # xLimitConstraint = croc.ConstraintModelResidual(
            #     state, xLimitResidual, lowerBoundsdx, upperBoundsdx
            # )
            # constraints.addConstraint("jointLimit", xLimitConstraint)
            xLimitActivation = croc.ActivationModelQuadraticBarrier(
                croc.ActivationBounds(lowerBoundsdx, upperBoundsdx, 0.1)
            )
            cost = croc.CostModelResidual(state, xLimitActivation, xLimitResidual)
            costs.addCost("jointLimit", cost, p.jointLimitWeight)

        if p.refJointAcceleration > 0:
            accResidual = croc.ResidualModelJointAcceleration(state, actuation.nu)
            accAct = croc.ActivationModelWeightedQuad(p.accImportance**2)
            accCost = croc.CostModelResidual(state, accAct, accResidual)
            costs.addCost("jointAcc", accCost, p.refJointAcceleration)

        # Action
        damodel = croc.DifferentialActionModelContactFwdDynamics(
            state, actuation, contacts, costs, constraints, p.kktDamping, True
        )
        amodel = croc.IntegratedActionModelEuler(damodel, p.DT)

        models.append(amodel)

    return models


# ### TERMINAL MODEL ##################################################################
def buildTerminalModel(robotWrapper, contactPattern, params, with_constraints=False):
    robot = robotWrapper
    p = params
    pattern = contactPattern[-1]

    # Horizon length
    T = len(contactPattern) - 1

    state = croc.StateMultibody(robot.model)
    if robot.actuationModel is None:
        actuation = croc.ActuationModelFloatingBase(state)
    else:
        from .actuation_matrix import ActuationModelMatrix
        act_matrix = np.zeros((robot.model.nv, len(robot.actuationModel.mot_ids_v)))
        for iu, iv in enumerate(robot.actuationModel.mot_ids_v):
            act_matrix[iv, iu] = 1
        actuation = ActuationModelMatrix(state, np.shape(act_matrix)[1], act_matrix)

    # Contacts
    contacts = croc.ContactModelMultiple(state, actuation.nu)
    for k, cid in enumerate(robot.contactIds):
        if not pattern[k]:
            continue
        contact = croc.ContactModel6D(
            state, cid, pin.SE3.Identity(), pin.WORLD, actuation.nu, p.baumgartGains
        )
        contacts.addContact(robot.model.frames[cid].name + "_contact", contact)
    for k, cm in enumerate(robot.loop_constraints_models):
        assert cm.type == pin.ContactType.CONTACT_6D and cm.reference_frame == pin.ReferenceFrame.LOCAL
        contact = croc.ContactModel6DLoop(
            state,
            cm.joint1_id,
            cm.joint1_placement,
            cm.joint2_id,
            cm.joint2_placement,
            pin.ReferenceFrame.LOCAL,
            actuation.nu,
            np.array([0., 0.])
        )
        contacts.addContact(f"loop_contact_{k}", contact)

    # Costs
    costs = croc.CostModelSum(state, actuation.nu)

    # if "stateTerminalTarget" not in locals():
    if p.stateTerminalWeight > 0:
        stateTerminalTarget = robot.x0.copy()
        stateTerminalTarget[:3] += p.vcomRef * T * p.DT
        stateTerminalResidual = croc.ResidualModelState(
            state, stateTerminalTarget, actuation.nu
        )
        stateTerminalAct = croc.ActivationModelWeightedQuad(p.stateTerminalImportance**2)
        stateTerminalCost = croc.CostModelResidual(
            state, stateTerminalAct, stateTerminalResidual
        )
        costs.addCost("stateReg", stateTerminalCost, p.stateTerminalWeight)

    damodel = croc.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, p.kktDamping, True
    )
    termmodel = croc.IntegratedActionModelEuler(damodel, p.DT)

    return termmodel


# ### SOLVER ########################################################################


def buildSolver(robotWrapper, contactPattern, walkParams, solver='FDDP'):
    with_constraints = False
    if solver == 'CSQP':
        print("Using CSQP solver, creating constraints")
        with_constraints = True
    models = buildRunningModels(robotWrapper, contactPattern, walkParams, with_constraints)
    termmodel = buildTerminalModel(robotWrapper, contactPattern, walkParams, with_constraints)

    problem = croc.ShootingProblem(robotWrapper.x0, models, termmodel)
    if solver == 'FDDP':
        ddp = croc.SolverFDDP(problem)
        ddp.verbose = True
        ddp.th_stop = walkParams.solver_th_stop
    elif solver == 'CSQP':
        try:
            import mim_solvers
        except ImportError:
            print("Please install the `mim_solvers` package to use CSQP")
            sys.exit(1)
        input("Press Enter to continue...")
        ddp = mim_solvers.SolverCSQP(problem)
        ddp.termination_tolerance = walkParams.solver_th_stop
        # ddp.eps_abs = 1e-3
        # ddp.eps_rel = 0
        ddp.mu = 100
        ddp.mu2 = 10
        ddp.max_qp_iters = 1000
        ddp.use_filter_line_search = True
        # ddp.alpha = 1e-5
        ddp.with_callbacks = True
    else:
        raise ValueError('Unknown solver: %s \n Supported option are FDDP and CSQP' % solver)
    return ddp


def buildInitialGuess(problem, walkParams):
    if walkParams.guessFile is not None:
        try:
            guess = sobec.wwt.load_traj(walkParams.guessFile)
            x0s = [x for x in guess["xs"]]
            u0s = [u for u in guess["us"]]
        except (FileNotFoundError, KeyError):
            x0s = []
            u0s = []
    else:
        x0s = []
        # us0 = []

    if len(x0s) != problem.T + 1 or len(u0s) != problem.T:
        print("No valid solution file, build quasistatic initial guess!")
        x0s = [problem.x0.copy() for _ in range(problem.T + 1)]
        u0s = [
            m.quasiStatic(d, x)
            for m, d, x in zip(problem.runningModels, problem.runningDatas, x0s)
        ]

    return x0s, u0s


# ### SOLUTION ######################################################################


class Solution:
    def __init__(self, robotWrapper, ddp):
        # model = ddp.problem.terminalModel.differential.pinocchio
        self.xs = np.array(ddp.xs)
        self.us = np.array(ddp.us)
        self.acs = np.array([d.differential.xout for d in ddp.problem.runningDatas])
        self.fs = [
            [
                (cd.data().jMf.inverse() * cd.data().f).vector
                if cm.data().contact.type == pin.LOCAL
                else cd.data().f.vector
                for cm,cd in zip(m.differential.contacts.contacts,
                                 d.differential.multibody.contacts.contacts)
            ]
            for m,d in zip(ddp.problem.runningModels,ddp.problem.runningDatas)
        ]
        self.fs0 = [
            np.concatenate(
                [
                    (
                        d.differential.multibody.contacts.contacts[
                            "%s_contact" % robotWrapper.model.frames[cid].name
                        ].jMf.inverse()
                        * d.differential.multibody.contacts.contacts[
                            "%s_contact" % robotWrapper.model.frames[cid].name
                        ].f
                        if m.differential.contacts.contacts[
                            "%s_contact" % robotWrapper.model.frames[cid].name
                        ].contact.type == pin.LOCAL
                        else
                         d.differential.multibody.contacts.contacts[
                            "%s_contact" % robotWrapper.model.frames[cid].name
                        ].f
                    ).vector
                    if "%s_contact" % robotWrapper.model.frames[cid].name
                    in d.differential.multibody.contacts.contacts
                    else np.zeros(6)
                    for cid in robotWrapper.contactIds
                ]
            )
            for m, d in zip(ddp.problem.runningModels, ddp.problem.runningDatas)
        ]
