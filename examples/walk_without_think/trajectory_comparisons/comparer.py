import pinocchio as pin
import sobec
import crocoddyl
import numpy as np
import matplotlib.pyplot as plt
import meshcat
from pinocchio.visualize import MeshcatVisualizer

from warnings import warn
import os
import sys

sys.path.append("../")
from loaders.loaders_virgile import load_complete_open, load_complete_closed
import specific_params


# Class to compare serveral trajectories
class Comparer:
    def __init__(self, slope_list=[0.0000], base_list=[0.575], velocity_list=[0.2]):
        self.logs = {}
        self.names = []
        self.robots = {}
        for b in base_list:
            self.robots[b] = {"open": load_complete_open(base_height=b), "closed": load_complete_closed(base_height=b)}
        robot_open = self.robots[base_list[0]]["open"]
        robot_closed = self.robots[base_list[0]]["closed"]
        self.walkParams = {"slope": {}, "velocity": {}}
        for s in slope_list:
            p_open = specific_params.WalkBattobotParams("open")
            p_open.slope = s
            p_closed = specific_params.WalkBattobotParams("closed")
            p_closed.slope = s
            self.walkParams["slope"][s] = {"open": p_open, "closed": p_closed}
        for v in velocity_list:
            p_open = specific_params.WalkBattobotParams("open")
            p_open.vcomRef[0] = v
            p_closed = specific_params.WalkBattobotParams("closed")
            p_closed.vcomRef[0] = v            
            self.walkParams["velocity"][v] = {"open": p_open, "closed": p_closed}
        walkParamsOpen = self.walkParams["slope"][slope_list[0]]["open"]
        walkParamsClosed = self.walkParams["slope"][slope_list[0]]["closed"]
        self.control_order_open = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.control_order_closed = [0, 1, 2, 5, 4, 3, 6, 7, 8, 11, 10, 9]
        ## Creating viewers
        # Open
        print("Opening meshcat viewer for open robot at port 6000")
        self.viz_open = MeshcatVisualizer(
            robot_open.model,
            robot_open.collision_model,
            robot_open.visual_model,
        )
        self.viz_open.viewer = meshcat.Visualizer(zmq_url=f"tcp://127.0.0.1:6000")
        self.viz_open.clean()
        self.viz_open.loadViewerModel(rootNodeName="universe")
        # Closed
        print("Opening meshcat viewer for closed robot at port 6001")
        self.viz_closed = MeshcatVisualizer(
            robot_closed.model,
            robot_closed.collision_model,
            robot_closed.visual_model,
        )
        self.viz_closed.viewer = meshcat.Visualizer(zmq_url=f"tcp://127.0.0.1:6001")
        self.viz_closed.clean()
        self.viz_closed.loadViewerModel(rootNodeName="universe")
        ## Define contacts pattern
        cycle = (
            [[1, 0]] * walkParamsOpen.Tsingle
            + [[1, 1]] * walkParamsOpen.Tdouble
            + [[0, 1]] * walkParamsOpen.Tsingle
            + [[1, 1]] * walkParamsOpen.Tdouble
        )
        self.contactPattern = (
            []
            + [[1, 1]] * walkParamsOpen.Tstart
            + (cycle * 3)
            + [[1, 1]] * walkParamsOpen.Tend
            + [[1, 1]]
        )
        ## Building the ddp problems
        self.ddps = {"slope": {}, "velocity": {}}
        for key in self.walkParams["slope"].keys():
            self.ddps["slope"][key] = {
                "open": sobec.wwt.buildSolver(
                    robot_open, self.contactPattern, self.walkParams["slope"][key]["open"]
                ),
                "closed": sobec.wwt.buildSolver(
                    robot_closed, self.contactPattern, self.walkParams["slope"][key]["closed"]
                ),
            }
        for key in self.walkParams["velocity"].keys():
            self.ddps["velocity"][key] = {
                "open": sobec.wwt.buildSolver(
                    robot_open, self.contactPattern, self.walkParams["velocity"][key]["open"]
                ),
                "closed": sobec.wwt.buildSolver(
                    robot_closed, self.contactPattern, self.walkParams["velocity"][key]["closed"]
                ),
            }
        ## Default plotting parameters
        self.colors = ["b", "r", "g", "c", "m", "y", "k"]
        self.linestyles = ["-", "--", "-.", ":"]
        self.figsize = (12, 6)

    def setColors(self, colors):
        self.colors = colors
    
    def setLinestyles(self, linestyles):
        self.linestyles = linestyles
    
    def setFigsize(self, figsize):
        self.figsize = figsize
    
    def printNames(self):
        for i, name in enumerate(self.names):
            print(f"{i}: {name}")

    def add_log(self, path, robot_type, name, s=0.0000, b=0.575, v=0.2):
        if robot_type not in ["open", "closed"]:
            raise ValueError("robot_type must be 'open' or 'closed'")
        if not os.path.isfile(path):
            raise ValueError("path must be a valid file")
        if name in self.logs.keys():
            warn("name already exists...")
            if (
                input("Should we continue with overwriting the previous log? (y/n)")
                != "y"
            ):
                return
            else:
                self.names = [n for n in self.names if n != name]
            
        robot = self.robots[b][robot_type]
        viz = self.viz_open if robot_type == "open" else self.viz_closed
        if s != 0.0000:
            params = self.walkParams["slope"][s][robot_type]
            ddp = self.ddps["slope"][s][robot_type]
        else:
            params = self.walkParams["velocity"][v][robot_type]
            ddp = self.ddps["velocity"][v][robot_type]
        control_order = (
            self.control_order_open if robot_type == "open" else self.control_order_closed
        )
        guess = sobec.wwt.load_traj(path)
        xs = guess["xs"]
        us = guess["us"]
        if "n_iter" in guess.keys():
            n_iter = guess["n_iter"]
        else:
            warn(f"n_iter not found in the log {name}, setting it to -1")
            n_iter = -1

        self.logs[name] = {
            "xs": xs,
            "us": us,
            "n_iter": n_iter,
            "robot": robot,
            "params": params,
            "viz": viz,
            "ddp": ddp,
            "control_order": control_order,
        }
        self.names.append(name)

    def assertLogsCompatible(self):
        assert (
            self.walkParamsClosed.Tsingle == self.walkParamsOpen.Tsingle
        ), "Tsingle are different for open and closed problems"
        assert (
            self.walkParamsClosed.Tdouble == self.walkParamsOpen.Tdouble
        ), "Tdouble are different for open and closed problems"
        assert (
            self.walkParamsClosed.Tstart == self.walkParamsOpen.Tstart
        ), "Tstart are different for open and closed problems"
        assert (
            self.walkParamsClosed.Tend == self.walkParamsOpen.Tend
        ), "Tend are different for open and closed problems"

        for log in self.logs.values():
            assert len(log["xs"]) == len(
                self.contactPattern
            ), f"Contact pattern and xs length are different for log {log["name"]}"
            assert (
                len(log["us"]) == len(self.contactPattern) - 1
            ), f"Contact pattern and us length are inconsistent for log {log["name"]}"

    def printIters(self):
        print("%%%%%%% Number of iterations %%%%%%%")
        for log in self.logs.values():
            print(f"{log["name"]}: {log["n_iter"]}")

    def playTraj(self, name="", number=-1):
        if name == "" and number == -1:
            raise ValueError("You must specify a name or a log number")
        if name != "" and number != -1:
            raise ValueError("You must specify either a name or a log number, not both")
        if name == "":
            name = self.names[number]
        log = self.logs[name]
        log["viz"].play(log.xs[:, : log["robot"].model.nq], log["params"].DT)

    ##############################################
    ## Trajectory Gaps computation and plotting ##
    ##############################################
    def computeTrajGaps(self, name):
        log = self.logs[name]
        gaps = []
        for t, (x0, x1, u) in enumerate(zip(log["xs"][:-1], log["xs"][1:], log["us"])):
            rm = log["ddp"].problem.runningModels[t]
            rd = log["ddp"].problem.runningDatas[t]
            rm.calc(rd, x0, u)
            xnext = rd.xnext
            gap = rm.state.diff(xnext, x1)
            gaps.append(gap)
        log["gaps"] = gaps

    def computeAllTrajGaps(self):
        for name in self.names:
            self.computeTrajGaps(name)

    def plotTrajGaps(self, name):
        log = self.logs[name]
        fig, (axis_pos, axis_vel) = plt.subplots(
            nrows=2, ncols=1, figsize=(12, 6), sharex=True
        )
        vel_gaps = [gap[: log["robot"].model.nv] for gap in log["gaps"]]
        pos_gaps = [gap[log["robot"].model.nv :] for gap in log["gaps"]]
        for i, pos in enumerate(pos_gaps):
            axis_pos.plot(pos, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
        for i, vel in enumerate(vel_gaps):
            axis_vel.plot(vel, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
        axis_pos.set_title("Position gaps")
        axis_vel.set_title("Velocity gaps")
        # axis_pos.set_xlabel("t")
        axis_vel.set_xlabel("t")
        axis_pos.grid()
        axis_vel.grid()
        plt.suptitle(f"{name} gaps")
        plt.show()

    def plotAllTrajGaps(self, names=[]):
        if names == []:
            names = self.names
        for name in names:
            self.plotTrajGaps(name)

    ############################################################
    ## Contact constraints violation computation and plotting ##
    ############################################################
    # ? Would be nice if we could plot the contact acceleration as well
    # ? It would help verifying that the trajectory is correct
    def computeConstraintsResidual(self, name):
        log = self.logs[name]
        model = log["robot"].model
        data = model.createData()
        loop_pos_res = []
        loop_vel_res = []
        floor_pos_res = []
        floor_vel_res = []
        for t, x in enumerate(log["xs"]):
            q, v = x[: model.nq], x[model.nq :]
            pin.forwardKinematics(model, data, q, v)
            # Loop constraints
            loops_res_pos_t = []
            loops_res_vel_t = []
            if len(log["robot"].loop_constraints_models) != 0:
                for loop_cm in log["robot"].loop_constraints_models:
                    oMc1 = data.oMi[loop_cm.joint1_id] * loop_cm.joint1_placement
                    oMc2 = data.oMi[loop_cm.joint2_id] * loop_cm.joint2_placement
                    c1Mc2 = oMc1.inverse() * oMc2
                    pos_err = pin.log(c1Mc2).vector
                    c1vc1 = pin.getFrameVelocity(
                        model,
                        data,
                        loop_cm.joint1_id,
                        loop_cm.joint1_placement,
                        pin.LOCAL,
                    )
                    c2vc2 = pin.getFrameVelocity(
                        model,
                        data,
                        loop_cm.joint2_id,
                        loop_cm.joint2_placement,
                        pin.LOCAL,
                    )
                    c1vc2 = c1Mc2.act(c2vc2)
                    vel_err = (c1vc1 - c1vc2).vector
                    loops_res_pos_t.append(pos_err)
                    loops_res_vel_t.append(vel_err)
                loop_pos_res.append(np.concatenate(loops_res_pos_t))
                loop_vel_res.append(np.concatenate(loops_res_vel_t))
            # Floor constraints
            floor_pos_res_t = {i: [] for i in log["robot"].contactIds}
            floor_vel_res_t = {i: [] for i in log["robot"].contactIds}
            for contact, active in enumerate(self.contactPattern[t]):
                contact = log["robot"].contactIds[contact]
                if active == 0:
                    continue
                pos_err = pin.log6(data.oMf[contact]).vector[[2, 3, 4]]
                vel_err = pin.getFrameVelocity(model, data, contact, pin.WORLD).vector[
                    [2, 3, 4]
                ]
                floor_pos_res_t[contact].append(pos_err)
                floor_vel_res_t[contact].append(vel_err)
            for i in log["robot"].contactIds:
                if len(floor_pos_res_t[i]) == 0:
                    floor_pos_res_t[i].append(np.zeros(3))
                    floor_vel_res_t[i].append(np.zeros(3))
            floor_pos_res.append(
                np.concatenate([floor_pos_res_t[i] for i in log["robot"].contactIds])
            )
            floor_vel_res.append(
                np.concatenate([floor_vel_res_t[i] for i in log["robot"].contactIds])
            )
        if len(log["robot"].loop_constraints_models) != 0:
            loop_pos_res = np.reshape(loop_pos_res, (len(log["xs"]), -1, 6))
            loop_vel_res = np.reshape(loop_vel_res, (len(log["xs"]), -1, 6))
        floor_pos_res = np.reshape(floor_pos_res, (len(log["xs"]), -1, 3))
        floor_vel_res = np.reshape(floor_vel_res, (len(log["xs"]), -1, 3))
        log["constraints_res"] = {
            "loop_pos": loop_pos_res,
            "loop_vel": loop_vel_res,
            "floor_pos": floor_pos_res,
            "floor_vel": floor_vel_res,
        }

    def computeAllConstraintsResidual(self):
        for name in self.names:
            self.computeConstraintsResidual(name)

    def plotConstraintsResidual(self, name):
        fig, ((ax_pos_loop, ax_pos_floor), (ax_vel_loop, ax_vel_floor)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        loop_pos_res, loop_vel_res, floor_pos_res, floor_vel_res = (
            self.logs[name]["constraints_res"]["loop_pos"],
            self.logs[name]["constraints_res"]["loop_vel"],
            self.logs[name]["constraints_res"]["floor_pos"],
            self.logs[name]["constraints_res"]["floor_vel"],
        )
        ax_pos_loop.plot(np.linalg.norm(loop_pos_res, axis=0))
        ax_vel_loop.plot(np.linalg.norm(loop_vel_res, axis=0))
        ax_pos_floor.plot(np.linalg.norm(floor_pos_res, axis=0))
        ax_vel_floor.plot(np.linalg.norm(floor_vel_res, axis=0))

        ax_pos_loop.set_title("Loop constraints position residuals")
        ax_pos_floor.set_title("Floor constraints position residuals")
        ax_vel_loop.set_title("Loop constraints velocity residuals")
        ax_vel_floor.set_title("Floor constraints velocity residuals")
        for ax in (ax_pos_loop, ax_pos_floor, ax_vel_loop, ax_vel_floor):
            ax.grid()
        # plt.legend()
        plt.suptitle(f"{name} constraints residuals")
        plt.show()

    def plotAllConstraintsResidual(self, names=[]):
        if names == []:
            names = self.names
        for name in names:
            self.plotConstraintsResidual(name)

    ##############################################
    ## Trajectory Cost computation and plotting ##
    ##############################################
    def computeCosts(self, name):
        log = self.logs[name]
        running_costs = {}
        ddp = log["ddp"]
        for t, (x, u) in enumerate(zip(log["xs"][:-1], log["us"])):
            rm = ddp.problem.runningModels[t]
            rd = ddp.problem.runningDatas[t]
            rm.calc(rd, x, u)
            costs_dict = rd.differential.costs.costs.todict()
            for key, cost in costs_dict.items():
                if key not in running_costs:
                    running_costs[key] = []
                running_costs[key].append(cost.cost)
            for key in running_costs.keys():
                if key not in costs_dict.keys():
                    running_costs[key].append(0)
        # Add missing zeros at the begginning
        for key in running_costs.keys():
            running_costs[key] = [0] * (
                ddp.problem.T - len(running_costs[key])
            ) + running_costs[key]
        ddp.problem.terminalModel.calc(ddp.problem.terminalData, log["xs"][-1])
        terminal_cost = ddp.problem.terminalData.cost
        total_cost = np.sum(list(running_costs.values()), axis=0)
        log["costs"] = {
            "running": running_costs,
            "terminal": terminal_cost,
            "total": total_cost,
        }

    def computeAllCosts(self):
        for name in self.names:
            self.computeCosts(name)

    def compareTotalCosts(self, names=[], ids=[], title=None):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        fig, ax = plt.subplots(figsize=self.figsize)
        for i, name in enumerate(names):
            total_running_costs = self.logs[name]["costs"]["total"]
            ax.plot(
                total_running_costs,
                label=name,
                c=self.colors[i % len(self.colors)],
                linestyle=self.linestyles[i % len(self.linestyles)],
            )
            ax.set_xlabel("t")
            ax.set_ylabel("Total cost")
            ax.grid()
            ax.legend()
        plt.suptitle(title)
        plt.show()
    
    def printTerminalCosts(self, names=[], ids=[]):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        print("%%%%%%% Terminal costs %%%%%%%")
        for name in names:
            print(f"{name}: {self.logs[name]["costs"]["terminal"]}")

    def compareRunningCosts(self, names=[], ids=[], title=None, keys=[]):
        if keys == []:
            warn("No keys specified, plotting all costs")
            keys = self.logs[self.names[0]]["costs"]["running"].keys()
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        n_costs = len(keys)
        n_cols = 2
        n_rows = n_costs // n_cols
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
        for i, name in enumerate(names):
            running_costs = self.logs[name]["costs"]["running"]
            for j, (key, ax) in enumerate(zip(keys, axs.flatten())):
                cost = running_costs[key]
                ax.plot(
                    cost,
                    label=f"{name}", 
                    c=self.colors[i%len(self.colors)], 
                    linestyle=self.linestyles[i%len(self.linestyles)]
                )
                ax.set_xlabel("t")
                
                major_ticks = np.arange(0, len(cost), len(cost)//10)
                minor_ticks = np.arange(0, len(cost), len(cost)//2)

                # ? Need to fix the ticks for the grid plotting
                ax.set_xticks(major_ticks)
                ax.set_xticks(minor_ticks, minor=True)

                ax.grid(True, which='both')
                # ax.grid(which='minor', alpha=0.2)
                # ax.grid(which='major', alpha=0.5)
                ax.set_title(f"{key}")
                ax.legend()
        plt.suptitle(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    #################################################
    ## Controls Trajectory comparison and plotting ##
    #################################################
    def plotControls(self, names=[], ids=[], title=None):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        fig, axs = plt.subplots(
            nrows=6, ncols=2, figsize=self.figsize, sharex=True, sharey=False
        )
        for i, name in enumerate(names):
            us = self.logs[name]["us"].T
            us = us[self.logs[name]["control_order"]]
            for j, (us_i, ax) in enumerate(zip(us, axs.T.flatten())):
                ax.plot(
                    us_i,
                    label=name,
                    c=self.colors[i % len(self.colors)],
                    linestyle=self.linestyles[i % len(self.linestyles)],
                )
                ax.set_title(f"u_{j}")
                ax.grid()
                ax.legend()
        plt.suptitle(title)
        plt.show()
    
    def plotControlsDiff(self, names=[], ids=[], title=None, labels=[]):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        if len(names) % 2 != 0:
            raise ValueError("You must specify an even number of names or ids to compare")
        if len(labels) != 0 and len(labels) != len(names)//2:
            raise ValueError("You must specify the same number of labels as names")
        if len(labels) == 0:
            labels = [f"Diff {names[i]} / {names[i+1]}" for i in range(0, len(names), 2)
            ]
        names = [(names[i], names[i+1]) for i in range(0, len(names), 2)
        ]
        fig, axs = plt.subplots(
            nrows=6, ncols=2, figsize=self.figsize, sharex=True, sharey=False
        )
        for i, name_tuple in enumerate(names):
            us1 = self.logs[name_tuple[0]]["us"].T
            us1 = us1[self.logs[name_tuple[0]]["control_order"]]
            us2 = self.logs[name_tuple[1]]["us"].T
            us2 = us2[self.logs[name_tuple[1]]["control_order"]]
            us_diff = us1 - us2
            for j, (us_diff_i, ax) in enumerate(zip(us_diff, axs.T.flatten())):
                ax.plot(us_diff_i, label=labels[i], linestyle=self.linestyles[i%len(self.linestyles)])
                ax.set_title(f"u_{j}")
                ax.grid()
                ax.legend()
        plt.suptitle(title)
        plt.show()

    #############################################
    ## Feet Trajectory comparison and plotting ##
    #############################################
    def computeFootTrajectory(self, name):
        log = self.logs[name]
        foottraj = []
        footvtraj = []
        
        model = log["robot"].model
        data = model.createData()
        for t, x in enumerate(log["xs"]):
            q, v = x[: model.nq], x[model.nq :]
            pin.forwardKinematics(model, data, q, v)
            pin.updateFramePlacements(model, data)
            foottraj.append(
                    [
                        data.oMf[cid].translation.copy()
                        for cid in log["robot"].contactIds                     
                    ]
            )
            footvtraj.append(
                    [
                        pin.getFrameVelocity(model, data, cid, pin.WORLD).linear
                        for cid in log["robot"].contactIds
                    ]
            )
        log["foot_traj"] = {'position': foottraj, 'velocity': footvtraj}

    def computeAllFootTrajectories(self):
        for name in self.names:
            self.computeFootTrajectory(name)
    
    def plotFootTrajectory(self, names=[], ids=[], title=None):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        for traj_type in ["position", "velocity"]:
            fig, axs = plt.subplots(
                nrows=3, ncols=2, figsize=self.figsize, sharex=True, sharey=False
            )
            for i, name in enumerate(names):
                traj = np.reshape(self.logs[name]["foot_traj"][traj_type], (-1, 6))
                for j, (traj_ax, ax) in enumerate(zip(traj.T, axs.T.flatten())):
                    ax.plot(traj_ax, label=name, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
                    ax.set_title(f"Foot {["left", "right"][j//3]} {['x', 'y', 'z'][j%3]} {traj_type}")
                    ax.grid()
                    ax.legend()
            plt.suptitle(title)
            plt.show()

    def plotFootXYTrajectory(self, names=[], ids=[], title=None):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        for traj_type in ["position", "velocity"]:
            fig, axs = plt.subplots(
                nrows=1, ncols=2, figsize=self.figsize, sharex=True, sharey=True
            )
            for i, name in enumerate(names):
                traj = np.reshape(self.logs[name]["foot_traj"][traj_type], (-1, 6))
                traj_x = traj[:, [0, 3]]
                traj_y = traj[:, [1, 4]]
                print(np.shape(traj_x))
                for j, (traj_xi, traj_yi, ax) in enumerate(zip(traj_x.T, traj_y.T, axs.T.flatten())):
                    ax.plot(traj_xi, traj_yi, label=name, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title(f"Foot {["left", "right"][j]} {traj_type} - XY plane")
                    ax.grid()
                    ax.legend()
                    # ax.axis("equal")
            plt.suptitle(title)
            plt.show()

    ############################################
    ## COM Trajectory comparison and plotting ##
    ############################################
    def computeCOMTrajectory(self, name):
        log = self.logs[name]
        comtraj = []
        comvtraj = []
        
        model = log["robot"].model
        data = model.createData()
        for t, x in enumerate(log["xs"]):
            q, v = x[: model.nq], x[model.nq :]
            pin.centerOfMass(model, data, q, v)
            comtraj.append(
                data.com[0].copy()
            )
            comvtraj.append(
                data.vcom[0].copy()
            )
        log["com"] = {'position': comtraj, 'velocity': comvtraj}
    
    def computeAllCOMTrajectories(self):
        for name in self.names:
            self.computeCOMTrajectory(name)
    
    def plotCOMTrajectory(self, names=[], ids=[], title=None, velocity=True):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        if velocity:
            traj_types = ["position", "velocity"]
        else:
            traj_types = ["position"]
        for traj_type in traj_types:
            fig, axs = plt.subplots(
                nrows=1, ncols=3, figsize=self.figsize, sharex=True, sharey=False
            )
            for i, name in enumerate(names):
                traj = np.array(self.logs[name]["com"][traj_type])
                for j, (traj_ax, ax) in enumerate(zip(traj.T, axs.T.flatten())):
                    ax.plot(traj_ax, label=name, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
                    ax.set_title(f"COM {['x', 'y', 'z'][j]} {traj_type}")
                    ax.grid()
                    ax.legend()
            plt.suptitle(title)
            plt.show()

    def plotCOMXYTrajectory(self, names=[], ids=[], title=None, velocity=True):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        if velocity:
            traj_types = ["position", "velocity"]
        else:
            traj_types = ["position"]
        for traj_type in traj_types:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=self.figsize
            )
            for i, name in enumerate(names):
                traj = np.array(self.logs[name]["com"][traj_type])
                traj_x = traj[:, 0]
                traj_y = traj[:, 1]
                ax.plot(traj_x, traj_y, label=name, c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"COM {traj_type} - XY plane")
                ax.grid()
                ax.legend()
                ax.axis("equal")
            plt.suptitle(title)
            plt.show()

    def plotCOMDiff(self, names=[], ids=[], title=None, labels=[], velocity=True):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        if len(names) % 2 != 0:
            raise ValueError("You must specify an even number of names or ids to compare")
        if len(labels) != 0 and len(labels) != len(names)//2:
            raise ValueError("You must specify the same number of labels as names")
        if len(labels) == 0:
            labels = [f"Diff {names[i]} / {names[i+1]}" for i in range(0, len(names), 2)]
        names = [(names[i], names[i+1]) for i in range(0, len(names), 2)]
        if velocity:
            traj_types = ["position", "velocity"]
        else:
            traj_types = ["position"]
        for traj_type in traj_types:
            fig, axs = plt.subplots(
                nrows=1, ncols=3, figsize=self.figsize, sharex=True, sharey=False
            )
            for i, name_tuple in enumerate(names):
                com_diff = np.array(self.logs[name_tuple[0]]["com"][traj_type]) - np.array(self.logs[name_tuple[1]]["com"][traj_type])
                for j, (com_diff, ax) in enumerate(zip(com_diff.T, axs.T.flatten())):
                    ax.plot(com_diff, label=labels[i], linestyle=self.linestyles[i%len(self.linestyles)])
                    ax.set_title(f"{['X', 'Y', 'Z'][j]} Axis")
                    ax.grid()
                    ax.legend()
            plt.suptitle(title)
            plt.show()

    def plotCOMXYDiff(self, names=[], ids=[], title=None):
        if names == [] and ids == []:
            names = self.names
        elif names != [] and ids != []:
            raise ValueError("You must specify either names or ids, not both")
        elif ids != []:
            names = [self.names[i] for i in ids]
        if len(names) % 2 != 0:
            raise ValueError("You must specify an even number of names or ids to compare")
        names = [(names[i], names[i+1]) for i in range(0, len(names), 2)]
        for traj_type in ["position", "velocity"]:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, figsize=self.figsize
            )
            for i, name_tuple in enumerate(names):
                traj1 = np.array(self.logs[name_tuple[0]]["com"][traj_type])
                traj1_x = traj1[:, 0]
                traj1_y = traj1[:, 1]
                traj2 = np.array(self.logs[name_tuple[1]]["com"][traj_type])
                traj2_x = traj2[:, 0]
                traj2_y = traj2[:, 1]
                traj_x = traj1_x - traj2_x
                traj_y = traj1_y - traj2_y
                ax.plot(traj_x, traj_y, label=f"Diff {name_tuple[0]} / {name_tuple[1]}", c=self.colors[i%len(self.colors)], linestyle=self.linestyles[i%len(self.linestyles)])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"COM {traj_type} - XY plane")
                ax.grid()
                ax.legend()
                ax.axis("equal")
            plt.suptitle(title)
            plt.show()
