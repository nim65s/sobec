import matplotlib.pylab as plt
import pinocchio as pin
import numpy as np
from numpy.linalg import norm
from matplotlib.collections import LineCollection
from itertools import cycle

class CostPlotter:
    def __init__(self, model, ddp):
        self.model = model
        self.problem = ddp.problem
        self.runningModels = self.problem.runningModels
        self.terminalModel = self.problem.terminalModel
        self.data = self.model.createData()

    def setData(self):
        self.runningDatas = self.problem.runningDatas
        self.terminalData = self.problem.terminalData
    
    def plotCosts(self):
        terminal_costs = {}
        running_costs = {"total_cost": []}

        for k, (rm, rd) in enumerate(zip(self.runningModels, self.runningDatas)):
            # Add each running cost to list
            for key, cost in rd.differential.costs.costs.todict().items():
                if key not in running_costs.keys():
                    running_costs[key] = [0] * k
                w = rm.differential.costs.costs.todict()[key].weight
                running_costs[key].append(cost.cost * w)
            running_costs["total_cost"].append(rd.differential.cost)
            # Add the missing values if needed
            for logged_keys in running_costs.keys():
                if len(running_costs[logged_keys]) < k+1:
                    running_costs[logged_keys].append(0)
        # Terminal costs
        for key, cost in self.terminalData.differential.costs.costs.todict().items():
            terminal_costs[key] = cost.cost
        
        plt.figure("Running Costs")
        n_running = len(running_costs.keys())
        color = iter(plt.cm.hsv(np.linspace(0, 1, n_running)))
        linestyle = cycle(["-", "--", ":"])
        for key, values in running_costs.items():
            c = next(color)
            ls = next(linestyle)
            plt.plot(values, linestyle=ls, label=key, c=c)
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.savefig("running_costs.png")

        plt.figure("Terminal Costs")
        n_terminal = len(terminal_costs.keys())
        color = iter(plt.cm.hsv(np.linspace(0, 1, n_terminal)))
        linestyle = cycle(["-", "--", ":"])
        for key, value in terminal_costs.items():
            c = next(color)
            plt.axhline(value, linestyle="-.", label=key, c=c)
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.tight_layout()       
        # plt.savefig("terminal_costs.png") 

class WalkPlotter:
    """
    Plot all the data out of the walk OCP.
    """

    def __init__(self, model, contactIds):
        self.model = model
        self.data = self.model.createData()
        self.contactIds = contactIds

    def setData(self, contactPattern, xs_sol, us_sol, fs_sol0):
        self.contactPattern = contactPattern
        self.xs = xs_sol
        self.us = us_sol
        self.fs = fs_sol0
        self.computeFootTraj()

    def computeFootTraj(self):
        self.foottraj = []
        self.footvtraj = []
        self.Mfoottraj = []
        for x in self.xs:
            pin.forwardKinematics(
                self.model, self.data, x[: self.model.nq], x[self.model.nq :]
            )
            pin.updateFramePlacements(self.model, self.data)
            self.foottraj.append(
                np.concatenate(
                    [self.data.oMf[cid].translation for cid in self.contactIds]
                )
            )
            self.Mfoottraj.append(
                { cid: self.data.oMf[cid].copy() for cid in self.contactIds }
            )
            self.footvtraj.append(
                np.concatenate(
                    [
                        pin.getFrameVelocity(self.model, self.data, cid).vector
                        for cid in self.contactIds
                    ]
                )
            )
        self.foottraj = np.array(self.foottraj)
        self.footvtraj = np.array(self.footvtraj)

    def plotBasis(self, target):
        # Robot basis movement
        legend = ["x", "y", "z"]
        plt.figure("Basis move")
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.title("Base link position_" + legend[i])
            plt.plot(self.xs[:, i])
            plt.axhline(y=target[i], color="black", linestyle="--")

    def plotTimeCop(self):
        # Cop of each foot vs time
        plt.figure("cop time local")
        for ifig, cid in enumerate(self.contactIds):
            plt.subplot(len(self.contactIds), 1, ifig + 1)
            # ftraj = [
            # [t, f[6 * ifig : 6 * ifig + 6]]
            # for t, (f, p) in enumerate(zip(self.fs, self.contactPattern))
            # if cid in patternToId(p)
            # ]
            ftraj = [
                [t, f[6 * ifig : 6 * ifig + 6]]
                for t, (f, p) in enumerate(zip(self.fs, self.contactPattern))
                if p[ifig]
            ]
            cop = [[t, [f[4] / f[2], -f[3] / f[2]]] for (t, f) in ftraj]
            plt.plot([t for t, p in cop], [p for t, p in cop], ".")

    def plotComAndCopInXY(self):
        plt.figure('COM and COP in XY plane')
        complot = []
        for x in self.xs:
            pin.centerOfMass(
                self.model, self.data, x[: self.model.nq], x[self.model.nq :]
            )
            complot.append(self.data.com[0].copy())
        complot = np.array(complot)

        plt.plot(complot[:,0],complot[:,1], label='com')
        
        colors = 'byrm'
        cops = []
        for ic,cid in enumerate(self.contactIds):
            ftraj = [
                [t, (M[cid]*pin.Force(f[6 * ic : 6 * ic + 6])).vector]
                for t, (f, M, p) in enumerate(zip(self.fs, self.Mfoottraj, self.contactPattern))
                if p[ic]
            ]
            cop = np.array([[t, -f[4] / f[2], f[3] / f[2] ] for (t, f) in ftraj])
            cops.append( cop )
            plt.plot(cop[:,1],cop[:,2],'.',color=colors[ic],
                     label=f'cop_{self.model.frames[cid].name.split("_")[0]}')

        ftraj = [
            sum([ (M[cid]*pin.Force(f[6*ic:6*ic+6])).vector
                  for ic,cid in enumerate(self.contactIds) ])
            for M,f in zip(self.Mfoottraj,self.fs) ]
        coptraj = np.array([ [-f[4]/f[2], f[3]/f[2] ] for f in ftraj ])
        plt.plot(coptraj[:,0],coptraj[:,1],'k:',label='cop')
        plt.legend()
        
    def plotTimeCof(self):
        plt.figure("cof time local")
        for ifig, cid in enumerate(self.contactIds):
            plt.subplot(len(self.contactIds), 1, ifig + 1)
            # ftraj = [
            # [t, f[6 * ifig : 6 * ifig + 6]]
            # for t, (f, p) in enumerate(zip(self.fs, self.contactPattern))
            # if cid in patternToId(p)
            # ]
            ftraj = [
                [t, f[6 * ifig : 6 * ifig + 6]]
                for t, (f, p) in enumerate(zip(self.fs, self.contactPattern))
                if p[ifig]
            ]
            cof = [[t, [f[1] / f[2], -f[0] / f[2]]] for (t, f) in ftraj]
            plt.plot([t for t, p in cof], [p for t, p in cof], ".")

    def plotCopAndFeet(self, footSize, ARENA_SIZE=0.6):
        # Cop of each foot in x-vs-y (with limits)
        plt.figure(figsize=(12, 6))
        plt.title("cop local")
        l_foot = np.array(
            [
                [-footSize, -footSize, 0, 1],
                [-footSize, footSize, 0, 1],
                [footSize, footSize, 0, 1],
                [footSize, -footSize, 0, 1],
                [-footSize, -footSize, 0, 1],
            ]
        ).T
        for ifig, cid in enumerate(self.contactIds):
            plt.subplot(1, len(self.contactIds), ifig + 1)
            if isinstance(ARENA_SIZE,float):
                plt.axis(
                    [-ARENA_SIZE / 4, ARENA_SIZE * 3 / 4, -ARENA_SIZE / 2, ARENA_SIZE / 2]
                )
            else:
                plt.axis(ARENA_SIZE)
            plt.xlabel(self.model.frames[cid].name)
            for t, pattern in enumerate(self.contactPattern[:-1]):
                # if cid not in patternToId(pattern): continue
                if not self.contactPattern[t][ifig]:
                    continue
                f = self.fs[t][6 * ifig : 6 * ifig + 6]
                l_cop = np.array([f[4] / f[2], -f[3] / f[2], 0])
                pin.framesForwardKinematics(
                    self.model, self.data, self.xs[t][: self.model.nq]
                )
                w_cop = self.data.oMf[cid] * l_cop
                plt.plot(w_cop[0], w_cop[1], "r.")
                w_foot = np.dot(self.data.oMf[cid].homogeneous, l_foot)
                plt.plot(w_foot[0, :], w_foot[1, :], "grey")

    def plotForces(self, referenceForces=None):
        if referenceForces is None:
            referenceForces = []
        # Forces and reference forces wrt time
        plt.figure("forces")
        frefplot = np.array(referenceForces)
        fs0plot = np.array(self.fs)
        plt.subplot(211)
        plt.plot(fs0plot[:, 2])
        if len(frefplot) > 0:
            plt.plot(frefplot[:, 2])
        plt.xlabel(self.model.frames[self.contactIds[0]].name)
        plt.subplot(212)
        plt.plot(fs0plot[:, 8])
        if len(frefplot > 0):
            plt.plot(frefplot[:, 8])
        plt.xlabel(self.model.frames[self.contactIds[1]].name)

    def plotAllForces(self):
        # Forces and reference forces wrt time
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 8))
        fs0plot = np.array(self.fs)

        hp1 = axs[0].plot(fs0plot[:, [0, 1]])
        hp2 = axs[0].plot(fs0plot[:, [6, 7]])
        axs[0].set_xlabel("lat. f")
        axs[0].legend(hp1 + hp2, ["xl", "yl", "xr", "yl"])

        hp1 = axs[1].plot(fs0plot[:, [2, 8]])
        axs[1].set_xlabel("norm f")
        axs[1].legend(hp1 + hp2, ["zl", "zr"])

        hp1 = axs[2].plot(fs0plot[:, [3, 4]])
        hp2 = axs[2].plot(fs0plot[:, [9, 10]])
        axs[2].set_xlabel("lat. tau")
        axs[2].legend(hp1 + hp2, ["xl", "yl", "xr", "yl"])

        hp1 = axs[3].plot(fs0plot[:, [5, 11]])
        axs[3].set_xlabel("norm. tau")
        axs[3].legend(hp1 + hp2, ["zl", "zr"])

    def plotCom(self, com0):
        # COM position and velocity (x+y separated from z)
        plt.figure("com", figsize=(6, 8))
        complot = []
        vcomplot = []
        for x in self.xs:
            pin.centerOfMass(
                self.model, self.data, x[: self.model.nq], x[self.model.nq :]
            )
            complot.append(self.data.com[0].copy())
            vcomplot.append(self.data.vcom[0].copy())
        complot = np.array(complot)
        vcomplot = np.array(vcomplot)
        plt.subplot(411)
        plt.plot(complot[:, :2])
        plt.ylabel("pos x-y")
        plt.subplot(412)
        plt.plot(complot[:, 2])
        plt.ylabel("pos z")
        ax = plt.axis()
        plt.axis(
            (ax[0], ax[1], com0[2] - 2.5e-2, com0[2] + 2.5e-2)
        )  # yaxis is 5cm around 0 position
        plt.subplot(413)
        plt.plot(vcomplot[:, :2])
        plt.ylabel("vel x-y")
        plt.legend(["x", "y"])
        plt.subplot(414)
        plt.plot(vcomplot[:, 2])
        plt.ylabel("vel z")

    def plotFeet(self):
        # Foot position and velocity

        plt.figure("foot")
        plt.subplot(311)
        hplot = []
        names = []
        for i, cid in enumerate(self.contactIds):
            hplot.extend(plt.plot(self.foottraj[:, 3 * i + 2]))
            names.append(self.model.frames[cid].name)
        plt.legend(hplot, names)
        plt.ylabel("altitude")
        plt.subplot(313)
        hplot = []
        for i, cid in enumerate(self.contactIds):
            hplot.extend(plt.plot(self.foottraj[:, 3 * i], self.foottraj[:, 3 * i + 2]))
        plt.legend(hplot, names)
        plt.ylabel("x-z traj")
        plt.subplot(312)
        hplot = []
        for i, cid in enumerate(self.contactIds):
            hplot.extend(
                plt.plot(np.sqrt(np.sum(self.footvtraj[:, 6 * i : 6 * i + 2] ** 2, 1)))
            )
        plt.legend(hplot, names)
        plt.ylabel("horz vel")

    def plotFootCollision(self, footMinimalDistance, subsample=0):
        plt.figure("foot collision")
        h1 = plt.plot([f[0] for f in self.foottraj], [f[1] for f in self.foottraj])
        h2 = plt.plot([f[3] for f in self.foottraj], [f[4] for f in self.foottraj])
        plt.legend(h1 + h2, ["left", "right"])
        for t, __p in enumerate(self.xs):
            if subsample > 0 and t % subsample:
                continue
            a = self.foottraj[t][:2]
            b = self.foottraj[t][3:5]
            m = (a + b) / 2
            d = b - a
            d /= norm(d)
            aa = m + d * footMinimalDistance / 2
            bb = m - d * footMinimalDistance / 2
            plt.plot([aa[0], bb[0]], [aa[1], bb[1]], "grey")
        plt.axis([-0.1, 0.4, -0.25, 0.25])

    def plotJointTorques(self):
        others = len(self.us[0]) > 12
        fig, axs = plt.subplots(
            6, 2 if others else 1, sharex=True, figsize=(12 if others else 6, 8)
        )
        axlegs = axs[:, 0] if others else axs
        for iax, ax in enumerate(axlegs):
            h1 = ax.plot(self.us[:, iax])
            h2 = ax.plot(self.us[:, iax + 6])
            if iax == 0:
                ax.legend(h1 + h2, ["left", "right"])
            ax.set_ylabel("torque %d" % iax)

        for i in range(12, len(self.us[0])):
            axs[i - 12, 1].plot(self.us[:, i])
            ax.set_ylabel("torque %d" % iax)

# ######################################################################
# ### PLOT FORCES ######################################################
# ######################################################################


def getForcesFromProblemDatas(problem, cid):
    fs = []
    for t, (m, d) in enumerate(zip(problem.runningModels, problem.runningDatas)):
        dm = m.differential
        model = dm.pinocchio
        cname = "%s_contact" % model.frames[cid].name
        if cname not in dm.contacts.contacts:
            fs.append(np.zeros(6))
        else:
            dd = d.differential.multibody.contacts.contacts[cname]
            fs.append((dd.jMf.inverse() * dd.f).vector)
    fs = np.array(fs)
    return fs


def getReferenceForcesFromProblemModels(problem, cid):
    fs = []
    for t, (m, d) in enumerate(zip(problem.runningModels, problem.runningDatas)):
        dm = m.differential
        model = dm.pinocchio
        cname = "%s_forceref" % model.frames[cid].name
        if cname not in dm.costs.costs:
            fs.append(np.zeros(6))
        else:
            cm = dm.costs.costs[cname].cost
            fs.append(cm.residual.reference.vector.copy())
    fs = np.array(fs)
    return fs


def plotProblemForces(problem, contactIds):
    fig, axs = plt.subplots(len(contactIds), 1, sharex=True)
    for ax, cid in zip(axs, contactIds):
        fs = getForcesFromProblemDatas(problem, cid)
        ax.plot(fs[:, 2])


# ######################################################################
# ### MPC ##############################################################
# ######################################################################

# from matplotlib import colors as mcolors


def vanishingPlot(t0, xs, axs=None, color=None):
    if color is None:
        color = np.arange(xs.shape[0])
    if axs is None:
        fig, axs = plt.subplots(xs.shape[1], 1, sharex=True)
    ts = np.arange(t0, t0 + xs.shape[0])
    for ix, x in enumerate(xs.T):
        points = np.array([ts, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(color.min(), color.max())
        lc = LineCollection(segments, cmap="Blues_r", norm=norm)
        lc.set_array(color)
        lc.set_linewidth(2)

        axs[ix].set_xlim(0, len(x))
        axs[ix].set_ylim(x.min(), x.max())


class WalkRecedingPlotter:
    def __init__(self, model, contactIds, hxs):
        self.contactIds = contactIds
        self.plotters = []
        for xs in hxs:
            p = WalkPlotter(model, contactIds)
            p.setData(None, xs, None, None)
            self.plotters.append(p)

    def plotFeet(self):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.canvas.manager.set_window_title("Receding feet")
        feetMin = np.array([np.inf] * 3)
        feetMax = np.array([-np.inf] * 3)
        for k, cid in enumerate(self.contactIds):
            for t0, p in enumerate(self.plotters):
                if t0 % 50:
                    continue
                vanishingPlot(t0, p.foottraj[:, :3], axs=axs[:, 0])
                m, M = np.min(p.foottraj[:, :3], 0), np.max(p.foottraj[:, :3], 0)
                feetMin = np.min([m, feetMin], 0)
                feetMax = np.max([M, feetMax], 0)
                print(m, M, feetMin, feetMax)
            for iax, ax in enumerate(axs[:, 0]):
                ax.set_xlim([0, len(p.foottraj) + len(self.plotters)])
                ax.set_ylim(feetMin[iax], feetMax[iax])
                print("set %s %s:%s" % (iax, feetMin[iax], feetMax[iax]))
            break
