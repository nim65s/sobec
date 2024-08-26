import crocoddyl
from numpy.testing import assert_allclose
import numpy as np
import pinocchio as pin
from tqdm import tqdm

class CallbackNumDiff(crocoddyl.CallbackAbstract):
    def __init__(self, eps=1e-6):
        crocoddyl.CallbackAbstract.__init__(self)
        self.eps = eps

    def __call__(self, solver):
        print("Computing the derivatives of the dynamics and residuals...")
        for i, (x, u) in tqdm(enumerate(zip(solver.xs, solver.us))):
            dam = solver.problem.runningModels[i].differential
            dam_data_fd = dam.createData()
            dam_data_fd2 = dam.createData()
            
            costs_names = dam.costs.costs.todict().keys()

            Fx = np.zeros((dam.pinocchio.nv, 2*dam.pinocchio.nv))
            Fu = np.zeros((dam.pinocchio.nv, dam.actuation.nu))
            Rxs = {k: np.zeros((dam.costs.costs.todict()[k].cost.residual.nr, 2*dam.pinocchio.nv)) for k in costs_names}
            Rus = {k: np.zeros((dam.costs.costs.todict()[k].cost.residual.nr, dam.actuation.nu)) for k in costs_names}

            q, v = x[:dam.pinocchio.nq], x[dam.pinocchio.nq:]
            q_eps = np.zeros(dam.pinocchio.nv)
            v1, v2 = v.copy(), v.copy()
            u1, u2 = u.copy(), u.copy()
            for j in range(dam.pinocchio.nv):
                q_eps[j] = self.eps
                q1 = pin.integrate(dam.pinocchio, q, q_eps)
                q2 = pin.integrate(dam.pinocchio, q, -q_eps)
                dam.calc(dam_data_fd, np.concatenate([q1, v]), u)
                dam.calc(dam_data_fd2, np.concatenate([q2, v]), u)
                xout_fd = dam_data_fd.xout
                xout_fd2 = dam_data_fd2.xout
                Fx[:, j] = (xout_fd - xout_fd2) / (2*self.eps)
                for k in costs_names:
                    res_fd = dam_data_fd.costs.costs.todict()[k].residual.r
                    res_fd2 = dam_data_fd2.costs.costs.todict()[k].residual.r
                    Rxs[k][:, j] = (res_fd - res_fd2) / (2*self.eps)
                q_eps[j] = 0
            for j in range(dam.pinocchio.nv):
                v1[j] += self.eps
                v2[j] -= self.eps
                dam.calc(dam_data_fd, np.concatenate([q, v1]), u)
                dam.calc(dam_data_fd2, np.concatenate([q, v2]), u)
                xout_fd = dam_data_fd.xout
                xout_fd2 = dam_data_fd2.xout
                Fx[:, j + dam.pinocchio.nv] = (xout_fd - xout_fd2) / (2*self.eps)
                for k in costs_names:
                    res_fd = dam_data_fd.costs.costs.todict()[k].residual.r.copy()
                    res_fd2 = dam_data_fd2.costs.costs.todict()[k].residual.r.copy()
                    Rxs[k][:, j + dam.pinocchio.nv] = (res_fd - res_fd2) / (2*self.eps)
                v1[j] -= self.eps
                v2[j] += self.eps
            for j in range(dam.actuation.nu):
                u1[j] += self.eps
                u2[j] -= self.eps
                dam.calc(dam_data_fd, x, u1)
                dam.calc(dam_data_fd2, x, u2)
                xout_fd = dam_data_fd.xout
                xout_fd2 = dam_data_fd2.xout
                Fu[:, j] = (xout_fd - xout_fd2) / (2*self.eps)
                for k in costs_names:
                    res_fd = dam_data_fd.costs.costs.todict()[k].residual.r.copy()
                    res_fd2 = dam_data_fd2.costs.costs.todict()[k].residual.r.copy()
                    Rus[k][:, j] = (res_fd - res_fd2) / (2*self.eps)
                u1[j] -= self.eps
                u2[j] += self.eps
            dam.calc(dam_data_fd, x, u)
            dam.calcDiff(dam_data_fd, x, u)

            assert_allclose(dam_data_fd.Fx, Fx, atol=2*np.sqrt(self.eps))
            assert_allclose(dam_data_fd.Fu, Fu, atol=2*np.sqrt(self.eps))
            for k in costs_names:
                assert_allclose(dam_data_fd.costs.costs.todict()[k].residual.Rx, Rxs[k], atol=2*np.sqrt(self.eps))
                assert_allclose(dam_data_fd.costs.costs.todict()[k].residual.Ru, Rus[k], atol=2*np.sqrt(self.eps))
        print("All the derivatives of the dynamics and residuals are correct!")