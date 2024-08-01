import crocoddyl
import numpy as np
import sobec

class ActuationModelMatrix(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, nu, act_matrix):
        super(ActuationModelMatrix, self).__init__(state, nu)
        self.ntau = state.nv
        assert(act_matrix.shape[0] == self.ntau)
        assert(act_matrix.shape[1] == nu)

        self.act_matrix = act_matrix
        
    def calc(self, data, x, u):
        data.tau = self.act_matrix @ u

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        
        # wrt u
        data.dtau_du = self.act_matrix # np.transpose(act_matrix)

        # wrt x
        data.dtau_dx[:, :] = np.zeros((self.ntau, self.state.ndx))

    def createData(self):
        data = ActuationDataMatrix(self)
        return data
    
class ActuationDataMatrix(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        super(ActuationDataMatrix, self).__init__(model)
