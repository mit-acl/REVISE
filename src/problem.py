import math
import logging

import numpy as np
from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def mean_dynamics(self, x, u):
        pass

    @abstractmethod
    def stochastic_dynamics(self, x, u):
        pass

    @abstractmethod
    def stochastic_dynamics_known_w(self, x, u, w):
        pass

    @abstractmethod
    def get_A(self):
        pass

    @abstractmethod
    def get_B(self):
        pass

    @abstractmethod
    def get_G(self):
        pass

    @abstractmethod
    def get_state_constraints(self):
        pass
    
    @abstractmethod
    def get_control_constraints(self):
        pass

    @abstractmethod
    def get_mean_W(self, x_traj):
        pass

    @abstractmethod
    def get_cov_W(self, x_i, x_j):
        pass

    def get_block_A(self, n_steps):
        state_size = self.state_size
        block_A_mat = np.zeros(((n_steps + 1)*state_size, state_size))

        block_A_mat[:state_size, :state_size] = np.eye(state_size)
        for i in range(1, n_steps + 1):
            block_A_mat[state_size*i:state_size*(i+1), :] = self.get_A()@block_A_mat[state_size*(i-1):state_size*i, :]

        return block_A_mat

    def get_block_B(self, n_steps):
        state_size = self.state_size
        control_size = self.control_size

        block_B_mat = np.zeros(((n_steps + 1)*state_size, n_steps*control_size))
        for i in range(1, n_steps + 1):
            block_B_mat[state_size*i:state_size*(i+1), :] = self.get_A()@block_B_mat[state_size*(i-1):state_size*i, :]
            block_B_mat[state_size*i:state_size*(i+1), control_size*(i-1):control_size*i] = self.get_B()

        return block_B_mat

    def get_block_G(self, n_steps):
        state_size = self.state_size
        disturbance_size = self.disturbance_size

        block_G_mat = np.zeros(((n_steps+1)*state_size, (n_steps+1)*disturbance_size))
        for i in range(1, n_steps+1):
            block_G_mat[state_size*i:state_size*(i+1), :] = self.get_A()@block_G_mat[state_size*(i-1):state_size*i, :]
            block_G_mat[state_size*i:state_size*(i+1), disturbance_size*(i-1):disturbance_size*i] = self.get_G()

        return block_G_mat

    def get_block_W(self, x_traj, n_steps):
        disturbance_size = self.disturbance_size
        block_W_mat = np.zeros(((n_steps+1)*disturbance_size, (n_steps+1)*disturbance_size))

        for i in range(1, n_steps+1):
            for j in range(i, n_steps+1):
                x_i = x_traj[:, i-1]
                x_j = x_traj[:, j-1]
                block_W_ij = self.get_cov_W(x_i, x_j) 
                block_W_mat[i*disturbance_size:(i+1)*disturbance_size, j*disturbance_size:(j+1)*disturbance_size] = block_W_ij
        block_W_mat = np.triu(block_W_mat)
        block_W_mat = block_W_mat + block_W_mat.T - np.diag(np.diag(block_W_mat))

        return block_W_mat

    def get_A_B_G_W(self, x_0, u_traj):
        n_steps = u_traj.shape[1]
        x_traj = np.zeros((self.state_size, n_steps + 1))
        x_traj[:, 0] = x_0
        for i in range(n_steps):
            x_traj[:, i+1] = self.mean_dynamics(x_traj[:, i], u_traj[:, i])

        A = self.get_block_A(n_steps)
        B = self.get_block_B(n_steps)
        G = self.get_block_G(n_steps)
        W = self.get_block_W(x_traj, n_steps)
        mean_W = self.get_mean_W(x_traj)

        return A, B, G, mean_W, W

