import math
import logging

import numpy as np
from src.problem import Problem

class Quadrotor2DGRF(Problem):
    def __init__(self, dt=0.1, obstacles="none"):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0, dt**2/2, 0],
                           [0, 1, 0, dt, 0, dt**2/2],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [dt, 0],
                           [0, dt]])
        self.G = np.array([[dt, 0],
                           [0, dt],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0]])
        self.wind_mean = np.array([0, 0])
        self.wind_var = 0.2
        self.max_velocity = 10
        self.max_acceleration = 100
        self.sample_radius_lims = [np.array([-2, -2, -2.5, -2.5, -25, -25]),
                                   np.array([2, 2, 2.5, 2.5, 25, 25])]
        self.proximity_weights = np.array([1/2, 1/2, 1/2.5, 1/2.5, 1/25, 1/25])
        self.min_position = 0
        self.max_position = 10
        self.state_size = 6
        self.control_size = 2
        self.disturbance_size = 2

        self.obstacles = obstacles
        self.grf_cov = self.grf_cov()
        x_circle, y_circle = self.generate_circle_wind()
        self.grf_mean = self.xy_to_wind_profile(x_circle, y_circle)
        self.grf_vel = self.sample_grf(n_samples=1)
        self.grf_vel = self.grf_vel.reshape(-1,)
        self.wind_profile = self.grf_vel

    def mean_dynamics(self, x, u):
        wind = self.get_wind_at_location(x[0], x[1], self.grf_mean)
        return self.A@x + self.B@u + self.G@wind

    def stochastic_dynamics(self, x, u):
        wind = self.get_wind_at_location(x[0], x[1], self.wind_profile)
        return self.A@x + self.B@u + self.G@wind
            
    def stochastic_dynamics_known_w(self, x, u, wind_profile):
        wind = self.get_wind_at_location(x[0], x[1], wind_profile)
        return self.A@x + self.B@u + self.G@wind

    def get_A(self):
        return self.A
        
    def get_B(self):
        return self.B
        
    def get_G(self):
        return self.G

    def get_state_constraints(self):
        beta_x_mat = np.array([self.max_velocity, 
                               self.max_velocity,
                               self.max_velocity,
                               self.max_velocity,
                               self.max_acceleration,
                               self.max_acceleration,
                               self.max_acceleration,
                               self.max_acceleration,
                               self.min_position,
                               self.min_position,
                               self.max_position,
                               self.max_position])
        epsilon_x_mat = np.array([(1-0.9973)/2, 
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2,
                                  (1-0.9973)/2])
        alpha_x_mat = np.array([[0, 0, 1, 1, 0, 0],
                                [0, 0, 1, -1, 0, 0],
                                [0, 0, -1, 1, 0, 0],
                                [0, 0, -1, -1, 0, 0],
                                [0, 0, 0, 0, 1, 1],
                                [0, 0, 0, 0, 1, -1],
                                [0, 0, 0, 0, -1, 1],
                                [0, 0, 0, 0, -1, -1],
                                [-1, 0, 0, 0, 0, 0],
                                [0, -1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0]])
        return alpha_x_mat, beta_x_mat, epsilon_x_mat

    def get_control_constraints(self):
        alpha_u_mat = np.array([])
        epsilon_u_mat = np.array([])
        beta_u_mat = np.array([])
        return alpha_u_mat, beta_u_mat, epsilon_u_mat

    def get_mean_W(self, x_traj):
        disturbance_size = self.disturbance_size
        n_steps = x_traj.shape[1]
        mean_W = np.zeros((n_steps*disturbance_size))
        for i in range(n_steps):
            mean_W[i*disturbance_size:(i+1)*disturbance_size] = self.get_wind_at_location(x_traj[0, i], x_traj[1, i], self.grf_mean)
        return mean_W

    def get_cov_W(self, x_i, x_j):
        # Get covariance of w(x_i) and w(x_j)
        cov = self.get_wind_covariance_at_locations(x_i, x_j, self.grf_cov)
        return cov

    def get_mean_disturbance_at_location(self, x, y):
        return (5-y)/4, (x-5)/4

    def generate_circle_wind(self):
        x_profile = np.zeros((11, 11))
        y_profile = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                x_profile[i, j], y_profile[i, j] = self.get_mean_disturbance_at_location(i, j)
        return x_profile, y_profile

    def xy_to_wind_profile(self, x_profile, y_profile):
        wind_profile = np.zeros(242,)
        for i in range(11):
            for j in range(11):
                wind_profile[22*i + 2*j] = x_profile[i, j]
                wind_profile[22*i + 2*j + 1] = y_profile[i, j]
        return wind_profile

    def resample_grf(self):
        self.grf_vel = self.sample_grf(n_samples=1)
        self.grf_vel = self.grf_vel.reshape(-1,)
        self.wind_profile = self.grf_vel

    def grf_var(self, x_1, x_2):
        if self.obstacles == "none":
            return self.wind_var
        elif self.obstacles == "center":
            if 3 < x_1 and x_1 < 7 and 3 < x_2 and x_2 < 7:
                return 30*self.wind_var
            else:
                return self.wind_var

    def grf_corr(self, x_1, x_2, x_3, x_4):
        if x_1 == x_3 and x_2 == x_4:
            return 1

        corr = (0.3 - (np.sqrt((x_1 - x_3)**2 + (x_2 - x_4)**2))/(10*np.sqrt(2)))
        corr = max(corr, 0)

        assert corr >= 0
        assert corr <= 1

        return corr

    def grf_cov(self):
        cov_mat = np.zeros((242, 242))
        for x_1 in range(11):
            for x_2 in range(11):
                for x_3 in range(x_1+1):
                    for x_4 in range(x_2+1):
                        corr = self.grf_corr(x_1, x_2, x_3, x_4)
                        var_1 = self.grf_var(x_1, x_2)
                        var_2 = self.grf_var(x_3, x_4)
                        cov = corr*np.sqrt(var_1*var_2)
                        cov_mat[22*x_1 + 2*x_2, 22*x_3 + 2*x_4] = cov
                        cov_mat[22*x_3 + 2*x_4, 22*x_1 + 2*x_2] = cov
                        cov_mat[22*x_1 + 2*x_2 + 1, 22*x_3 + 2*x_4 + 1] = cov
                        cov_mat[22*x_3 + 2*x_4 + 1, 22*x_1 + 2*x_2 + 1] = cov
        cov_mat = 0.5*(cov_mat + cov_mat.T) + np.eye(242)*1e-8
        return cov_mat

    def sample_grf(self, n_samples=1):
        return np.random.multivariate_normal(self.grf_mean, self.grf_cov, size=n_samples)

    def get_wind_at_location(self, x_1, x_2, wind_profile):
        x_1 = min(10, max(0, x_1))
        x_2 = min(10, max(0, x_2))
        a = math.ceil(x_1) - x_1
        b = math.ceil(x_2) - x_2
        x_1_low = int(math.floor(x_1))
        x_1_high = int(math.ceil(x_1))
        x_2_low = int(math.floor(x_2))
        x_2_high = int(math.ceil(x_2))

        wind_x = a*b*wind_profile[22*x_1_low + 2*x_2_low]
        wind_x += a*(1-b)*wind_profile[22*x_1_low + 2*x_2_high]
        wind_x += (1-a)*b*wind_profile[22*x_1_high + 2*x_2_low]
        wind_x += (1-a)*(1-b)*wind_profile[22*x_1_high + 2*x_2_high]

        wind_y = a*b*wind_profile[22*x_1_low + 2*x_2_low + 1]
        wind_y += a*(1-b)*wind_profile[22*x_1_low + 2*x_2_high + 1]
        wind_y += (1-a)*b*wind_profile[22*x_1_high + 2*x_2_low + 1]
        wind_y += (1-a)*(1-b)*wind_profile[22*x_1_high + 2*x_2_high + 1]

        return np.array([wind_x, wind_y])

    def get_wind_covariance_at_locations(self, x_i, x_j, wind_cov):
        """
        x_i and x_j are two different states.
        """
        a_i = math.ceil(x_i[0]) - x_i[0]
        b_i = math.ceil(x_i[1]) - x_i[1]
        a_j = math.ceil(x_j[0]) - x_j[0]
        b_j = math.ceil(x_j[1]) - x_j[1]

        x_i_1_low = int(math.floor(x_i[0]))
        x_i_1_high = int(math.ceil(x_i[0]))
        x_i_2_low = int(math.floor(x_i[1]))
        x_i_2_high = int(math.ceil(x_i[1]))

        x_i_1_low = min(10, max(0, x_i_1_low))
        x_i_1_high = min(10, max(0, x_i_1_high))
        x_i_2_low = min(10, max(0, x_i_2_low))
        x_i_2_high = min(10, max(0, x_i_2_high))

        x_j_1_low = int(math.floor(x_j[0]))
        x_j_1_high = int(math.ceil(x_j[0]))
        x_j_2_low = int(math.floor(x_j[1]))
        x_j_2_high = int(math.ceil(x_j[1]))

        x_j_1_low = min(10, max(0, x_j_1_low))
        x_j_1_high = min(10, max(0, x_j_1_high))
        x_j_2_low = min(10, max(0, x_j_2_low))
        x_j_2_high = min(10, max(0, x_j_2_high))

        cov_x_i_wts = [(a_i*b_i, 22*x_i_1_low + 2*x_i_2_low),
                       (a_i*(1-b_i), 22*x_i_1_low + 2*x_i_2_high),
                       ((1-a_i)*b_i, 22*x_i_1_high + 2*x_i_2_low),
                       ((1-a_i)*(1-b_i), 22*x_i_1_high + 2*x_i_2_high)]
        cov_x_j_wts = [(a_j*b_j, 22*x_j_1_low + 2*x_j_2_low),
                       (a_j*(1-b_j), 22*x_j_1_low + 2*x_j_2_high),
                       ((1-a_j)*b_j, 22*x_j_1_high + 2*x_j_2_low),
                       ((1-a_j)*(1-b_j), 22*x_j_1_high + 2*x_j_2_high)]
        cov_y_i_wts = [(a_i*b_i, 22*x_i_1_low + 2*x_i_2_low + 1),
                       (a_i*(1-b_i), 22*x_i_1_low + 2*x_i_2_high + 1),
                       ((1-a_i)*b_i, 22*x_i_1_high + 2*x_i_2_low + 1),
                       ((1-a_i)*(1-b_i), 22*x_i_1_high + 2*x_i_2_high + 1)]
        cov_y_j_wts = [(a_j*b_j, 22*x_j_1_low + 2*x_j_2_low + 1),
                       (a_j*(1-b_j), 22*x_j_1_low + 2*x_j_2_high + 1),
                       ((1-a_j)*b_j, 22*x_j_1_high + 2*x_j_2_low + 1),
                       ((1-a_j)*(1-b_j), 22*x_j_1_high + 2*x_j_2_high + 1)]

        cov_x = 0
        for (wt_i, ind_i) in cov_x_i_wts:
            for (wt_j, ind_j) in cov_x_j_wts:
                cov_x += wt_i*wind_cov[ind_i, ind_j]*wt_j
        cov_y = 0
        for (wt_i, ind_i) in cov_y_i_wts:
            for (wt_j, ind_j) in cov_y_j_wts:
                cov_y += wt_i*wind_cov[ind_i, ind_j]*wt_j

        return np.array([[cov_x, 0],
                         [0, cov_y]])
