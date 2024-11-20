import logging
from enum import Enum

import numpy as np
import cvxpy as cp
import scipy
import mosek

from graph import *
from math_utils import *

class EdgeController(Enum):
    """
    Enum representing choice of edge controller.
    """
    BASELINE = 1
    ROBUST_SIGMA_POINT = 2

def get_E_k_x(k, n_states, state_size):
    """
    Get :math:`E_k^x` such that :math:`E_k^x \mathbf{X} = \mathbf{x_k}`.

    Parameters
    -------------
        k: int
            Timestep chosen so that :math:`E_k^x \mathbf{X} = \mathbf{x_k}`
        n_states: int
            Number of states in `\mathbf{X}`
        state_size: int
            Size of state vector

    Returns
    ----------
        E_k: :math:`E_k^x` such that :math:`E_k^x \mathbf{X} = \mathbf{x_k}`
    """
    E_k = np.zeros((state_size, n_states*state_size))
    E_k[:, state_size*k:state_size*(k+1)] = np.eye(state_size)
    return E_k

def get_E_k_u(k, n_controls, control_size):
    """
    Get :math:`E_k^u` such that :math:`E_k^u \mathbf{U} = \mathbf{u_k}`.
    Wrapper around :func:`quadrotor_experiment.get_E_k_x`
    """
    return get_E_k_x(k, n_controls, control_size)

def get_lower_triangular_L(n_states, n_controls, state_size, control_size):
    """
    Get lower triangular matrix of cvxpy variables, of size (`n_controls * control_size`) x (`n_states * state_size`).

    Parameters
    -----------
        n_states: int
            Number of states in trajectory
        n_controls: int
            Number of control variables in trajectory. Should be
            equal to `n_states` - 1
        state_size: int
            Size of state vector
        control_size: int
            Size of control vector
    """
    L_list = []
    for n_control in range(n_controls):
        for _ctrl in range(control_size):
            L_i = []
            for n_state in range(n_states):
                if n_state <= n_control:
                    for _ in range(state_size):
                        L_i.append(cp.Variable())
                else:
                    for _ in range(state_size):
                        L_i.append(0)
            L_list.append(L_i)
    L = cp.bmat(L_list)
    return L

def robust_sigma_point_covariance_steering(problem, x_0, A, B, G, mean_W, cov_W, As, Bs, Gs, mean_Ws, cov_Ws, sigma_points, cov_0, x_f):
    """
    Robust sigma point steering algorithm for steering in a Gaussian random
    field.

    This algorithm corresponds to Problem IV.1 in the paper. See section IV
    of the paper for additional details.

    Parameters
    -----------
        problem: Problem
            Problem class with dynamics and constraints
        A: np.ndarray
            State transition matrix, such that the open loop system dynamics
            in block-matrix notation are given by :math:`\mathbf{X} = A\mathbf{x_0} + B\mathbf{U} + G\mathbf{W}`
        B: np.ndarray
            Control matrix in state-space dynamics, such that the open loop system dynamics in block-matrix notation are given by :math:`\mathbf{X} = A\mathbf{x_0} + B\mathbf{U} + G\mathbf{W}`
        G: np.ndarray
            Disturbance matrix in state-space dynamics, such that the open loop system dynamics in block-matrix notation are given by :math:`\mathbf{X} = A\mathbf{x_0} + B\mathbf{U} + G\mathbf{W}`
        mean_W: np.ndarray
            Mean of the Gaussian random field along a nominal trajectory
        cov_W: np.ndarray
            Covariance of the Gaussian random field along a nominal trajectory
        As: list
            List of state-transition (:math:`A`) matrices for each sigma point
        Bs: list
            List of control (:math:`B`) matrices for each sigma point
        Gs: list
            List of disturbance (:math:`G`) matrices for each sigma point
        mean_Ws: list
            List of mean disturbance trajectories starting at each sigma point,
            when initial guess control is applied
        cov_Ws: list
            List of covariances of disturbance trajectories starting at each
            sigma point, when initial guess control is applied
        sigma_points: list
            List of initial state for each sigma point
        cov_0: np.ndarray
            Covariance of the initial state distribution
        x_f: np.ndarray
            Desired mean of the final state distribution

    Returns
    ---------
        success: bool
            True if MOSEK can solve the covariance steering problem, otherwise False
        X_traj: np.ndarray
            State trajectory solution to covariance steering problem (or none if no solution)
        U_traj: np.ndarray
            Nominal control trajectory solution to covariance steering problem (or None if no
            solution)
        P_traj: np.ndarray
            State covariance trajectory solution to covariance steering problem (or None if no solution). Computed by approximating the mixture of Gaussians given by the mixture of sigma points as a Gaussian at each timestep.
        K: np.ndarray
            Control feedback gain solution to covariance steering problem (or None if no 
            solution)
        cov_f: np.ndarray
            Final state covariance (or None if no solution to covariance steering problem). Computed as an over-approximation of the state covariance, equivalent
            to the worst-case contribution to the state covariance over all sigma points. See Section IV in the paper for further details.
    """
    # Useful vars
    state_size = problem.state_size
    control_size = problem.control_size
    n_states = As[0].shape[0]//state_size
    n_controls = n_states - 1

    # Initialize U_bar and L (control decision variables)
    U_bar = cp.Variable((n_controls*control_size,))
    L = get_lower_triangular_L(n_states, n_controls, state_size, control_size)

    # Get X_bar and S for x_0, P_0 and introduce chance constraints
    X_bar = (A@x_0.reshape(state_size, 1)).reshape(n_states*state_size,) + B@U_bar + G@mean_W
    S = ensure_psd(A@cov_0@A.T + G@cov_W@G.T)
    S_12 = scipy.linalg.cholesky(S, lower=True)
    constraints = []

    # Same constraint formulation as baseline
    # State chance constraints
    for k in range(n_states):
        E_k = get_E_k_x(k, n_states, state_size)
        alpha_x_mat, beta_x_mat, epsilon_x_mat = problem.get_state_constraints()
        for j in range(epsilon_x_mat.shape[0]):
            alpha_j = alpha_x_mat[j, :]
            beta_j = beta_x_mat[j]
            epsilon_j = epsilon_x_mat[j]
            
            state_chance_constraint = scipy.stats.norm.ppf(1-epsilon_j)*cp.norm(S_12.T@(np.eye(n_states*state_size) + B@L).T@E_k.T@alpha_j) + alpha_j.T@E_k@X_bar <= beta_j
            constraints.append(state_chance_constraint)

    # Control chance constraints
    for k in range(n_controls):
        E_k = get_E_k_u(k, n_controls, control_size)
        alpha_u_mat, beta_u_mat, epsilon_u_mat = problem.get_control_constraints()
        for j in range(epsilon_u_mat.shape[0]):
            alpha_j = alpha_u_mat[j, :]
            beta_j = beta_u_mat[j]
            epsilon_j = epsilon_u_mat[j]
            
            control_chance_constraint = scipy.stats.norm.ppf(1-epsilon_j)*cp.norm(S_12.T@L.T@E_k.T@alpha_j) + alpha_j.T@E_k@U_bar <= beta_j

            constraints.append(control_chance_constraint)

    # Get X_bar and S for all sigma points, as well as contribution
    # to overall state covariance.
    X_bars = []
    X_bars_mean_W = []
    Ss = []
    Ss_mean_W = []
    cov_contributions = []

    for i in range(len(sigma_points)):
        # X_bar for each sigma point depends on state feedback, because
        # sigma point trajectory doesn't start at mean state x_0
        X_bar_i = (np.eye(n_states*state_size) + Bs[i]@L)@(As[i]@(sigma_points[i].reshape(state_size,) - x_0.reshape(state_size,)) + Gs[i]@(mean_Ws[i] - mean_W)) + X_bar
        X_bar_i_mean_W = (np.eye(n_states*state_size) + Bs[i]@L)@(As[i]@(sigma_points[i].reshape(state_size,) - x_0.reshape(state_size,))) + X_bar

        # Get open loop covariance for each sigma point (S_i)
        S_i = ensure_psd(Gs[i]@cov_Ws[i]@Gs[i].T)
        S_i_mean_W = ensure_psd(G@cov_W@G.T)

        A_x0 = As[i]@(sigma_points[i].reshape(state_size,1)-x_0.reshape(state_size,1))
        A_i_mean_W = ensure_psd(A_x0.reshape(-1, 1)@A_x0.reshape(1, -1))
        A_x0_Gw = A_x0 + Gs[i]@(mean_Ws[i]-mean_W).reshape(-1, 1)
        A_i = ensure_psd(A_x0_Gw.reshape(-1, 1)@A_x0_Gw.reshape(1, -1))
    
        # S_i + A_i = contribution to overall state covariance,
        # equal to S_i in Equation 11 in the paper (yes, slightly
        # confusing)
        cov_contributions.append(S_i + A_i)
        cov_contributions.append(S_i_mean_W + A_i_mean_W)
        X_bars.append(X_bar_i)
        X_bars_mean_W.append(X_bar_i_mean_W)
        Ss.append(S_i)
        Ss_mean_W.append(S_i_mean_W)

    # Get objective as described in Problem IV.1 in the paper
    obj_mat = cp.Variable((state_size, state_size))
    obj = cp.lambda_max(obj_mat)

    E_N = get_E_k_x(n_states-1, n_states, state_size)
    for cov_contribution in cov_contributions:
        cov_contribution_12 = scipy.linalg.cholesky(cov_contribution, lower=True)
        constraints.append(cp.bmat([[np.eye(n_states*state_size, n_states*state_size), cov_contribution_12.T@(np.eye(n_states*state_size) + B@L).T@E_N.T],
                                    [E_N@(np.eye(n_states*state_size) + B@L)@cov_contribution_12, obj_mat]]) >> 0)

    # Get final mean state constraint. Assuming mean disturbance field is
    # a linear function of the state, this is equivalent to the
    # constraint given in Problem IV.1 and is much more efficient.
    constraints.append(E_N@X_bar == x_f)

    # Solve problem with MOSEK and unpack state and control
    # trajectories.
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        if prob.status == 'optimal':
            success = True
            K = L.value@np.linalg.inv(np.eye(n_states*state_size) + B@L.value)
            X_traj = X_bar.value.reshape(n_states, state_size).T
            U_traj = U_bar.value.reshape(n_controls, control_size).T
            P_traj = np.zeros((state_size, state_size, n_states))

            # Can approximate state covariance by fitting a Gaussian to
            # the mixture of Gaussians given by the sigma points
            for k in range(n_states):
                E_k = get_E_k_x(k, n_states, state_size)
                cov_k = np.zeros_like(E_k@S@E_k.T)
                for i in range(len(sigma_points)):
                    wt = 1/(len(sigma_points))
                    cov_k_i = 0.5*wt*E_k@(np.eye(n_states*state_size) + B@L.value)@Ss[i]@(np.eye(n_states*state_size) + B@L.value).T@E_k.T
                    cov_k_i += 0.5*wt*E_k@(np.eye(n_states*state_size) + B@L.value)@Ss_mean_W[i]@(np.eye(n_states*state_size) + B@L.value).T@E_k.T
                    cov_k_i += 0.5*wt*(E_k@X_bars[i].value.reshape(-1, 1)@X_bars[i].value.reshape(1, -1)@E_k.T)
                    cov_k_i += 0.5*wt*(E_k@X_bars_mean_W[i].value.reshape(-1, 1)@X_bars_mean_W[i].value.reshape(1, -1)@E_k.T)
                    cov_k_i -= wt*(E_k@X_bar.value.reshape(-1, 1)@X_bar.value.reshape(1, -1)@E_k.T)
                    cov_k += cov_k_i

                P_traj[:, :, k] = cov_k

                # Final state covariance is over-approximated by the 
                # value of the objective, as described in Problem IV.1
                cov_f = obj_mat.value
            return success, X_traj, U_traj, P_traj, K, cov_f
        else: # Problem is infeasible 
            success = False
            return success, -1, -1, -1, -1, -1

    except: # MOSEK error, or too many iterations
        success = False
        return success, -1, -1, -1, -1, -1

def baseline_covariance_steering(problem, A, B, G, mean_W, cov_W, x_0, cov_0, x_f):
    """
    Baseline covariance steering algorithm for steering in a Gaussian random field. Combines the
    coverage-maximizing objective from Aggarwal & How 2024 with the approach to covariance
    steering in a GRF from Ridderhof & Tsiotras 2022.

    This algorithm corresponds to Problem III.1 in the paper. See section III of the paper for
    additional details.

    Parameters
    ------------
        problem: Problem
            Problem class with dynamics and constraints
        A: np.ndarray
            State transition matrix, such that the open loop system dynamics in block-matrix
            notation are given by :math:`X = Ax_0 + BU + GW`
        B: np.ndarray
            Control matrix in state-space dynamics, such that :math:`X = AX_0 + BU + GW`
        G: np.ndarray
            Disturbance matrix in state-space dynamics, such that :math:`X + AX_0 + BU + GW`
        mean_W: np.ndarray
            Mean of the Gaussian random field along x_nominal
        cov_W: np.ndarray
            Covariance of the Gaussian random field along x_nominal
        x_0: np.ndarray
            Mean of the initial state distribution
        cov_0: np.ndarray
            Covariance of the initial state distribution
        x_f: np.ndarray
            Desired mean of the final state distribution

    Returns
    ---------
        success: bool
            True if MOSEK can solve the covariance steering problem, otherwise False
        X_traj: np.ndarray
            State trajectory solution to covariance steering problem (or none if no solution)
        U_traj: np.ndarray
            Nominal control trajectory solution to covariance steering problem (or None if no
            solution)
        P_traj: np.ndarray
            State covariance trajectory solution to covariance steering problem (or None if
            no solution)
        K: np.ndarray
            Control feedback gain solution to covariance steering problem (or None if no 
            solution)
        cov_f: np.ndarray
            Final state covariance (or None if no solution to covariance steering problem)
    """
    # Useful vars
    state_size = problem.state_size
    control_size = problem.control_size
    n_states = A.shape[0]//state_size
    n_controls = n_states - 1

    # Initialize U_bar and L (control decision variables)
    U_bar = cp.Variable((n_controls*control_size,))
    L = get_lower_triangular_L(n_states, n_controls, state_size, control_size)
    
    # Initialize X_bar (mean state) and S (open loop state covariance)
    X_bar = (A@x_0.reshape(state_size, 1)).reshape(n_states*state_size,) + B@U_bar + G@mean_W
    S = ensure_psd(A@cov_0@A.T + G@cov_W@G.T)
    S_12 = scipy.linalg.cholesky(S, lower=True)

    # Construct coverage-maximizing objective
    E_N = get_E_k_x(n_states-1, n_states, state_size)
    cov_obj = cp.norm((E_N@(np.eye(n_states*state_size) + B@L)@S_12).T)
    obj = cp.Minimize(cov_obj)
    constraints = []

    # Final state constraint
    constraints.append(E_N@X_bar == x_f)

    # State chance constraints
    for k in range(n_states):
        E_k = get_E_k_x(k, n_states, state_size)
        alpha_x_mat, beta_x_mat, epsilon_x_mat = problem.get_state_constraints()
        for j in range(epsilon_x_mat.shape[0]):
            alpha_j = alpha_x_mat[j, :]
            beta_j = beta_x_mat[j]
            epsilon_j = epsilon_x_mat[j]
            
            state_chance_constraint = scipy.stats.norm.ppf(1-epsilon_j)*cp.norm(S_12.T@(np.eye(n_states*state_size) + B@L).T@E_k.T@alpha_j) + alpha_j.T@E_k@X_bar <= beta_j
            constraints.append(state_chance_constraint)

    # Control chance constraints
    for k in range(n_controls):
        E_k = get_E_k_x(k, n_controls, control_size)
        alpha_u_mat, beta_u_mat, epsilon_u_mat = problem.get_control_constraints()
        for j in range(epsilon_u_mat.shape[0]):
            alpha_j = alpha_u_mat[j, :]
            beta_j = beta_u_mat[j]
            epsilon_j = epsilon_u_mat[j]
            
            control_chance_constraint = scipy.stats.norm.ppf(1-epsilon_j)*cp.norm(S_12.T@L.T@E_k.T@alpha_j) + alpha_j.T@E_k@U_bar <= beta_j
            constraints.append(control_chance_constraint)

    # Solve covariance steering problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.MOSEK)
        if prob.status == 'optimal':
            success = True
            # Extract K (feedback gain) from L and B
            K = L.value@np.linalg.inv(np.eye(n_states*state_size) + B@L.value)

            # Final state covariance (closed-loop)
            cov_f = E_N@(np.eye(n_states*state_size) + B@L.value)@S@(np.eye(n_states*state_size) + B@L.value).T@E_N.T
            # Get state trajectory, nominal control trajectory, and state covariance trajectory
            X_traj = X_bar.value.reshape(n_states, state_size).T
            U_traj = U_bar.value.reshape(n_controls, control_size).T
            P_traj = np.zeros((state_size, state_size, n_states))
            for k in range(n_states):
                E_k = get_E_k_x(k, n_states, state_size)
                cov_k = E_k@(np.eye(n_states*state_size) + B@L.value)@S@(np.eye(n_states*state_size) + B@L.value).T@E_k.T
                P_traj[:, :, k] = cov_k

            return success, X_traj, U_traj, P_traj, K, cov_f
        else: # Probably need more iterations to converge properly
            success = False
            return success, None, None, None, None, None

    except: # MOSEK error
        success = False
        return success, None, None, None, None, None

def zero_control_rollout(problem, x_0, n_states):
    """
    Roll out mean state trajectory assuming zero control.
    Wrapper around :func:`quadrotor_experiment.nominal_rollout`
    """
    n_controls = n_states - 1
    u_traj = np.zeros((problem.control_size, n_controls))
    return nominal_rollout(problem, x_0, u_traj, n_states), u_traj

def nominal_rollout(problem, x_0, u_traj, n_states):
    """
    Roll out mean state trajectory of a system given an initial 
    state and nominal control trajectory.

    Parameters
    -----------
        problem: Problem
            Problem class with associated dynamics to roll out
        x_0: np.ndarray
            Initial mean state
        u_traj: np.ndarray
            Nominal control trajectory
        n_states: int
            Number of states to roll out

    Returns
    ---------
        x_traj: np.ndarray
            Mean state trajectory
    """
    n_controls = n_states - 1
    x_traj = np.zeros((problem.state_size, n_states))
    x_traj[:, 0] = x_0
    for i in range(n_controls):
        x_traj[:, i+1] = problem.mean_dynamics(x_traj[:, i], u_traj[:, i])
    return x_traj 

def mean_steering(problem, x_0, x_f, n_states):
    """
    Initialize a feasible mean state trajectory, subject
    to dynamics constraints but not to state or control
    chance constraints. Assumes disturbance is a linear
    function of the state. Minimizes control usage.

    Parameters
    ------------
        problem: Problem
            Problem class with associated dynamics to compute
            mean trajectory for
        x_0: np.ndarray
            Initial state
        x_f: np.ndarray
            Final state
        n_states: int
            Number of states in trajectory

    Returns
    -----------
        success: bool
            True if SCS can find a valid trajectory, False otherwise
        u_traj: np.ndarray
            Control trajectory used to steer from x_0 to x_f (None if not success)
    """
    state_size = problem.state_size
    control_size = problem.control_size
    n_controls = n_states - 1

    u_traj = cp.Variable((control_size, n_controls))
    x_traj = cp.Variable((state_size, n_states))

    constraints = []
    constraints.append(x_traj[:, 0] == x_0)
    constraints.append(x_traj[:, -1] == x_f)

    # Dynamics constraint
    for i in range(n_states-1):
        disturbance = cp.Variable((2,))
        constraints.append(disturbance[0] == problem.get_mean_disturbance_at_location(x_traj[0, i], x_traj[1, i])[0])
        constraints.append(disturbance[1] == problem.get_mean_disturbance_at_location(x_traj[0, i], x_traj[1, i])[1])
        constraints.append(x_traj[:, i+1] == problem.A@x_traj[:, i] + problem.B@u_traj[:, i] + (problem.G@disturbance).reshape(6,))

    obj = 0
    for i in range(n_controls):
        obj += cp.quad_form(u_traj[:, i], np.eye(control_size))
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(verbose=False, solver=cp.SCS)
        if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
            return True, u_traj.value
        else:
            return False, None
    except:
        return False, None

def select_sigma_points(x_0, P_0):
    """
    Select 2 * `state_size` symmetrically distributed sigma points 
    on the :math:`\sqrt{state\_size}` th covariance contour of
    a Gaussian state distribution.

    See Julier & Uhlmann 2004 for further details.

    Parameters
    ------------
        x_0: np.ndarray
            State mean
        P_0: np.ndarray
            State covariance

    Returns
    --------
        sigma_points: list
            List of sigma points
    """
    state_size = x_0.shape[0]

    sigma_points = []

    S = scipy.linalg.cholesky(P_0, lower=True)

    for i in range(state_size):
        sigma_points.append(x_0 + np.sqrt(state_size)*S[:, i])

    for i in range(state_size):
        sigma_points.append(x_0 - np.sqrt(state_size)*S[:, i])

    return sigma_points 

def robust_sigma_point_edge_controller(problem, x_0, P_0, x_f, n_states):
    """
    Get a belief roadmap edge from :math:`\mathcal{N}(\mathbf{x_0}, P_0)`
    to a distribution with mean :math:`\mathbf{x_f}`, if such an edge
    exists, using :func:`robust_sigma_point_covariance_steering` as
    the edge controller.

    Parameters
    ------------
        problem: Problem
            Problem class with associated dynamics and constraints
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        x_f: np.ndarray
            Final state mean
        n_states: int
            State trajectory length

    Returns
    ---------
        success: bool
            True iff valid edge found
        x_traj: np.ndarray
            Mean state trajectory along edge, or None if no edge found
        u_traj: np.ndarray
            Nominal control trajectory along edge, or None if no edge found
        cov_traj: np.ndarray
            State covariance trajectory along edge, or None if no edge found
        K: np.ndarray
            State feedback gain along edge, or None if no edge found
        cov_f: np.ndarray
            Final state covariance, or None if no edge found
    """
    n_controls = n_states - 1

    # Select sigma points
    sigma_points = select_sigma_points(x_0, P_0)

    # Rollout each sigma point
    success, u_nominal = mean_steering(problem, x_0, x_f, n_states)
    if not success:
        logging.warning("Mean steering failed, initializing with zero control...")
        _, u_nominal = zero_control_rollout(problem, x_0, n_states)

    # Linearize around each sigma point
    As, Bs, Gs, mean_Ws, cov_Ws = [], [], [], [], []
    for i in range(len(sigma_points)):
        A, B, G, mean_W, cov_W = problem.get_A_B_G_W(sigma_points[i], u_nominal)
        As.append(A)
        Bs.append(B)
        Gs.append(G)
        mean_Ws.append(mean_W)
        cov_Ws.append(cov_W)
    A, B, G, mean_W, cov_W = problem.get_A_B_G_W(x_0, u_nominal)

    # Steer
    success, x_traj, u_traj, cov_traj, K, cov_f = robust_sigma_point_covariance_steering(problem, x_0, A, B, G, mean_W, cov_W, As, Bs, Gs, mean_Ws, cov_Ws, sigma_points, P_0, x_f)

    return success, x_traj, u_traj, cov_traj, K, cov_f

def baseline_edge_controller(problem, x_0, P_0, x_f, n_states):
    """
    Get a belief roadmap edge from :math:`\mathcal{N}(\mathbf{x_0}, P_0)`
    to a distribution with mean :math:`\mathbf{x_f}`, if such an edge
    exists, using :func:`baseline_edge_controller` as the edge controller.

    Parameters
    ------------
        problem: Problem
            Problem class with associated dynamics and constraints
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        x_f: np.ndarray
            Final state mean
        n_states: int
            State trajectory length

    Returns
    ---------
        success: bool
            True iff valid edge found
        x_traj: np.ndarray
            Mean state trajectory along edge, or None if no edge found
        u_traj: np.ndarray
            Nominal control trajectory along edge, or None if no edge found
        cov_traj: np.ndarray
            State covariance trajectory along edge, or None if no edge found
        K: np.ndarray
            State feedback gain along edge, or None if no edge found
        cov_f: np.ndarray
            Final state covariance, or None if no edge found
    """

    # Do a rollout with zero control and do an initial solve of the problem
    n_controls = n_states - 1
    success, u_nominal = mean_steering(problem, x_0, x_f, n_states)
    if not success:
        logging.warning("Mean steering, initializing with zero control...")
        _, u_nominal = zero_control_rollout(problem, x_0, n_states)
    A, B, G, mean_W, cov_W = problem.get_A_B_G_W(x_0, u_nominal)
    success, x_traj, u_traj, cov_traj, K, cov_f = baseline_covariance_steering(problem, A, B, G, mean_W, cov_W, x_0, P_0, x_f)

    return success, x_traj, u_traj, cov_traj, K, cov_f
