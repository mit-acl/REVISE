import os
import pickle
import logging
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt

from src.quadrotor_problem import Quadrotor2DGRF
from src.covariance_steering import *
from src.graph import *
from src.roadmap_generation import *
from src.utils.io_utils import *

def single_query_experimental_trial(random_seed, save_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby):
    """
    Construct two belief roadmaps for a single-query planning experiment, one
    with edge rewiring, and one without. Wrapper around :func:`construct_belief_roadmaps_to_goal_and_rewire`.
    """
    np.random.seed(random_seed)
    rewired_graph, graph = construct_belief_roadmaps_to_goal_and_rewire(save_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby)

def multi_query_experimental_trial(random_seed, save_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby):
    """
    Construct two belief roadmaps for a multi-query planning experiment, one
    with edge rewiring, and one without. Wrapper around :func:`construct_belief_roadmaps_and_rewire`.
    """
    np.random.seed(random_seed)
    rewired_graph, graph = construct_belief_roadmaps_and_rewire(save_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby)

def single_query_experiment(save_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, near_cutoff, max_nearby, n_trials):
    """
    Run multiple trials for single-query experiment in parallel, with baseline
    and robust sigma-point edge controllers.

    Parameters
    ------------
        save_dir: string
            Parent directory to save data to
        problem: Problem
            Problem class with dynamics, constraints, and parameters
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        x_f: np.ndarray
            Final state mean
        P_f: np.ndarray
            Final state covariance
        graph_lims: tuple
            2-tuple with minimum and maximum bounds of state space
        n_states: int
            Trajectory length for edges in belief roadmap
        n_nodes: int
            Number of nodes for belief roadmap
        near_cutoff: float
            Proximity cutoff for neighboring nodes for edge rewiring
        max_nearby: int
            Maximum number of nearby nodes to consider for rewiring
        n_trials: int
            Number of experimental trials to run
    """
    edge_controllers = (EdgeController.BASELINE, EdgeController.ROBUST_SIGMA_POINT)
    trial_processes = []
    for trial in range(n_trials):
        trial_dir = os.path.join(save_dir, f"trial_{trial}")
        os.mkdir(trial_dir)

        pickle.dump(problem, open(os.path.join(trial_dir, f"problem.pkl"), "wb"))
        for edge_controller in edge_controllers:
            trial_process = Process(target=single_query_experimental_trial, args=(trial, trial_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby))
            trial_processes.append(trial_process)
    for trial_process in trial_processes:
        trial_process.start()
    for trial_process in trial_processes:
        trial_process.join()

def multi_query_experiment(save_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, near_cutoff, max_nearby):
    """
    Run multiple trials for multi-query experiment in parallel, with baseline
    and robust sigma-point edge controllers.

    Parameters
    ------------
        save_dir: string
            Parent directory to save data to
        problem: Problem
            Problem class with dynamics, constraints, and parameters
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        graph_lims: tuple
            2-tuple with minimum and maximum bounds of state space
        n_states: int
            Trajectory length for edges in belief roadmap
        n_nodes: int
            Number of nodes for belief roadmap
        near_cutoff: float
            Proximity cutoff for neighboring nodes for edge rewiring
        max_nearby: int
            Maximum number of nearby nodes to consider for rewiring
        n_trials: int
            Number of experimental trials to run
    """

    edge_controllers = (EdgeController.BASELINE, EdgeController.ROBUST_SIGMA_POINT)
    trial_processes = []
    pickle.dump(problem, open(os.path.join(save_dir, f"problem.pkl"), "wb"))
    for edge_controller in edge_controllers:
        trial_process = Process(target=multi_query_experimental_trial, args=(0, save_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby))
        trial_processes.append(trial_process)
    for trial_process in trial_processes:
        trial_process.start()
    for trial_process in trial_processes:
        trial_process.join()

if __name__ == '__main__':
    #TODO put parameters in yaml
    # Multi-query experiments
    parent = os.path.join(__file__, os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")
    multi_query_dir = os.path.join(quad_results_dir, "multi_query_results")
    x_0 = np.array([5, 5, 0, 0, 0, 0])
    P_0 = 0.1*np.eye(6)
    problem = Quadrotor2DGRF(dt=0.1, obstacles="none")
    graph_min = np.array([0, 0, -10, -10, -100, -100])
    graph_max = np.array([10, 10, 10, 10, 100, 100])
    graph_lims = (graph_min, graph_max)
    n_states = 6
    n_nodes = 500
    near_cutoff = 36
    max_nearby = 5
    #multi_query_experiment(multi_query_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, near_cutoff, max_nearby)

    # Single-query experiments
    single_query_dir = os.path.join(quad_results_dir, "single_query_results")
    x_0 = np.array([2, 2, 0, 0, 0, 0])
    P_0 = 0.1*np.eye(6)
    x_f = np.array([8, 8, 0, 0, 0, 0])
    P_f = 0.2*np.eye(6)
    problem = Quadrotor2DGRF(dt=0.2, obstacles="center")
    graph_min = np.array([0, 0, -10, -10, -100, -100])
    graph_max = np.array([10, 10, 10, 10, 100, 100])
    graph_lims = (graph_min, graph_max)
    n_states = 6
    n_nodes = 200
    near_cutoff = 36
    max_nearby = 5
    n_trials = 20
    single_query_experiment(single_query_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, near_cutoff, max_nearby, n_trials)
