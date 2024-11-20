import os
import pickle
from multiprocessing import Process
import logging

import numpy as np

from src.roadmap_generation import get_roadmap_edge, get_nearest_nodes
from src.utils.io_utils import get_root_filenames, save_roadmap
from src.covariance_steering import EdgeController
from src.graph import Node, Edge, Graph

def parallelize_random_goal_generation():
    parent = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")
    multi_query_updated_dir = os.path.join(quad_results_dir, "multi_query_results")
    n_graph_nodes = 500
    n_goals = 100
    max_cutoff = 36
    max_nearby = 5
    n_states = 6
    graph_min = np.array([0, 0, -10, -10, -100, -100])
    graph_max = np.array([10, 10, 10, 10, 100, 100])
    graph_lims = [graph_min, graph_max]

    goal_processes = []
    for n_goal in range(n_goals):
        random_seed = n_goal
        goal_process = Process(target=generate_random_multi_query_goals, args=(random_seed, multi_query_updated_dir, multi_query_updated_dir, graph_lims, n_states, n_goal, n_goal+1, n_graph_nodes, max_cutoff, max_nearby))
        goal_processes.append(goal_process)
    for goal_process in goal_processes:
        goal_process.start()
    for goal_process in goal_processes:
        goal_process.join()

def parallelize_multi_query_simulation():
    parent = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")
    multi_query_updated_dir = os.path.join(quad_results_dir, "multi_query_results")
    n_mc_trials = 200
    n_graph_nodes = 500
    n_goals = 100
    n_batches = 10
    n_goals_per_batch = 10
    for n_batch in range(n_batches):
        simulation_processes = [] 
        for n_goal in range(n_batch*n_goals_per_batch, (n_batch+1)*n_goals_per_batch):
            map_load_path = os.path.join(multi_query_updated_dir, f"rand_goal_{n_goal}")
            results_save_path = os.path.join(map_load_path, 'mc_results')
            os.makedirs(results_save_path, exist_ok=True)
            simulation_process = Process(target=simulate_path_to_goal, args=(0, map_load_path, results_save_path, n_mc_trials, n_graph_nodes+1))
            simulation_processes.append(simulation_process)
        for simulation_process in simulation_processes:
            simulation_process.start()
        for simulation_process in simulation_processes:
            simulation_process.join()

def parallelize_single_query_simulation():
    #TODO cleanup
    parent = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")
    single_query_updated_dir = os.path.join(quad_results_dir, "single_query_results")
    n_trials = 20
    n_mc_trials = 200
    n_graph_nodes = 200

    simulation_processes = []
    for trial in range(n_trials):
        map_load_path = os.path.join(single_query_updated_dir, f"trial_{trial}")
        results_save_path = os.path.join(map_load_path, 'mc_results')
        os.makedirs(results_save_path, exist_ok=True)
        simulation_process = Process(target=simulate_path_to_goal, args=(0, map_load_path, results_save_path, n_mc_trials, n_graph_nodes))
        simulation_processes.append(simulation_process)
    for simulation_process in simulation_processes:
        simulation_process.start()
    for simulation_process in simulation_processes:
        simulation_process.join()


def load_problem_and_roadmaps(map_load_path, n_graph_nodes):
    """
    Load problem and roadmaps from saved experimental results.

    Parameters
    ------------
        map_load_path: string
            Directory to load roadmaps from
        n_graph_nodes: int
            Number of nodes in roadmap (used to load correct roadmap)

    Returns
    ---------
        problem: Problem
            Problem used to generate experimental results
        baseline_graph: Graph
            Roadmap generated with baseline edge controller and no edge
            rewiring
        rewired_ablation_graph: Graph
            Roadmap generated with baseline edge controller and edge rewiring
        robust_ablation_graph: Graph
            Roadmap generated with robust sigma point edge controller and
            no edge rewiring
        revise_graph: Graph
            Roadmap generated with REVISE
    """
    # Get root filenames for roadmaps
    revise_root, robust_ablation_root = get_root_filenames(EdgeController.ROBUST_SIGMA_POINT)
    rewired_ablation_root, baseline_root = get_root_filenames(EdgeController.BASELINE)

    # Load roadmaps and problem
    robust_ablation_graph = pickle.load(open(os.path.join(map_load_path, f'{robust_ablation_root}_{n_graph_nodes}.pkl'), "rb"))
    revise_graph = pickle.load(open(os.path.join(map_load_path, f'{revise_root}_{n_graph_nodes}.pkl'), "rb"))
    baseline_graph = pickle.load(open(os.path.join(map_load_path, f'{baseline_root}_{n_graph_nodes}.pkl'), "rb"))
    rewired_ablation_graph = pickle.load(open(os.path.join(map_load_path, f'{rewired_ablation_root}_{n_graph_nodes}.pkl'), "rb"))
    problem = pickle.load(open(os.path.join(map_load_path, f"problem.pkl"), "rb"))

    return problem, baseline_graph, rewired_ablation_graph, robust_ablation_graph, revise_graph


def simulate_path_to_goal(random_seed, map_load_path, results_save_path, n_mc_trials, n_graph_nodes):
    """
    Simulate the true dynamics forward from the start node to the goal for
    any roadmap containing both start and goal nodes.

    Parameters
    -----------
        random_seed: int
            Random seed to use for generating Monte Carlo initial conditions
        map_load_path: string
            Folder to load roadmaps from
        results_save_path: string
            Folder to save trajectories to
        n_mc_trials: int
            Number of Monte Carlo trials to run
        n_graph_nodes: int
            Number of nodes in each roadmap (used to make sure correct
            roadmap is loaded)
    """
    # Load roadmaps 
    revise_root, robust_ablation_root = get_root_filenames(EdgeController.ROBUST_SIGMA_POINT)
    rewired_ablation_root, baseline_root = get_root_filenames(EdgeController.BASELINE)
    problem, baseline_graph, rewired_ablation_graph, robust_ablation_graph, revise_graph = load_problem_and_roadmaps(map_load_path, n_graph_nodes)

    # Generate initial states for MC
    u_traj, K_traj, x_traj, P_traj = revise_graph.get_plan_to_goal()
    x_0 = x_traj[0][:, 0]
    P_0 = revise_graph.get_start_node().covariance#P_traj[0][:, :, 0]
    x_inits = []

    # Reset random seed
    np.random.seed(random_seed)
    for i in range(n_mc_trials):
        x_inits.append(np.random.multivariate_normal(x_0, P_0))

    # Get MC results
    graphs = [(robust_ablation_graph, robust_ablation_root),
              (revise_graph, revise_root),
              (baseline_graph, baseline_root),
              (rewired_ablation_graph, rewired_ablation_root)]

    for graph, graph_name in graphs:
        try:
            # Reset random seed (so all maps experience the same changes
            # in the Gaussian random field)
            np.random.seed(random_seed)
            simulate_graph_forward_monte_carlo(results_save_path, f"x_mc_{graph_name}", f"u_mc_{graph_name}", graph, problem, x_inits)
        except: # No valid path to goal (sometimes happens for baseline + robust ablation)
            pass

def add_random_goal_to_graph(problem, graph, edge_controller, rewire_edges, rand_goal, n_states, near_cutoff, max_nearby):
    """
    Add random goal to graph and try steering to it. If rewiring edges, try
    steering from up to `max_nearby` neighboring nodes and choose lowest
    cost edge. Otherwise, try steering from up to `max_nearby` neighboring
    nodes, but accept first successful edge.

    Parameters
    ------------
        problem: Problem
            Problem class with associated dynamics & constraints
        graph: Graph
            Belief roadmap to extend to goal
        edge_controller: EdgeController
            Valid edge controller to use to steer to goal
        rewire_edges: bool
            True if rewiring edges, False otherwise
        rand_goal: np.ndarray
            Random goal mean in state space
        n_states: int
            Trajectory length for edges in roadmap
        near_cutoff: float
            Proximity cutoff for nodes to try to steer to goal
        max_nearby: int
            Maximum number of nodes to try steering to goal

    Returns
    --------
        success: bool
            True if goal successfully added to graph
        graph: Graph
            Updated graph including goal
    """
    # Get neighboring nodes and check if closest node can steer to goal
    closest_node, near_nodes = get_nearest_nodes(problem, rand_goal, graph.nodes, near_cutoff, max_nearby)
    parent_best = closest_node
    overall_success, x_traj_best, u_traj_best, cov_traj_best, K_best, P_f_best = get_roadmap_edge(problem, closest_node.mean, closest_node.covariance, rand_goal, n_states, edge_controller)

    # If closest node failed, or rewiring edges, keep trying
    if rewire_edges or not overall_success:
        for neighbor in near_nodes:
            success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, neighbor.mean, neighbor.covariance, rand_goal, n_states, edge_controller)
            if success and not overall_success:
                overall_success = success
                x_traj_best, u_traj_best, cov_traj_best, K_best, P_f_best = x_traj, u_traj, cov_traj, K, P_f
                parent_best = neighbor
            # If rewiring, only accept edge if it's better than previous best
            if success and overall_success and rewire_edges:
                if np.linalg.eigvals(P_f).max() < np.linalg.eigvals(P_f_best).max():
                    x_traj_best, u_traj_best, cov_traj_best, K_best, P_f_best = x_traj, u_traj, cov_traj, K, P_f
                    parent_best = neighbor

    # Add best edge to the graph
    if overall_success:
        new_node = Node(rand_goal, P_f_best, is_goal=True)
        new_edge = Edge(parent_best, new_node, x_traj_best, cov_traj_best, u_traj_best, K_best)
        graph.add_node(new_node)
        graph.add_edge(new_edge)

        return True, graph
    else:
        return False, graph

def generate_random_multi_query_goals(random_seed, map_load_path, results_save_path, graph_lims, n_states, n_successful_goals, n_goals, n_graph_nodes, near_cutoff, max_nearby):
    """
    Given pre-existing belief roadmaps, generate multiple random goals and add
    them to the roadmaps so that paths can be planned to the random goals.

    Instead of running this function once with `n_successful_goals` = 0 and
    `n_goals` equal to the number of desired goals, this function can be run
    in parallel to speed things up, as long as the random seed is different
    in each process.

    Parameters
    -------------
        random_seed: int
            Random seed to use for sampling random goals
        map_load_path: string
            Folder to load roadmaps from
        results_save_path: string
            Folder to save roadmaps with goals to
        graph_lims: tuple
            2-tuple with minimum and maximum bounds on state space
        n_states: int
            Trajectory length for edges in roadmap
        n_successful_goals: int
            Number of already-existing random goals
        n_goals: int
            Number of random goals to find
        n_graph_nodes: int
            Number of nodes in roadmaps to load (used to find correct roadmap)
        near_cutoff: float
            Proximity cutoff for nodes to try to steer to goal
        max_nearby: int
            Maximum number of nodes to try steering to goal
    """
    # Get root filenames
    revise_root, robust_ablation_root = get_root_filenames(EdgeController.ROBUST_SIGMA_POINT)
    rewired_ablation_root, baseline_root = get_root_filenames(EdgeController.BASELINE)

    # Set random seed
    np.random.seed(random_seed)

    # While n_successful_goals < n_goals, sample + steer
    while n_successful_goals < n_goals:
        logging.warning(f"Success on {n_successful_goals}, we want {n_goals}")

        # Reload graphs so we aren't adding multiple goal nodes
        problem, baseline_graph, rewired_ablation_graph, robust_ablation_graph, revise_graph = load_problem_and_roadmaps(map_load_path, n_graph_nodes)

        graphs = [(baseline_root, baseline_graph, EdgeController.BASELINE, False),
                  (robust_ablation_root, robust_ablation_graph, EdgeController.ROBUST_SIGMA_POINT, False),
                  (rewired_ablation_root, rewired_ablation_graph, EdgeController.BASELINE, True),
                  (revise_root, revise_graph, EdgeController.ROBUST_SIGMA_POINT, True)]

        # Randomly sample goal node
        dims = graph_lims[0].shape
        rand_goal = np.random.rand(dims[0])
        rand_goal = graph_lims[0] + (graph_lims[1]-graph_lims[0])*rand_goal

        # For each graph, try steering to the goal
        any_fail = False
        for i, (graph_name, graph, edge_controller, rewire_edges) in enumerate(graphs):
            success, new_graph = add_random_goal_to_graph(problem, graph, edge_controller, rewire_edges, rand_goal, n_states, near_cutoff, max_nearby)
            if not success:
                logging.warning(f"Graph with name {graph_name} failed")
                any_fail = True
                break
            else:
                logging.warning(f"Graph with name {graph_name} succeeded")
                graphs[i] = (graph_name, new_graph, edge_controller, rewire_edges)

        # If success, save roadmaps and update successful goal count
        if not any_fail:
            results_dir = os.path.join(results_save_path, f"rand_goal_{n_successful_goals}")
            os.makedirs(results_dir, exist_ok=True)
            pickle.dump(problem, open(os.path.join(results_dir, f"problem.pkl"), "wb"))
            for (graph_name, graph, _, _) in graphs:
                save_roadmap(results_dir, graph_name, graph)

            n_successful_goals += 1

def simulate_closed_loop_trajectory(u_traj, K_traj, x_traj, x_init, wind_field, problem):
    control_size = problem.control_size
    state_size = problem.state_size
    n_edges = len(u_traj)
    len_edge = u_traj[0].shape[1]

    x_mc_traj = np.zeros((state_size, n_edges*len_edge + 1))
    u_mc_traj = np.zeros((control_size, n_edges*len_edge))
    x_mc_traj[:, 0] = x_init

    for i in range(n_edges):
        for j in range(len_edge):
            u = u_traj[i][:, j] + K_traj[i][control_size*j:control_size*(j+1), :]@(x_mc_traj[:, i*len_edge:i*len_edge + (len_edge+1)].T.reshape(-1,) - x_traj[i].T.reshape(-1,))
            x = problem.stochastic_dynamics_known_w(x_mc_traj[:, i*len_edge + j], u, wind_field)
            u_mc_traj[:, i*len_edge+j] = u
            x_mc_traj[:, i*len_edge+j+1] = x

    return x_mc_traj, u_mc_traj

def simulate_graph_forward_monte_carlo(save_dir, x_fname_root, u_fname_root, graph, problem, x_inits):
    # Get nominal state, feedforward and feedback control
    # from start to goal
    u_traj, K_traj, x_traj, P_traj = graph.get_plan_to_goal()
    n_edges = len(u_traj)
    len_edge = u_traj[0].shape[1]
    n_mc_runs = len(x_inits)

    # Simulate forward
    for i in range(n_mc_runs):
        # Sample wind field
        problem.resample_grf()
        wind_field = problem.wind_profile

        x_mc_traj, u_mc_traj = simulate_closed_loop_trajectory(u_traj, K_traj, x_traj, x_inits[i], wind_field, problem)
        np.savetxt(os.path.join(save_dir, f"{x_fname_root}_{i}.csv"), x_mc_traj, delimiter=',')
        np.savetxt(os.path.join(save_dir, f"{u_fname_root}_{i}.csv"), u_mc_traj, delimiter=',')

if __name__ == '__main__':
    n_trials = 20
    n_mc_trials = 200
    n_graph_nodes = 200
    max_cutoff = 36
    max_nearby = 5
    n_states = 6
    graph_min = np.array([0, 0, -10, -10, -100, -100])
    graph_max = np.array([10, 10, 10, 10, 100, 100])
    graph_lims = [graph_min, graph_max]
    n_graph_nodes_mq = 500

    parallelize_random_goal_generation()
    parallelize_multi_query_simulation()
    parallelize_single_query_simulation()
