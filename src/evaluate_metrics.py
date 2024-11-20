import os
import pickle
import numpy as np
import scipy
from monte_carlo_simulation import *

def wasserstein_between_gaussians(mu_1, mu_2, cov_1, cov_2):
    """
    Calculate Wasserstein distance between two Gaussian
    distributions.

    Parameters
    -----------
        mu_1: np.ndarray
            Distribution 1 mean
        mu_2: np.ndarray
            Distribution 2 mean
        cov_1: np.ndarray
            Distribution 1 covariance
        cov_2: np.ndarray
            Distribution 2 covariance

    Returns
    --------
        wass_dist: float
            2-Wasserstein distance between distributions
    """
    wass_dist = np.linalg.norm(mu_1-mu_2)**2 + np.trace(cov_1 + cov_2 - 2*np.real(scipy.linalg.sqrtm(scipy.linalg.sqrtm(cov_2)@cov_1@scipy.linalg.sqrtm(cov_2))))
    return np.real(wass_dist)

def calculate_metrics(graph, x_mc_trajs):
    mu_ex = graph.get_goal_node().mean
    cov_ex = graph.get_goal_node().covariance
    mu_emp = np.mean(x_mc_trajs[:, -1, :], axis=1)
    cov_emp = np.cov(x_mc_trajs[:, -1, :])
    wass_dist = wasserstein_between_gaussians(mu_ex, mu_emp, cov_ex, cov_emp)
    mse = 0
    n_trials = x_mc_trajs.shape[2]
    for j in range(n_trials):
        mse += np.linalg.norm(x_mc_trajs[:, -1, j] - mu_ex)**2
    mse /= n_trials
    max_eigval_plan = np.linalg.eigvals(cov_ex).max()
    return wass_dist, mse, max_eigval_plan

def load_graphs_and_mc_trials(graph_dir, mc_results_dir, n_trials, n_graph_nodes):
    """
    Load graphs and MC results for baseline and sigma point method,
    with and without edge refinement, for a single trial.
    """
    # Load graphs
    revise_root, robust_ablation_root = get_root_filenames(EdgeController.ROBUST_SIGMA_POINT)
    rewired_ablation_root, baseline_root = get_root_filenames(EdgeController.BASELINE)

    problem, baseline_graph, rewired_ablation_graph, robust_ablation_graph, revise_graph = load_problem_and_roadmaps(graph_dir, n_graph_nodes)

    graphs = [(baseline_graph, baseline_root),
              (robust_ablation_graph, robust_ablation_root),
              (rewired_ablation_graph, rewired_ablation_root),
              (revise_graph, revise_root)]

    # Load MC
    mc_results = []
    for (graph, graph_name) in graphs:
        try:
            x_mc_0 = np.genfromtxt(os.path.join(mc_results_dir, f"x_mc_{graph_name}_0.csv"), delimiter=',')
            u_mc_0 = np.genfromtxt(os.path.join(mc_results_dir, f"u_mc_{graph_name}_0.csv"), delimiter=',')

            x_mc_trajs = np.zeros((x_mc_0.shape[0], x_mc_0.shape[1], n_trials))
            u_mc_trajs = np.zeros((u_mc_0.shape[0], u_mc_0.shape[1], n_trials))

            for j in range(n_trials):
                x_mc_j = np.genfromtxt(os.path.join(mc_results_dir, f"x_mc_{graph_name}_{j}.csv"), delimiter=',')
                u_mc_j = np.genfromtxt(os.path.join(mc_results_dir, f"u_mc_{graph_name}_{j}.csv"), delimiter=',')

                x_mc_trajs[:, :, j] = x_mc_j
                u_mc_trajs[:, :, j] = u_mc_j

            mc_results.append((graph_name, x_mc_trajs, u_mc_trajs))
        except:
            mc_results.append((graph_name, None, None))

    return graphs, mc_results

def evaluate_single_query(results_path, metrics_save_path, n_trials=20, n_mc_trials=200, n_graph_nodes=200):
    wass_dists = [[], [], [], []]
    mses = [[], [], [], []]
    eigs = [[], [], [], []]

    for n_trial in range(n_trials):
        trial_dir = os.path.join(results_path, f"trial_{n_trial}")
        mc_results_dir = os.path.join(trial_dir, "mc_results")
        graphs, mc_results = load_graphs_and_mc_trials(trial_dir, mc_results_dir, n_mc_trials, n_graph_nodes)
        for i, (graph, graph_name) in enumerate(graphs):
            x_mc_trajs = mc_results[i][1]
            if x_mc_trajs is not None:
                wass_dist, mse, plan_eig = calculate_metrics(graph, x_mc_trajs)
                wass_dists[i].append(wass_dist)
                mses[i].append(mse)
                eigs[i].append(plan_eig)

    for i, (_, graph_name) in enumerate(graphs):
        np.savetxt(os.path.join(metrics_save_path, f"{n_trials}_{graph_name}_wass_dist.csv"), np.array(wass_dists[i]), delimiter=',')
        np.savetxt(os.path.join(metrics_save_path, f"{n_trials}_{graph_name}_mse.csv"), np.array(mses[i]), delimiter=',')
        np.savetxt(os.path.join(metrics_save_path, f"{n_trials}_{graph_name}_max_eig.csv"), np.array(eigs[i]), delimiter=',')

def evaluate_multi_query(results_path, metrics_save_path, n_goals=100, n_trials=200, n_graph_nodes=501):
    wass_dists = [[], [], [], []]
    mses = [[], [], [], []]
    eigs = [[], [], [], []]

    for n_goal in range(n_goals):
        logging.warning(f"Loading {n_goal}")
        goal_dir = os.path.join(results_path, f"rand_goal_{n_goal}")
        mc_results_dir = os.path.join(goal_dir, "mc_results")
        graphs, mc_results = load_graphs_and_mc_trials(goal_dir, mc_results_dir, n_trials, n_graph_nodes)
        for i, (graph, graph_name) in enumerate(graphs):
            x_mc_trajs = mc_results[i][1]
            wass_dist, mse, plan_eig = calculate_metrics(graph, x_mc_trajs)
            wass_dists[i].append(wass_dist)
            mses[i].append(mse)
            eigs[i].append(plan_eig)

    for i, (_, graph_name) in enumerate(graphs):
        np.savetxt(os.path.join(metrics_save_path, f"{n_goals}_{graph_name}_wass_dist.csv"), np.array(wass_dists[i]), delimiter=',')
        np.savetxt(os.path.join(metrics_save_path, f"{n_goals}_{graph_name}_mse.csv"), np.array(mses[i]), delimiter=',')
        np.savetxt(os.path.join(metrics_save_path, f"{n_goals}_{graph_name}_max_eig.csv"), np.array(eigs[i]), delimiter=',')

if __name__ == '__main__':
    parent = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")
    multi_query_dir = os.path.join(quad_results_dir, "multi_query_results")
    single_query_dir = os.path.join(quad_results_dir, "single_query_results")
    single_query_corrected_dir = os.path.join(quad_results_dir, "single_query_results_11-4-24")
    multi_query_updated_dir = os.path.join(quad_results_dir, "multi_query_results_11-4-24")
    test_dir = os.path.join(os.path.abspath(parent), "test")
    artifacts_dir = os.path.join(test_dir, "artifacts")

    # Calculate and save metrics
    evaluate_single_query(single_query_corrected_dir, single_query_corrected_dir, n_trials=20)
    evaluate_multi_query(multi_query_updated_dir, multi_query_updated_dir)
