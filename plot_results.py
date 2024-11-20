from graph import Graph, Edge, Node
from src.utils.plot_utils import plot_quad_chart, confidence_ellipse

import copy
from quadrotor_experiment import *
import os
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle


def plot_single_query_results(results_path, metrics_path, plot_save_path, n_trials=20, n_mc_trials=200, n_graph_nodes=200, quad_chart_trial=0):
    graph_names = ['baseline', 'robust_ablation', 'rewired_ablation', 'revise']
    pkl_graph_names = [f'{graph_names[i]}_{n_graph_nodes}.pkl' for i in range(len(graph_names))]
    pretty_print_names = ["baseline", "robust ablation", "rewired ablation", "REVISE (ours)"]
    trimmed_graphs = []
    x_trajs = []
    for i, graph_name in enumerate(graph_names):
        mse = np.genfromtxt(os.path.join(metrics_path, f"{n_trials}_{graph_name}_mse.csv"), delimiter=',')
        wasserstein = np.genfromtxt(os.path.join(metrics_path, f"{n_trials}_{graph_name}_wass_dist.csv"), delimiter=',')
        max_eig = np.genfromtxt(os.path.join(metrics_path, f"{n_trials}_{graph_name}_max_eig.csv"), delimiter=',')
        print(f"Median for {pretty_print_names[i]}: MSE: {np.median(mse)}, Wass: {np.median(wasserstein)}, Eig: {np.median(max_eig)}")

    # Load results from trial 0 for plotting
    for i in range(len(graph_names)):
        graph_name = graph_names[i]
        load_dir = os.path.join(results_path, f'trial_{quad_chart_trial}')
        graph = pickle.load(open(os.path.join(load_dir, pkl_graph_names[i]), "rb"))
        mc_results_dir = os.path.join(load_dir, "mc_results")
        x_mc_0 = np.genfromtxt(os.path.join(mc_results_dir, f"x_mc_{graph_name}_0.csv"), delimiter=',')
        x_mc_trajs = np.zeros((x_mc_0.shape[0], x_mc_0.shape[1], n_mc_trials))
        for j in range(n_mc_trials):
            x_mc_j = np.genfromtxt(os.path.join(mc_results_dir, f"x_mc_{graph_name}_{j}.csv"), delimiter=',')
            x_mc_trajs[:, :, j] = x_mc_j
        trimmed_graph = graph.trim()
        trimmed_graphs.append(trimmed_graph)
        x_trajs.append(x_mc_trajs)

    # Only use baseline and REVISE
    trimmed_graphs = [trimmed_graphs[0], trimmed_graphs[3]]
    x_trajs = [x_trajs[0], x_trajs[3]]

    plot_quad_chart([0, 10], [0, 10], trimmed_graphs, x_trajs, title="", plot_obs=True, save_fig=True, save_fname=plot_save_path)

def plot_multi_query_results(metrics_path, plot_save_path, n_goals=100):
    graph_names = ['baseline', 'revise', 'robust_ablation', 'rewired_ablation']
    pretty_print_names = ["Baseline [1, 2]", "REVISE (ours)", "Robust ablation", "Rewired ablation"]
    mses = []
    wassersteins = []
    max_eigs = []
    for graph_name in graph_names:
        mse = np.genfromtxt(os.path.join(metrics_path, f"{n_goals}_{graph_name}_mse.csv"), delimiter=',')
        wasserstein = np.genfromtxt(os.path.join(metrics_path, f"{n_goals}_{graph_name}_wass_dist.csv"), delimiter=',')
        max_eig = np.genfromtxt(os.path.join(metrics_path, f"{n_goals}_{graph_name}_max_eig.csv"), delimiter=',')
        mses.append(mse)
        wassersteins.append(wasserstein)
        max_eigs.append(max_eig)

    fig = plt.figure()
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    plt.hist(wassersteins, range=(0, 1), label=pretty_print_names)
    plt.xlabel('$W_2$(planned final state, truth)')
    plt.ylabel('Frequency (out of 100)')
    plt.legend()
    plt.savefig(plot_save_path, dpi=1200, bbox_inches='tight')
    plt.show()

    for i in range(len(pretty_print_names)):
        print(f"Wass for {pretty_print_names[i]}: min: {wassersteins[i].min()}, median: {np.median(wassersteins[i])}, max: {wassersteins[i].max()}")

if __name__ == '__main__':
    parent = os.path.join(os.path.join(__file__, os.pardir), os.pardir)
    quad_results_dir = os.path.join(os.path.abspath(parent), "paper_results")

    single_query_updated_dir = os.path.join(quad_results_dir, "single_query_results")
    multi_query_updated_dir = os.path.join(quad_results_dir, "multi_query_results")
    single_query_plot_path = os.path.join(single_query_updated_dir, "quad.png")
    multi_query_plot_path = os.path.join(multi_query_updated_dir, "hist.png")

    plot_single_query_results(single_query_updated_dir, single_query_updated_dir, single_query_plot_path, n_trials=20)
    plot_multi_query_results(multi_query_updated_dir, multi_query_plot_path, n_goals=100)
