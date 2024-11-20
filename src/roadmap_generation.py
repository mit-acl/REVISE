import os
import pickle
import logging
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt

from src.quadrotor_problem import Quadrotor2DGRF
from src.covariance_steering import *
from src.graph import *
from src.utils.io_utils import *

def randomize_candidate_mean(problem, graph, graph_lims, sample_radius_lims):
    """
    Sample mean for a candidate node, biasing towards underexplored
    regions of the state space.

    Parameters
    -----------
        problem: Problem
            Problem class with `proximity_weights` property, which
            specifies weights for computing Euclidean distance (e.g.
            weight position more than acceleration)
        graph: Graph
            Belief roadmap
        graph_lims: tuple
            2-tuple of numpy arrays specifying minimum and maximum
            bounds of state space 
        sample_radius_lims: tuple
            2-tuple of numpy arrays specifying minimum and maximum
            bounds for sampling around an existing node in the graph

    Returns
    ---------
        candiate_mean: np.ndarray
            Random point in state space, falling within `sample_radius_lims`
            of `expansion_node`
        expansion_node: Node
            Node in the graph that is a candidate for expansion towards
            `candidate_mean`
    """
    # Sample random point from state space
    rand_pt = np.random.rand(problem.state_size)
    rand_pt = graph_lims[0] + (graph_lims[1]-graph_lims[0])*rand_pt

    # Find closest node to random point (this biases towards expanding
    # node with largest Voronoi region)
    expansion_node, _ = get_nearest_nodes(problem, rand_pt, graph.nodes, 0, 0)

    # Sample in radius around the selected node
    candidate_mean = np.random.rand(problem.state_size)
    candidate_mean = sample_radius_lims[0] + (sample_radius_lims[1]-sample_radius_lims[0])*candidate_mean + expansion_node.mean

    return candidate_mean, expansion_node 

def get_nearest_nodes(problem, sample, nodes, near_cutoff, max_nearby):
    """
    Get closest node (regardless of distance) and up to `max_nearby` 
    closest nodes which are  all within `near_cutoff` of the sample. 
    Uses weighted Euclidean distance to compute closeness.

    Parameters
    -----------
        problem: Problem
            Problem class with `proximity_weights` property for computing
            weighted Euclidean distance
        sample: np.ndarray
            Random point in the state space
        nodes: set
            Set of Node objects in belief roadmap
        near_cutoff: float
            Cutoff for proximity for computing nearby nodes
        max_nearby: int
            Maximum number of nearby nodes to return

    Returns
    --------
        nearest_neighbor: Node
            Node in `nodes` which is closest to `sample`
        nearby_nodes: list
            List of closest nodes in `nodes` to `sample`, subject
            to the constraint that distance to `sample` is less than
            `near_cutoff`. Contains 0 to `max_nearby` nodes.

    """
    closest_distance = np.inf
    nearest_neighbor = -1
    near_nodes = []
    for node in nodes:
        if not node.is_goal: #OK to sample near goal
            wt = problem.proximity_weights
            dist = np.linalg.norm(wt*(node.mean-sample))**2
            if dist < closest_distance:
                closest_distance = min(closest_distance, dist)
                nearest_neighbor = node
            if dist < near_cutoff:
                near_nodes.append((dist, node))
    nearest_nodes = [node_mean for _, node_mean in sorted(near_nodes, key=lambda x: x[0])]
    nearby_nodes = nearest_nodes[0:max_nearby]
    return nearest_neighbor, nearby_nodes 

def get_roadmap_edge(problem, x_0, P_0, x_f, n_states, edge_controller):
    """
    Get a belief roadmap edge from :math:`\mathcal{N}(\mathbf{x_0}, \lambda_{\max}(P_0)I)`
    to a distribution with mean :math:`\mathbf{x_f}`, if such an edge
    exists.

    Parameters
    -----------
        problem: Problem
            Problem class with associated dynamics and constraints
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance. Note that this function does not
            steer from :math:`\mathcal{N}(\mathbf{x_0}, P_0)`, but rather
            from :math:`\mathcal{N}(\mathbf{x_0}, P_{0, \max})`, where
            :math:`P_{0, \max} = \lambda_{\max}(P_0)I`.
        x_f: np.ndarray
            Final state mean
        n_states: int
            Trajectory length
        edge_controller: EdgeController
            Type of edge controller to use

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
        P_f: np.ndarray
            Final state covariance, or None if no edge found

    Raises
    --------
        ValueError
            If `edge_controller` is not a valid edge controller type. Currently
            supported edge controllers are EdgeController.BASELINE and
            EdgeController.ROBUST_SIGMA_POINT.
    """
    # Get mouth of funnel
    max_eig = np.linalg.eigvals(P_0).max()
    cov = np.eye(P_0.shape[0])*max_eig

    # Attempt steering
    if edge_controller == EdgeController.ROBUST_SIGMA_POINT:
        success, x_traj, u_traj, cov_traj, K, P_f = robust_sigma_point_edge_controller(problem, x_0, cov, x_f, n_states)
    elif edge_controller == EdgeController.BASELINE:
        success, x_traj, u_traj, cov_traj, K, P_f = baseline_edge_controller(problem, x_0, cov, x_f, n_states)
    else:
        raise ValueError("Edge controller is of unknown type.")

    return success, x_traj, u_traj, cov_traj, K, P_f

def update_node_and_descendants(problem, graph, new_parent, node, x_traj, u_traj, cov_traj, K, P_f, n_states, edge_controller):
    """
    Update `node` in a belief roadmap, such that its parent is now `new_parent`.
    Recursively update state covariance at `node` and all of its descendants
    by setting the state covariance at `node` to `P_f` and then recursively 
    recomputing edges from `node` to its descendants.

    Rather than directly modifying `graph`, this function deepcopies `graph`,
    modifies the copy, and returns the modified copy.

    Parameters
    ------------
        problem: Problem
            Problem class with associated dynamics, constraints, and parameters
        graph: Graph
            Belief roadmap to copy and update
        new_parent: Node
            Node in `graph` which is now the parent of `node`
        node: Node
            Node in `graph` whose parent is now `new_parent`
        x_traj: np.ndarray
            Mean state trajectory for edge from `new_parent` to `node`
        u_traj: np.ndarray
            Nominal control for edge from `new_parent` to `node`
        cov_traj: np.ndarray
            State covariance trajectory for edge from `new_parent` to `node`
        K: np.ndarray
            Feedback gain for edge from `new_parent` to `node`
        P_f: np.ndarray
            Final state covariance for edge from `new_parent` to `node`
        n_states: int
            Trajectory length for each edge
        edge_controller: EdgeController
            Edge controller to use for recomputing edges from `node` to
            its descendants

    Returns
    ----------
        new_graph: Graph
            Updated belief roadmap, where `new_parent` is now the parent of `node`, and where the updated covariance of `node` is propagated to its descendants
    """
    logging.info("In update...")
    logging.info(f"Updating a node with {len(graph.get_descendants(node))} descendants")

    # Update node and get its children (if it has any)
    children = graph.get_children(node)
    nodes = copy.deepcopy(graph.nodes)
    edges = copy.deepcopy(graph.edges)
    old_parent_edge = -1
    for edge in edges:
        if edge.end_node == node:
            old_parent_edge = edge
    edges.remove(old_parent_edge)
    nodes.remove(node)
    if node.is_goal:
        new_node = Node(node.mean, covariance=P_f, is_goal=True)
    else:
        new_node = Node(node.mean, covariance=P_f)
    nodes.add(new_node)
    new_edge = Edge(new_parent, new_node, x_traj, covariance=cov_traj, ff_ctrl=u_traj, fb_ctrl=K)
    edges.add(new_edge)

    # Recursively update all descendants of node by recomputing edges
    new_graph = Graph(nodes, edges)
    for child in children:
        success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, new_node.mean, new_node.covariance, child.mean, n_states, edge_controller)
        # May fail rarely due to too many iterations required to converge,
        # or may fail if robust sigma point edge control is used and 
        # condition (1) or (2) from Theorem V.1 is violated.
        if success:
            # Recursively update descendants of child
            new_graph = update_node_and_descendants(problem, new_graph, new_node, child, x_traj, u_traj, cov_traj, K, P_f, n_states, edge_controller)
        else: 
            # Just update edge to use new parent node, but keep
            # same plan to get to child and to all of its descendants.
            # This works because at least one ancestor of child has
            # smaller state covariance than in the old plan. So the
            # old plan should still hold under mild assumptions.
            nodes = copy.deepcopy(new_graph.nodes)
            edges = copy.deepcopy(new_graph.edges)
            old_parent_edge = -1
            for edge in edges:
                if edge.end_node == child:
                    old_parent_edge = edge

            new_edge = Edge(new_node, child, old_parent_edge.mean, covariance=old_parent_edge.covariance, ff_ctrl=old_parent_edge.ff_ctrl, fb_ctrl=old_parent_edge.fb_ctrl)
            edges.remove(old_parent_edge)
            edges.add(new_edge)
            new_graph = Graph(nodes, edges)

    return new_graph

def add_node_to_roadmap(problem, graph, node_mean, parent, n_states, edge_controller, node_is_goal):
    """
    Add a new node to a belief roadmap if a valid edge can be found from `parent` to `node_mean`.
    
    Corresponds to lines 5-9 in Algorithm I in the paper.

    Parameters
    -----------
        problem: Problem
            Problem class with associated dynamics and constraints
        graph: Graph
            Belief roadmap to update
        node_mean: np.ndarray
            Candidate node mean to add to the roadmap
        parent: Node
            Existing node in the roadmap, candidate parent node
        n_states: int
            Trajectory length for edge
        edge_controller: EdgeController
            Edge controller to solve for edge
        node_is_goal: bool
            True if new node is added as a goal node

    Returns
    ---------
        graph: Graph
            Belief roadmap, updated if edge was found
        success: bool
            True if new node and edge added to roadmap
    """
    logging.info(f"Node mean: {node_mean}")
    logging.info(f"Closest node mean: {parent.mean}")

    success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, parent.mean, parent.covariance, node_mean, n_states, edge_controller)
    if success:
        # Add node and edge to graph
        if node_is_goal:
            graph.nodes.remove(graph.get_goal_node())
        new_node = Node(node_mean, P_f, is_goal=node_is_goal)

        logging.info(f"Added node with mean {node_mean} and cov {P_f}, graph now has {len(graph.nodes)} nodes")
        logging.info(f"Max cov eigval = {np.linalg.eigvals(P_f).max()}")

        new_edge = Edge(parent, new_node, x_traj, cov_traj, u_traj, K)
        graph.add_node(new_node)
        graph.add_edge(new_edge)

    return graph, success

def add_node_and_rewire_roadmap(problem, rewired_graph, graph, node_mean, parent, n_states, edge_controller, near_cutoff, max_nearby, node_is_goal):
    """
    Add a new node to `rewired_graph` if a valid edge can be found from `parent`
    to `node_mean`, and rewire edges.

    Corresponds to lines 6-30 in Algorithm II in the paper.

    Parameters
    -----------
        problem: Problem
            Problem class with associated dynamics and constraints
        rewired_graph: Graph
            Belief roadmap to update
        graph: Graph
            Belief roadmap with the same node means as `rewired_graph`,
            but with different node covariances, because `graph` is
            constructed without rewiring.
        node_mean: np.ndarray
            Candidate node mean to add to the roadmap
        parent: Node
            Existing node in the roadmap, candidate parent node
        n_states: int
            Trajectory length for edge
        edge_controller: EdgeController
            Edge controller to solve for edge
        near_cutoff: float
            Cutoff for proximity for rewiring (only look to rewire with 
            nodes which are closer than `near_cutoff`)
        max_nearby: int
            Maximum number of nodes to consider for rewiring
        node_is_goal: bool
            True if new node is added as a goal node

    Returns
    ---------
        rewired_graph: Graph
            Updated version of rewired_graph 
        graph: Graph
            Updated version of graph
        success: bool
            True if node was successfully added to `rewired_graph`
        new_node: Node
            New node which was added to `rewired_graph`
    """
    logging.info(f"Node mean: {node_mean}")
    logging.info(f"Closest node mean: {parent.mean}")

    # Get nodes neighboring the candidate node mean 
    _, neighboring_nodes = get_nearest_nodes(problem, node_mean, rewired_graph.nodes, near_cutoff, max_nearby)

    # Continue only if new node is a goal node, or if new node can
    # be added to graph
    if not node_is_goal:
        new_graph, success = add_node_to_roadmap(problem, Graph(copy.deepcopy(graph.nodes), copy.deepcopy(graph.edges)), node_mean, parent, n_states, edge_controller, False)
    else:
        success = True

    # Continue only if new node can be added to rewired_graph
    if success:
        parent = rewired_graph.look_up_by_mean(parent.mean)
        success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, parent.mean, parent.covariance, node_mean, n_states, edge_controller)

    # Add node to rewired_graph, and rewire rewired_graph accordingly
    if success:
        # Update graph to accept new edge
        if not node_is_goal:
            graph = new_graph

        # Current best parent is parent 
        src_best = parent 
        x_traj_best, u_traj_best, cov_traj_best, K_best, P_f_best = x_traj, u_traj, cov_traj, K, P_f

        # Try steering from all nodes in neighboring_node and get 
        # best edge to add to rewired_graph
        for nearby_neighbor in neighboring_nodes:
            if nearby_neighbor != parent and not nearby_neighbor.is_goal:
                success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, nearby_neighbor.mean, nearby_neighbor.covariance, node_mean, n_states, edge_controller)
                # Check if covariance is smaller than previous best
                if success and np.linalg.eigvals(P_f).max() < np.linalg.eigvals(P_f_best).max():
                    x_traj_best, u_traj_best, cov_traj_best, K_best, P_f_best = x_traj, u_traj, cov_traj, K, P_f
                    src_best = nearby_neighbor

        if node_is_goal:
            # Remove prior goal node and edges to prior
            # goal node
            old_goal_node = rewired_graph.get_goal_node()
            old_edges = []
            for edge in rewired_graph.edges:
                if edge.end_node == old_goal_node:
                    old_edges.append(edge)
            for edge in old_edges:
                rewired_graph.edges.remove(edge)
            rewired_graph.nodes.remove(old_goal_node)

        # Add best edge to node to rewired_graph
        print(f"Added node with mean {node_mean} and cov {P_f_best}, graph now has {len(rewired_graph.nodes)} nodes")
        print(f"Max cov eigval = {np.linalg.eigvals(P_f_best).max()}")

        new_node = Node(node_mean, P_f_best, is_goal=node_is_goal)
        new_edge = Edge(src_best, new_node, x_traj_best, covariance=cov_traj_best, ff_ctrl=u_traj_best, fb_ctrl=K_best)
        rewired_graph.add_node(new_node)
        rewired_graph.add_edge(new_edge)

        # Try steering from the new node to neighboring nodes, and rewire
        # if new edge is found to a neighboring node which reduces state
        # covariance at that node
        if not node_is_goal:
            for nearby_neighbor in neighboring_nodes:
                # Ensure neighbor is in graph (it may have been updated
                #based on an update to one of its ancestors)
                if nearby_neighbor not in rewired_graph.nodes:
                    nearby_neighbor = rewired_graph.look_up_by_mean(nearby_neighbor.mean)
                # Ensure we are not steering to an ancestor or self steering
                if nearby_neighbor != new_node and nearby_neighbor not in rewired_graph.get_ancestors(new_node):
                    # Steer from new_node to nearby_neighbor 
                    success, x_traj, u_traj, cov_traj, K, P_f = get_roadmap_edge(problem, new_node.mean, new_node.covariance, nearby_neighbor.mean, n_states, edge_controller)
                    # If this edge lowers the state covariance at
                    # nearby_neighbor, update nearby_neighbor and its
                    # descendants
                    if success and np.linalg.eigvals(P_f).max() < np.linalg.eigvals(nearby_neighbor.covariance).max():
                        rewired_graph = update_node_and_descendants(problem, rewired_graph, new_node, nearby_neighbor, x_traj, u_traj, cov_traj, K, P_f, n_states, edge_controller)

        success = True
    else:
        new_node = -1
        logging.info("Edge was not successful")

    return rewired_graph, graph, success, new_node

def construct_belief_roadmaps_and_rewire(save_dir, problem, x_0, P_0, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby):
    """
    Given an initial state distribution, build out two belief roadmaps with
    n_nodes, one with edge rewiring and one without, using the non-rewired 
    roadmap to sample nodes for expansion, then adding each new node to both 
    roadmaps, and rewiring the rewired roadmap along the way. Both roadmaps
    will always contain the same node means.

    Equivalent to Algorithm II in the paper.

    Parameters
    ------------
        save_dir: string
            Valid folder for saving data
        problem: Problem
            Problem class with associated dynamics, constraints, etc.
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        graph_lims: tuple
            2-tuple of numpy arrays specifying minimum and maximum bounds
            of state space
        n_states: int
            Trajectory length for each edge in roadmap
        n_nodes: int
            Desired number of nodes in roadmap
        edge_controller: EdgeController
            Valid edge controller for constructing roadmap
        near_cutoff: float
            Proximity cutoff for neighboring nodes for edge rewiring
        max_nearby: int
            Maximum number of neighboring nodes to consider for rewiring

    Returns
    ---------
        rewired_graph: Graph
            Belief roadmap constructed with edge rewiring
        graph: Graph
            Belief roadmap constructed without edge rewiring, with the same
            set of node means as `rewired_graph`
    """
    # Define start node, and create roadmaps
    graph = Graph({Node(x_0, P_0, is_start=True)}, set())
    rewired_graph = Graph({Node(x_0, P_0, is_start=True)}, set())

    # Get filenames for saving data
    rewired_fname_root, no_rewire_fname_root = get_root_filenames(edge_controller)
    # Continue adding nodes until both roadmaps contain n_nodes
    while len(rewired_graph.nodes) < n_nodes:
        # Sample w/ bias towards underexplored regions of state space
        # based on nodes in graph (NOT rewired_graph)
        node_mean, parent = randomize_candidate_mean(problem, graph, graph_lims, problem.sample_radius_lims)
        # Add node to both graphs, and rewire rewired_graph only
        rewired_graph, graph, _, _ = add_node_and_rewire_roadmap(problem, rewired_graph, graph, node_mean, parent, n_states, edge_controller, near_cutoff, max_nearby, False)
        # Save roadmaps
        save_roadmap(save_dir, rewired_fname_root, rewired_graph)
        save_roadmap(save_dir, no_rewire_fname_root, graph)

    return rewired_graph, graph
   
def construct_belief_roadmaps_to_goal_and_rewire(save_dir, problem, x_0, P_0, x_f, P_f, graph_lims, n_states, n_nodes, edge_controller, near_cutoff, max_nearby):
    """
    Given an initial state distribution and a final goal distribution,
    build out two belief roadmaps with n_nodes, one with edge rewiring
    and one without, using the non-rewired roadmap to sample nodes for
    expansion, then adding each new node to both roadmaps, and rewiring
    the rewired roadmap along the way. Both roadmaps will always contain 
    the same node means. The path to the goal will be rewired to maintain
    minimum cost in the rewired roadmap, but the first feasible path to
    the goal will be kept in the non-rewired roadmap without updating or
    rewiring.

    Terminates when n_nodes have been added to both roadmaps, so goal
    may be reached by both roadmaps, or neither, or the rewired roadmap
    only.

    Equivalent to Algorithm II in the paper, modified for single-query
    planning.

    Parameters
    ------------
        save_dir: string
            Valid folder for saving data
        problem: Problem
            Problem class with associated dynamics, constraints, etc.
        x_0: np.ndarray
            Initial state mean
        P_0: np.ndarray
            Initial state covariance
        x_f: np.ndarray
            Goal state mean
        P_f: np.ndarray
            Goal state covariance
        graph_lims: tuple
            2-tuple of numpy arrays specifying minimum and maximum bounds
            of state space
        n_states: int
            Trajectory length for each edge in roadmap
        n_nodes: int
            Desired number of nodes in roadmap
        edge_controller: EdgeController
            Valid edge controller for constructing roadmap
        near_cutoff: float
            Proximity cutoff for neighboring nodes for edge rewiring
        max_nearby: int
            Maximum number of neighboring nodes to consider for rewiring

    Returns
    ---------
        rewired_graph: Graph
            Belief roadmap constructed with edge rewiring
        graph: Graph
            Belief roadmap constructed without edge rewiring, with the same
            set of node means as `rewired_graph`
    """
    # Define start and goal nodes
    start_node = Node(x_0, P_0, is_start=True)
    goal_node = Node(x_f, P_f, is_goal=True)

    # Initialize graph and rewired graph with only start and goal
    graph = Graph({start_node, goal_node}, set())
    rewired_graph = Graph(copy.deepcopy(graph.nodes), set())

    # Start node is newest, goal is not reached, and
    # haven't yet checked for a path from start to goal
    newest_node = start_node
    goal_reached = False
    goal_reached_no_rewire = False
    goal_checked = False

    # Get root filenames for saving data
    rewired_fname_root, no_rewire_fname_root = get_root_filenames(edge_controller)

    # Continue adding nodes until graph reaches n_nodes
    while len(rewired_graph.nodes) < n_nodes:
        # If haven't checked for a path from the newest node
        # to the goal, check for a path (in order to refine
        # the path to the goal node)
        if not goal_checked:
            node_mean = goal_node.mean
            new_rewired_graph, new_graph, success, new_node = add_node_and_rewire_roadmap(problem, rewired_graph.deepcopy(), graph.deepcopy(), node_mean, newest_node, n_states, edge_controller, near_cutoff, max_nearby, True)
            if success:
                # If goal hasn't yet been reached, accept path to goal
                # as long as goal is reached with covariance smaller
                # than that at goal node
                if not goal_reached:
                    if check_psd(goal_node.covariance-new_node.covariance):
                        rewired_graph = new_rewired_graph
                        goal_reached = True
                        # Save data
                        save_roadmap(save_dir, rewired_fname_root, rewired_graph)
                else:
                    # If goal has already been reached, accept new path to
                    # goal only if new path has smaller funnel tail at goal
                    # than previous path
                    current_goal_node = rewired_graph.get_goal_node()
                    if np.linalg.eigvals(new_node.covariance).max() < np.linalg.eigvals(current_goal_node.covariance).max():
                        rewired_graph = new_rewired_graph
                        # Save data
                        save_roadmap(save_dir, rewired_fname_root, rewired_graph)
            # If goal hasn't been reached in graph without rewiring,
            # check for a path to goal. But if the goal has been
            # reached, keep the existing path without updating.
            if not goal_reached_no_rewire:
                newest_node_no_rewire = graph.look_up_by_mean(newest_node.mean)
                new_graph, success = add_node_to_roadmap(problem, graph.deepcopy(), node_mean, newest_node_no_rewire, n_states, edge_controller, True)
                if success:
                    new_node = new_graph.get_goal_node()
                    # Only accept path if goal is reached with covariance
                    # smaller than that at goal node
                    if check_psd(goal_node.covariance-new_node.covariance):
                        graph = new_graph
                        goal_reached_no_rewire = True
                        # Save data
                        save_roadmap(save_dir, no_rewire_fname_root, graph)
            goal_checked = True
        else:
            # If path to goal is already up-to-date, try adding
            # another node to rewired_graph (with rewiring) and to 
            # graph (without rewiring)
            node_mean, parent = randomize_candidate_mean(problem, graph, graph_lims, problem.sample_radius_lims)
            rewired_graph, graph, success, new_node = add_node_and_rewire_roadmap(problem, rewired_graph, graph, node_mean, parent, n_states, edge_controller, near_cutoff, max_nearby, False)
            if success: # Need to see if we can now steer to goal
                goal_checked = False
                newest_node = new_node
                # Save data
                save_roadmap(save_dir, rewired_fname_root, rewired_graph)
                save_roadmap(save_dir, no_rewire_fname_root, graph)

    return rewired_graph, graph
