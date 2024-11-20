import copy

import numpy as np

from math_utils import check_psd

class Node:
    """
    Represents a multidimensional Gaussian state distribution
    in a graph-structured belief roadmap.

    Parameters
    -----------
        mean: np.ndarray
            Mean of the Gaussian distribution
        covariance: np.ndarray
            Covariance of the Gaussian distribution, must be PSD
        is_start: bool
            True if start node, False otherwise
        is_goal: bool
            True if goal node, False otherwise
    """
    def __init__(self, mean, covariance, is_start=False, is_goal=False):
        self.mean = mean
        self.covariance = covariance
        self.is_start = is_start
        self.is_goal = is_goal

        # Check to make sure covariance is PSD
        if covariance is not None:
            check_psd(covariance)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        mean_match = np.allclose(self.mean, other.mean)
        if self.covariance is None and other.covariance is None:
            covariance_match = True
        elif self.covariance is None or other.covariance is None:
            covariance_match = False
        else:
            covariance_match = np.allclose(self.covariance, other.covariance)
        start_match = self.is_start == other.is_start
        goal_match = self.is_goal == other.is_goal

        return mean_match and covariance_match and start_match and goal_match

    def __hash__(self):
        covariance = tuple(map(tuple,self.covariance)) if self.covariance is not None else None
        return hash((tuple(self.mean.tolist()), covariance, self.is_start, self.is_goal))

class Edge:
    """
    Represents a directed edge in a belief roadmap between two
    Gaussian state distributions. Each edge is associated with a
    feedback control policy and a discrete-time Gaussian state
    trajectory between its start and end nodes.

    Parameters
    -----------
        start_node: Node
            Start node of the edge
        end_node: Node
            End node of the edge
        mean: np.ndarray
            Array of means of intermediate Gaussian states
        covariance: np.ndarray
            Array of covariances of intermediate Gaussian states
        ff_ctrl: np.ndarray
            Open-loop control between start and end nodes
        fb_ctrl: np.ndarray
            Feedback control gain between start and end nodes
    """
    def __init__(self, start_node, end_node, mean, covariance, ff_ctrl, fb_ctrl):
        self.start_node = start_node
        self.end_node = end_node
        self.mean = mean
        self.covariance = covariance
        self.ff_ctrl = ff_ctrl
        self.fb_ctrl = fb_ctrl

class Graph:
    """
    Represents a graph-structured belief roadmap, where the
    nodes in the roadmap are Gaussian distributions in the
    state space, and the directed edges in the roadmap are
    control policies steering between nodes.

    Parameters
    ------------
        nodes: set
            Set of Node objects representing nodes
        edges: set
            Set of Edge objects representing edges
    """
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def add_node(self, node):
        """
        Add node to the graph.

        Parameters
        -----------
            node: Node
                Node to be added to the graph.
        """
        self.nodes.add(node)

    def add_edge(self, edge):
        """
        Add edge to the graph, assuming the start and end
        nodes of the edge are already in the graph.

        Parameters
        ------------
            edge: Edge
                Edge to be added to the graph.
        """
        if edge.start_node not in self.nodes:
            raise Exception("Edge starts at a node that isn't in the graph")
        if edge.end_node not in self.nodes:
            raise Exception("Edge ends at a node that isn't in graph")
        self.edges.add(edge)

    def get_start_node(self):
        """
        Get the start node in the graph. Throws an error unless
        exactly one start node is in the graph.

        Returns
        ----------
            start_node: Node
                Start node of the graph
        """
        start_nodes = []
        for node in self.nodes:
            if node.is_start:
                start_nodes.append(node)
        if len(start_nodes) == 0:
            raise Exception("No start node found")
        if len(start_nodes) > 1:
            raise Exception("Multiple start nodes found")
        return start_nodes[0]

    def get_goal_node(self):
        """
        Get the goal node in the graph. Throws an error unless
        exactly one goal node is in the graph.

        Returns
        ----------
            goal_node: Node
                Goal node of the graph
        """
        goal_nodes = []
        for node in self.nodes:
            if node.is_goal:
                goal_nodes.append(node)
        if len(goal_nodes) == 0:
            raise Exception("No goal node found")
        if len(goal_nodes) > 1:
            raise Exception("Multiple goal nodes found")
        return goal_nodes[0]

    def get_parent(self, node):
        """
        Get the parent of a node in the graph. Throws
        an error if the node isn't in the graph.

        Parameters
        ------------
            node: Node
                Child node to look up parent
            parent: Node
                Parent of child node (None if child is start)
        """
        if node not in self.nodes:
            raise Exception("Node is not in graph")
        for edge in self.edges:
            if edge.end_node == node:
                return edge.start_node
        if node.is_start: #OK to have no parent
            return None
        raise Exception("Parent not found")

    def look_up_by_mean(self, node_mean):
        """
        Look up a node in a graph by its mean. Throws
        an error unless exactly one node with the given
        mean is in the graph.

        Parameters
        -----------
            node_mean: np.ndarray
                Mean of node to look up
            node: Node
                Node in graph with node_mean as its mean
        """
        matching_nodes = []
        for node in self.nodes:
            if np.allclose(node.mean, node_mean):
                matching_nodes.append(node)
        if len(matching_nodes) == 0:
            raise Exception("No matching nodes")
        if len(matching_nodes) > 1:
            raise Exception("Multiple nodes match")
        return matching_nodes[0]

    def get_children(self, node):
        """
        Get children of a node in the graph.

        Parameters
        -----------
            node: Node
                Node to look up children

        Returns
        ---------
            children: set
                Set of child nodes of node (can be empty)
        """
        if node not in self.nodes:
            raise Exception("Node is not in graph")
        children = set()
        for edge in self.edges:
            if edge.start_node == node:
                children.add(edge.end_node)
        return children

    def get_ancestors(self, node):
        """
        Get ancestors of a node in the graph, recursively,
        all the way back to the start node.

        Parameters
        -----------
            node: Node
                Node to look up ancestors
        Returns
        --------
            ancestors: set
                Ancestors of node
        """
        ancestors = set()
        parent = self.get_parent(node)
        while parent is not None:
            ancestors.add(parent)
            parent = self.get_parent(parent)
        return ancestors

    def get_descendants(self, node):
        """
        Get descendants of a node in the graph, recursively,
        all the way down to leaf nodes.

        Parameters
        ------------
            node: Node
                Node to find descendants of
        Returns
        --------
            descendants: set
                Set of descendants of node
        """
        descendants = set()
        children = self.get_children(node)
        while len(children) > 0:
            descendants = descendants.union(children)
            new_children = set()
            for child in children:
                grandchildren = self.get_children(child)
                new_children = new_children.union(grandchildren)
            children = copy.deepcopy(new_children)
        return descendants

    def trim(self):
        """
        Return minimal subgraph which connects the start node
        to the goal node. Assumes a path exists from the start
        to the goal.

        Returns 
        ------------
            trimmed_graph: Graph
                Minimal subgraph connecting start to goal
        """
        current_node = -1
        relevant_nodes = []
        relevant_edges = []
        for edge in self.edges:
            if edge.end_node.is_goal:
                current_node = edge.start_node
                relevant_edges.append(edge)
                relevant_nodes.append(edge.start_node)
                relevant_nodes.append(edge.end_node)

        while not current_node.is_start:
            for edge in self.edges:
                if edge.end_node == current_node:
                    current_node = edge.start_node
                    relevant_edges.append(edge)
                    relevant_nodes.append(edge.start_node)
                    break

        return Graph(set(relevant_nodes), set(relevant_edges))

    def get_plan_to_goal(self):
        """
        Trace edges from start to goal, and extract
        planned control and state. Can probably be refactored
        to traverse over trimmed graph only.

        Returns
        ---------
            u_traj: list 
                List of planned open-loop control for each
                edge from start to goal
            K_traj: list
                List of planned feedback control for each edge
            x_traj: list
                List of mean state trajectory for each edge
            P_traj: list
                List of state covariance for each edge
        """
        current_node = -1
        u_traj = []
        K_traj = []
        P_traj = []
        x_mean_traj = []
        goal_node = -1
        for node in self.nodes:
            if node.is_goal:
                goal_node = node
        for edge in self.edges:
            if edge.end_node.is_goal:
                u_traj.append(edge.ff_ctrl)
                K_traj.append(edge.fb_ctrl)
                P_traj.append(edge.covariance)
                x_mean_traj.append(edge.mean)
                current_node = edge.start_node

        while not current_node.is_start:
            for edge in self.edges:
                if edge.end_node == current_node:
                    u_traj.insert(0, edge.ff_ctrl)
                    K_traj.insert(0, edge.fb_ctrl)
                    P_traj.insert(0, edge.covariance)
                    x_mean_traj.insert(0, edge.mean)
                    current_node = edge.start_node
                    break

        return u_traj, K_traj, x_mean_traj, P_traj
    
    def deepcopy(self):
        """
        Deepcopy graph to a new Graph object.

        Returns
        --------
            graph_copy: Graph
                Deepcopy of self
        """
        node_copy = copy.deepcopy(self.nodes)
        edge_copy = copy.deepcopy(self.edges)
        return Graph(node_copy, edge_copy)
