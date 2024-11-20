import os
import pickle
from src.covariance_steering import EdgeController

def get_root_filenames(edge_controller):
    """
    Get root filenames for saving roadmap data.

    Parameters
    ------------
        edge_controller: EdgeController
            Valid edge controller for roadmap construction

    Returns
    --------
        rewired_fname_root: string
            Root filename for rewired roadmaps constructed with `edge_controller`
        no_rewire_fname_root: string
            Root filename for non-rewired roadmaps constructed with `edge_controller`

    Raises
    --------
        ValueError
            If `edge_controller` isn't a valid edge controller. Currently
            supported edge controllers are EdgeController.BASELINE and
            EdgeController.ROBUST_SIGMA_POINT.
    """
    if edge_controller == EdgeController.BASELINE:
        rewired_fname_root = "rewired_ablation"
        no_rewire_fname_root = "baseline"
    elif edge_controller == EdgeController.ROBUST_SIGMA_POINT:
        rewired_fname_root = "revise"
        no_rewire_fname_root = "robust_ablation"
    else:
        raise ValueError("Edge controller is of unknown type.")
    return rewired_fname_root, no_rewire_fname_root

def save_roadmap(save_dir, root_filename, graph):
    """
    Save a roadmap to a file.

    Parameters
    ------------
        save_dir: string
            Directory in which to save roadmap
        root_filename: string
            Root filename for saving roadmap
        graph: Graph
            Roadmap to save
    """
    save_filename = os.path.join(save_dir, f"{root_filename}_{len(graph.nodes)}.pkl")
    pickle.dump(graph, open(save_filename, "wb"))


