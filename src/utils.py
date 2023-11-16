"""utils.py"""

import os
import shutil
import numpy as np
import argparse
import torch
from sklearn import neighbors
import datetime

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_error(error):
    print('State Variable (L2 relative error)')
    lines = []

    for key in error.keys():
        e = error[key]
        error_mean = sum(e) / len(e)
        line = '  ' + key + ' = {:1.2e}'.format(error_mean)
        print(line)
        lines.append(line)
    return lines


def compute_connectivity(positions, radius, add_self_edges):
    """Get the indices of connected edges with radius connectivity.
    https://github.com/deepmind/deepmind-research/blob/master/learning_to_simulate/connectivity_utils.py
    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
      radius: Radius of connectivity.
      add_self_edges: Whether to include self edges or not.
    Returns:
      senders indices [num_edges_in_graph]
      receiver indices [num_edges_in_graph]
    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        # Remove self edges.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return torch.from_numpy(np.array([senders, receivers]))

def generate_folder(output_dir_exp, pahtDInfo, pathWeights):
    output_dir_exp = output_dir_exp + '_' +datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if os.path.exists(output_dir_exp):
        print("The experiment path exists.")
        action = input("¿Would you like to create a new one (c) or overwrite (o)?")
        if action == 'c':
            output_dir_exp = output_dir_exp + '_new'
            os.makedirs(output_dir_exp, exist_ok=True)
    else:
        os.makedirs(output_dir_exp, exist_ok=True)

    shutil.copyfile('src\gnn.py', os.path.join(output_dir_exp, 'gnn.py'))
    shutil.copyfile(os.path.join('data', pahtDInfo), os.path.join(output_dir_exp, os.path.basename(pahtDInfo)))
    shutil.copyfile(os.path.join('data', pathWeights), os.path.join(output_dir_exp, os.path.basename(pathWeights)))
    return output_dir_exp
