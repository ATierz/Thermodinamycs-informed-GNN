"""utils.py"""

import os
import numpy as np
import argparse
import torch
from sklearn import neighbors


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
        # for i in range(12):
        #     e = error[key][10 * i:(i + 1) * 10]
        #     error_mean = sum(e) / len(e)
        #     print('  ---' + key + ' = {:1.2e}'.format(error_mean))
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
