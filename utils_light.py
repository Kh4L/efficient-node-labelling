import argparse
import random
import subprocess

import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor, ToUndirected
from torch_geometric.utils import (degree, to_undirected,
                                   train_test_split_edges)

def get_dataset(root, name: str):
    dataset = PygLinkPropPredDataset(name=name, root=root)
    data = dataset[0]
    """
        SparseTensor's value is NxNx1 for collab. due to edge_weight is |E|x1
        NeuralNeighborCompletion just set edge_weight=None
        ELPH use edge_weight
    """

    split_edge = dataset.get_edge_split()
    if 'edge_weight' in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    if 'edge' in split_edge['train']:
        key = 'edge'
    else:
        key = 'source_node'
    print("-"*20)
    print(f"train: {split_edge['train'][key].shape[0]}")
    print(f"{split_edge['train'][key]}")
    print(f"valid: {split_edge['valid'][key].shape[0]}")
    print(f"test: {split_edge['test'][key].shape[0]}")
    print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    data = ToUndirected()(data)
    data = ToSparseTensor(remove_edge_index=False)(data)
    data.full_adj_t = data.adj_t
    # make node feature as float
    if data.x is not None:
        data.x = data.x.to(torch.float)
    if name != 'ogbl-ddi':
        del data.edge_index
    return data, split_edge

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def data_summary(name: str, data: Data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    n_degree = data.adj_t.sum(dim=1).to(torch.float)
    avg_degree = n_degree.mean().item()
    degree_std = n_degree.std().item()
    max_degree = n_degree.max().long().item()
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    if data.x is not None:
        attr_dim = data.x.shape[1]
    else:
        attr_dim = '-' # no attribute

    print("-"*30+'Dataset and Features'+"-"*60)
    print("{:<10}|{:<10}|{:<10}|{:<15}|{:<15}|{:<15}|{:<10}|{:<15}"\
        .format('Dataset','#Nodes','#Edges','Avg. node deg.','Std. node deg.','Max. node deg.', 'Density','Attr. Dimension'))
    print("-"*110)
    print("{:<10}|{:<10}|{:<10}|{:<15.2f}|{:<15.2f}|{:<15}|{:<9.4f}%|{:<15}"\
        .format(name, num_nodes, num_edges, avg_degree, degree_std, max_degree, density*100, attr_dim))
    print("-"*110)

def initial_embedding(data, hidden_channels, device):
    embedding= torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    return embedding

def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
