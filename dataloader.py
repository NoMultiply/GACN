import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def get_sparse_graph(n_nodes, all_neighbors, cache_path):
    try:
        pre_adj_mat = sp.load_npz(cache_path + '/s_pre_adj_mat.npz')
        norm_adj = pre_adj_mat
    except FileNotFoundError:
        us, vs = [], []
        for node, neighbors in all_neighbors.items():
            if len(neighbors) > 0:
                us.extend([node] * len(neighbors))
                vs.extend(neighbors)
        us = np.array(us)
        vs = np.array(vs)

        adj_mat = csr_matrix((np.ones(len(us)), (us, vs)), shape=(n_nodes, n_nodes))

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        sp.save_npz(cache_path + '/s_pre_adj_mat.npz', norm_adj)

    graph = convert_sp_mat_to_sp_tensor(norm_adj)
    return graph.coalesce()


def adj2graph(adj_mat):
    row_sum = np.array(adj_mat.sum(1)) + 1e-5
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    adj_matrix = norm_adj_tmp.dot(d_mat_inv)

    graph = convert_sp_mat_to_sp_tensor(adj_matrix)
    return graph.coalesce()


def adj2graph_torch(adj_mat):
    row_sum = adj_mat.sum(1) + 1e-5
    d_inv = row_sum.pow(-0.5)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat_inv = torch.diag(d_inv)

    return torch.mm(torch.mm(d_mat_inv, adj_mat), d_mat_inv)


def read_edges(filename):
    all_neighbors = {}
    nodes = set()
    train_edges = read_edges_from_file(filename)

    for u, v in train_edges:
        nodes.add(u)
        nodes.add(v)
        if u not in all_neighbors:
            all_neighbors[u] = []
        if v not in all_neighbors:
            all_neighbors[v] = []
        all_neighbors[u].append(v)
        all_neighbors[v].append(u)

    for k in all_neighbors:
        all_neighbors[k] = list(set(all_neighbors[k]))

    return len(nodes), all_neighbors, sorted(list(nodes))


def read_features(filename):
    if not os.path.exists(filename):
        return None, None
    features = {}
    n_dim = 0
    with open(filename) as fin:
        for line in fin:
            items = line.strip().split()
            node = int(items[0])
            feature = list(map(float, items[1:]))
            features[node] = feature
            n_dim = len(feature)
    return features, n_dim


def read_edges_from_file(filename):
    with open(filename, "r") as fin:
        edges = [[int(x) for x in line.strip().split()] for line in fin if line.strip()]
    return edges
