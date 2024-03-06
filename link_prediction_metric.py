import os

import numpy as np
import torch
import torch.nn as nn
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_neighbors(train_data):
    neighbors = {}
    for head, tail in train_data:
        if head not in neighbors:
            neighbors[head] = set()
        neighbors[head].add(tail)
        if tail not in neighbors:
            neighbors[tail] = set()
        neighbors[tail].add(head)
    return neighbors


def get_node_candidates(all_nodes, neighbors, node):
    return list(all_nodes.difference(neighbors.get(node, set())))


def difference(ta, tb):
    vs, cs = torch.cat([ta, tb]).unique(return_counts=True)
    return vs[cs == 1]


def get_node_candidates_gpu(all_nodes, neighbors, node, default_value):
    return difference(all_nodes, neighbors.get(node, default_value))


def rank_data(data):
    _, idx, count = torch.unique(data, return_inverse=True, return_counts=True)
    return (torch.cumsum(count, 0) - 0.5 * count + 0.5)[idx]


def rank(node, true_candidate, node_reps, candidate_reps, candidates):
    node_t = torch.LongTensor([node]).to(device)
    true_candidate = torch.LongTensor([true_candidate]).to(device)
    node_tensor = node_reps(node_t).view(-1, 1)
    candidate_tensors = candidate_reps(torch.cat([candidates, true_candidate]).to(device))
    length = len(candidates)
    scores = torch.mm(candidate_tensors, node_tensor)
    negative_scores_numpy = -scores.view(1, -1)
    rank_ = rank_data(negative_scores_numpy)[0][-1].item()
    return rank_, length


def get_ranks(test_data, head_reps, tail_reps, neighbors, all_nodes):
    head_ranks = []
    tail_ranks = []
    head_lengths = []
    tail_lengths = []
    all_nodes = torch.tensor(list(all_nodes)).to(device)
    for k in neighbors:
        neighbors[k] = torch.tensor(list(neighbors[k])).to(device)
    default_value = torch.tensor([]).to(device)
    for head_node, tail_node in test_data:
        if head_node in all_nodes and tail_node in all_nodes:
            candidates = get_node_candidates_gpu(all_nodes, neighbors, head_node, default_value)
            head_rank, head_length = rank(head_node, tail_node, head_reps, tail_reps, candidates)
            head_ranks.append(head_rank)
            head_lengths.append(head_length)

            candidates = get_node_candidates_gpu(all_nodes, neighbors, tail_node, default_value)
            tail_rank, tail_length = rank(tail_node, head_node, tail_reps, head_reps, candidates)
            tail_ranks.append(tail_rank)
            tail_lengths.append(tail_length)

    return head_ranks, tail_ranks, head_lengths, tail_lengths


def get_all_nodes(data):
    all_nodes = set()
    for head, tail in data:
        all_nodes.add(head)
        all_nodes.add(tail)
    return all_nodes


def calc_mrr(head_ranks, tail_ranks, head_lengths):
    head_ranks_numpy = np.asarray(head_ranks)
    tail_ranks_numpy = np.asarray(tail_ranks)
    head_lengths_numpy = np.asarray(head_lengths)

    mrr = (np.mean(1 / head_ranks_numpy) + np.mean(1 / tail_ranks_numpy)) / 2
    recall_50 = ((head_ranks_numpy <= 50).sum() + (tail_ranks_numpy <= 50).sum()) / head_lengths_numpy.shape[0] / 2
    return mrr, recall_50


def evaluate_lp(data_root, model=None, test_data=None):
    if test_data is None:
        test_data = []
        with open(os.path.join(data_root, 'test.txt')) as fin:
            for line in fin:
                test_data.append([int(x) for x in line.split()[:2]])
    train_data = []
    with open(os.path.join(data_root, 'train.txt')) as fin:
        for line in fin:
            train_data.append([int(x) for x in line.split()[:2]])
    all_nodes = get_all_nodes(train_data)
    neighbors = get_neighbors(train_data)
    
    model = sorted(model.items(), key=lambda x: x[0])
    model = torch.tensor([x[1].tolist() for x in model], device=device)
    head_reps = nn.Embedding.from_pretrained(model)
    tail_reps = nn.Embedding.from_pretrained(model)
    head_ranks, tail_ranks, head_lengths, _ = get_ranks(
        test_data, head_reps, tail_reps, neighbors, all_nodes)
    result = calc_mrr(head_ranks, tail_ranks, head_lengths)
    return result