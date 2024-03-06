import random
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch import nn
from torch.nn import functional as nf

from dataloader import adj2graph, adj2graph_torch
from link_prediction_metric import evaluate_lp
from node_classification_metric import evaluate_clf
from utils import EarlyStopper, color_print, progress, randint_choice, shuffle

LABEL_EDGE_DROPOUT = 1
LABEL_GENERATE = 0
SAMPLE_RATE = (2, 3)
RATIO_NOISE = 0.1


def _generate_edge_dropout_graph(n_nodes, us, vs, ratio, device, return_matrix=False):
    n = int(len(us) * (ratio + (random.random() - 0.5) * RATIO_NOISE))
    indices = torch.randperm(len(us), device=device)[:n]
    us = torch.tensor(us)[indices]
    vs = torch.tensor(vs)[indices]
    adj_mat = torch.zeros([n_nodes, n_nodes]).to(device)
    adj_mat[us, vs] = 1
    if return_matrix:
        return adj_mat
    return adj2graph_torch(adj_mat)



class LightGCN(nn.Module):
    def __init__(self, n_nodes, graph, device, n_dim=64, n_layers=3, keep_prob=0.6, dropout=0):
        super(LightGCN, self).__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.device = device
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        self.embeddings = torch.nn.Embedding(num_embeddings=self.n_nodes, embedding_dim=self.n_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)
        self.graph = graph
        self.dropout = dropout
        self.t_dropout = nn.Dropout(dropout)

    def __dropout_x(self, x, keep_prob):
        if not x.is_sparse:
            return self.t_dropout(x)
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, graph=None):
        if graph is None:
            graph = self.graph
        return self.__dropout_x(graph, keep_prob)

    def forward(self, training, graph=None, all_emb=None):
        if graph is None:
            graph = self.graph
        if all_emb is None:
            all_emb = self.embeddings.weight
        emb_list = [all_emb]
        g_dropped = self.__dropout(self.keep_prob, graph) if training and self.dropout else graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_dropped, all_emb) if g_dropped.is_sparse else torch.mm(g_dropped, all_emb)

            emb_list.append(all_emb)
        emb_list = torch.stack(emb_list, dim=1)
        light_out = torch.mean(emb_list, dim=1)
        return light_out


class Generator(nn.Module):
    def __init__(self, n_nodes, nodes, us, vs, device, n_dim=256, temperature=0.0001, lambda_new=0, lambda_count=0,
                 ssl_ratio=0.5, n_g_nodes=1200, init_rate=0.5):
        super(Generator, self).__init__()
        self.n_nodes = n_nodes
        self.nodes = nodes
        self.us = us
        self.vs = vs
        self.device = device
        self.n_dim = n_dim
        self.temperature = temperature
        self.lambda_new = lambda_new
        self.lambda_count = lambda_count
        self.ssl_ratio = ssl_ratio
        n_g_nodes = min(n_g_nodes, n_nodes)
        self.n_g_nodes = n_g_nodes

        self.g_nodes, self.o_us, self.o_vs, _, _ = self._partition_nodes()
        self.n_candidate = n_g_nodes * n_g_nodes
        self.n_target = int((len(us) - len(self.o_us)) * ssl_ratio)

        self.new_indices = []
        edges = {}
        g_us, g_vs = [], []
        _g_us, _g_vs = [], []
        for u, v in zip(us, vs):
            edges[(u, v)] = 1
        for idx in range(self.n_candidate):
            i, j = idx // n_g_nodes, idx % n_g_nodes
            if (self.g_nodes[i], self.g_nodes[j]) not in edges:
                self.new_indices.append(idx)
            else:
                g_us.append(i)
                g_vs.append(j)
                _g_us.append(self.g_nodes[i])
                _g_vs.append(self.g_nodes[j])
        self.g_us = np.array(g_us)
        self.g_vs = np.array(g_vs)
        self._g_us = np.array(_g_us)
        self._g_vs = np.array(_g_vs)
        assert len(g_us) == len(self._g_us) and len(g_vs) == len(self._g_vs)
        self.g_mat = csr_matrix((np.ones_like(self.g_us), (self.g_us, self.g_vs)),
                                shape=(n_g_nodes, n_g_nodes)).toarray()

        n_all = n_nodes * n_nodes
        color_print(f'Existing Edges: {len(us)} / {n_all} = {len(us) / n_all * 100:.2f}%')
        color_print(f'Candidate Edges: {self.n_candidate} / {n_all} = {self.n_candidate / n_all * 100:.2f}%')
        color_print(f'New Edges: {len(self.new_indices)} / {n_all} = {len(self.new_indices) / n_all * 100:.2f}%')

        if init_rate >= 0:
            n_existing_edges = int(self.n_target * init_rate)
            n_new_edges = int(self.n_target * (1 - init_rate))
            weight_existing = n_existing_edges / (self.n_candidate - len(self.new_indices))
            weight_new = n_new_edges / len(self.new_indices)
            weights = torch.ones(self.n_candidate).to(device) * weight_existing
            weights[self.new_indices] = weight_new
            self.weights = nn.Parameter(weights)
        else:
            self.weights = nn.Parameter(torch.rand(self.n_candidate).to(device))

    def _partition_nodes(self):
        if self.n_g_nodes >= self.n_nodes:
            g_nodes, o_us, o_vs, g_us, g_vs = self.nodes, [], [], self.us, self.vs
        else:
            degree = {}
            for u in self.us:
                degree[u] = degree.get(u, 0) + 1
            for v in self.vs:
                degree[v] = degree.get(v, 0) + 1
            g_nodes = list(x[0] for x in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:self.n_g_nodes])
            g_nodes_set = set(g_nodes)
            o_us, o_vs = [], []
            g_us, g_vs = [], []
            for u, v in zip(self.us, self.vs):
                if u not in g_nodes_set or v not in g_nodes_set:
                    o_us.append(u)
                    o_vs.append(v)
                else:
                    g_us.append(u)
                    g_vs.append(v)
        color_print(f'Nodes (g_nodes + o_nodes): {len(g_nodes)} + {self.n_nodes - self.n_g_nodes} = {self.n_nodes}')
        color_print(f'Edges (g_edges + o_edges): {len(g_us)} + {len(o_us)} = {len(self.us)}')
        return np.array(g_nodes), np.array(o_us), np.array(o_vs), np.array(g_us), np.array(g_vs)

    def forward(self):
        noises = torch.rand(self.n_candidate, device=self.device)
        matrix = self.weights - noises
        matrix = torch.sigmoid(matrix / self.temperature)
        return matrix

    def generate_g_graph(self, make_binary=False):
        g_matrix = self.forward().view(self.n_g_nodes, self.n_g_nodes)
        if make_binary:
            g_matrix[g_matrix >= 0.5] = 1
            g_matrix[g_matrix != 1] = 0
        return adj2graph_torch(g_matrix)

    def generate_g_edge_dropout_graph(self):
        return _generate_edge_dropout_graph(self.n_g_nodes, self.g_us, self.g_vs, self.ssl_ratio, self.device)

    def generate_adj_graph(self):
        matrix = self.forward().view(self.n_g_nodes, self.n_g_nodes)
        idx = torch.zeros_like(matrix)
        idx[matrix >= 0.5] = 1
        u_idx, v_idx = idx.to_sparse().indices().tolist()
        us = self.g_nodes[u_idx]
        vs = self.g_nodes[v_idx]
        mat = csr_matrix((np.ones_like(us), (us, vs)), shape=(self.n_nodes, self.n_nodes))
        if len(self.o_us):
            keep_idx = randint_choice(len(self.o_us), size=int(len(self.o_us) * self.ssl_ratio), replace=False)
            us = self.o_us[keep_idx]
            vs = self.o_vs[keep_idx]
            mat += csr_matrix((np.ones_like(us), (us, vs)), shape=(self.n_nodes, self.n_nodes))

        return adj2graph(mat).to(self.device), adj2graph_torch(matrix).to(self.device)

    def generate_edge_dropout_graph(self):
        g_keep_idx = randint_choice(len(self.g_us), size=int(len(self.g_us) * self.ssl_ratio), replace=False)
        g_us = self.g_us[g_keep_idx]
        g_vs = self.g_vs[g_keep_idx]
        g_ratings = np.ones_like(g_us, dtype=np.float32)
        g_adj_mat = sp.csr_matrix((g_ratings, (g_us, g_vs)), shape=(self.n_g_nodes, self.n_g_nodes))

        o_keep_idx = randint_choice(len(self.o_us), size=int(len(self.o_us) * self.ssl_ratio), replace=False)
        o_us = self.o_us[o_keep_idx]
        o_vs = self.o_vs[o_keep_idx]

        us = np.concatenate([self._g_us[g_keep_idx], o_us])
        vs = np.concatenate([self._g_vs[g_keep_idx], o_vs])
        ratings = np.ones_like(us, dtype=np.float32)
        adj_mat = sp.csr_matrix((ratings, (us, vs)), shape=(self.n_nodes, self.n_nodes))
        return adj2graph(adj_mat).to(self.device), adj2graph(g_adj_mat).to(self.device)

    def reg_loss(self):
        weights = self.weights
        edge_count_loss = (weights.abs().sum() - self.n_target).abs()
        new_edge_loss = weights[self.new_indices].abs().sum()
        return edge_count_loss * self.lambda_count + new_edge_loss * self.lambda_new

    def loss(self, discriminator, batch_size):
        scores = []
        labels = []
        for _ in range(batch_size):
            graph = self.generate_g_graph()
            score = discriminator.forward(graph, False)
            scores.append(score)
            labels.append(LABEL_EDGE_DROPOUT)
        scores = torch.cat(scores)
        labels = torch.FloatTensor(labels).to(self.device)
        loss = discriminator.loss_fn(scores, labels) + self.reg_loss()
        return loss, scores.mean()

    def warmup(self, batch_size):
        reg_loss = []
        for _ in range(batch_size):
            reg_loss.append(self.reg_loss())
        reg_loss = torch.stack(reg_loss).mean()
        return reg_loss


class Discriminator(nn.Module):
    def __init__(self, n_nodes, device, n_dim):
        super(Discriminator, self).__init__()
        self.device = device
        self.gcn = LightGCN(n_nodes, None, device, n_dim)
        self.gcn.embeddings.weight.requires_grad_(False)
        self.linear = nn.Sequential(
            nn.Linear(n_dim * 2, n_dim),
            nn.Sigmoid(),
            nn.Linear(n_dim, n_dim),
            nn.Dropout(0.1),
            nn.Sigmoid(),
            nn.Linear(n_dim, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, graph, training):
        all_embeddings = self.gcn.forward(training, graph)
        graph_embedding = torch.cat([torch.sum(all_embeddings, dim=0), torch.max(all_embeddings, dim=0)[0]])
        return self.linear(graph_embedding)

    def loss(self, graphs, labels):
        scores = []
        for graph in graphs:
            scores.append(self.forward(graph, True))
        scores = torch.cat(scores)
        labels = torch.FloatTensor(labels).to(self.device)
        return self.loss_fn(scores, labels)

    def evaluate(self, graphs, labels, acc_only=True):
        scores = []
        for graph in graphs:
            scores.append(self.forward(graph, True).item())
        scores = np.array(scores)
        median = 0.5
        index_pos = scores >= median
        index_neg = scores < median
        scores[index_pos] = 1
        scores[index_neg] = 0
        accuracy = accuracy_score(labels, scores)
        if not acc_only:
            precision = precision_score(labels, scores, zero_division=0.0)
            recall = recall_score(labels, scores, zero_division=0.0)
            f1 = f1_score(labels, scores, zero_division=0.0)
            report = classification_report(labels, scores, zero_division=0.0)
            return accuracy, precision, recall, f1, report
        return accuracy


class GCL(nn.Module):
    def __init__(self, n_nodes, nodes, all_neighbors, us, vs, graph, device, dataset, n_dim,
                 temperature=0.5, lambda_ssl=1, lambda_bpr=1, lambda_reg=0.0001, node_features=None):
        super(GCL, self).__init__()
        self.n_nodes = n_nodes
        self.nodes = nodes
        self.nodes_tensor = torch.tensor(self.nodes, device=device)
        self.all_neighbors = all_neighbors
        self.us = us
        self.vs = vs
        self.graph = graph
        self.device = device
        self.dataset = dataset
        self.temperature = temperature
        self.lambda_ssl = lambda_ssl
        self.lambda_bpr = lambda_bpr
        self.lambda_reg = lambda_reg
        self.gcn = LightGCN(n_nodes, graph, device, n_dim)
        if node_features:
            color_print('Using node features to init lgn with dim =', n_dim)
            for node in node_features:
                self.gcn.embeddings.weight.data[node] = torch.FloatTensor(node_features[node]).to(device)

        self.to(device)

    def get_all_embeddings(self, training, graph=None):
        return self.gcn.forward(training, graph)

    def forward(self, training, graph1, graph2, nodes=None):
        # ========================= ssl loss ========================= #
        all_embeddings1 = self.get_all_embeddings(training, graph1)
        all_embeddings2 = self.get_all_embeddings(training, graph2)

        if nodes is not None:
            all_embeddings1 = all_embeddings1[nodes]
            all_embeddings2 = all_embeddings2[nodes]

        normalize_all_embeddings1 = nf.normalize(all_embeddings1, 1)
        normalize_all_embeddings2 = nf.normalize(all_embeddings2, 1)

        pos_score = torch.sum(normalize_all_embeddings1 * normalize_all_embeddings2, dim=1)
        ttl_score = torch.matmul(normalize_all_embeddings1, normalize_all_embeddings2.t())
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.sum(torch.exp(ttl_score / self.temperature), dim=1)
        ssl_loss = -torch.sum(torch.log(pos_score / ttl_score))

        # ========================= bpr loss ========================= #
        if nodes is None:
            nodes = self.nodes_tensor
        node_list = []
        pos_list = []
        neg_list = []
        for u in nodes.tolist():
            pos_list.extend(self.all_neighbors[u])
            for _ in range(len(self.all_neighbors[u])):
                node_list.append(u)
                while True:
                    neg = random.choice(self.nodes)
                    if neg in self.all_neighbors[u]:
                        continue
                    else:
                        neg_list.append(neg)
                        break
        node_list = torch.tensor(node_list, device=self.device)
        pos_list = torch.tensor(pos_list, device=self.device)
        neg_list = torch.tensor(neg_list, device=self.device)
        embeddings = self.get_all_embeddings(True, self.graph)
        u_embeddings = embeddings[node_list]
        v_embeddings = embeddings[pos_list]
        n_embeddings = embeddings[neg_list]
        u_emb0 = self.gcn.embeddings(node_list)
        v_emb0 = self.gcn.embeddings(pos_list)
        n_emb0 = self.gcn.embeddings(neg_list)
        reg_loss = (1 / 2) * (u_emb0.norm(2).pow(2) +
                              v_emb0.norm(2).pow(2) +
                              n_emb0.norm(2).pow(2)) / float(nodes.size(0))

        pos_scores = torch.mul(u_embeddings, v_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(u_embeddings, n_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return ssl_loss * self.lambda_ssl + bpr_loss * self.lambda_bpr + reg_loss * self.lambda_reg

    def do_valid(self, valid_edges):
        model = {}
        embeddings = self.get_all_embeddings(False).detach().cpu().numpy()
        for i, embedding in enumerate(embeddings):
            model[i] = embedding
        mrr, _ = evaluate_lp(f'data/{self.dataset}', model, test_data=valid_edges)
        return mrr

    def get_embeddings(self):
        embeddings = self.get_all_embeddings(False).detach().cpu().numpy()
        model = {i: embedding for i, embedding in enumerate(embeddings)}
        return model


class GACN(nn.Module):
    def __init__(self, n_nodes, nodes, all_neighbors, graph, device, dataset, args, node_features=None):
        super(GACN, self).__init__()
        n_dim, ssl_ratio, lambda_new, lambda_count, n_g_nodes, temperature, st, init_rate = \
            args.n_dim, args.ssl_ratio, args.lambda_new, args.lambda_count, args.n_g_nodes, args.temperature, \
            args.st, args.init_rate
        self.n_nodes = n_nodes
        self.nodes = nodes
        self.all_neighbors = all_neighbors
        self.graph = graph
        self.device = device
        self.dataset = dataset
        self.n_dim = n_dim
        self.ssl_ratio = ssl_ratio
        self.st = st
        self.degree_map = {}

        us, vs = [], []
        for node, neighbors in all_neighbors.items():
            self.degree_map[node] = len(neighbors)
            if len(neighbors) > 0:
                us.extend([node] * len(neighbors))
                vs.extend(neighbors)
        self.us = np.array(us)
        self.vs = np.array(vs)

        self.generator = Generator(n_nodes, nodes, self.us, self.vs, device, n_dim, temperature, lambda_new,
                                   lambda_count, ssl_ratio, n_g_nodes, init_rate)
        self.g_nodes = torch.tensor(self.generator.g_nodes).to(self.device)
        self.discriminator = Discriminator(self.generator.n_g_nodes, device, n_dim)

        self.gcl = GCL(n_nodes, nodes, all_neighbors, self.us, self.vs, graph, device, dataset, n_dim,
                       args.cl_temperature, args.cl_lambda_ssl, args.cl_lambda_bpr, args.cl_lambda_reg, node_features)
        self.to(device)
        self.eval()

    def prepare_discriminator_data(self, n_samples):
        graphs, labels = [], []
        for i in range(n_samples):
            if i % SAMPLE_RATE[1] < SAMPLE_RATE[0]:
                graphs.append(self.generator.generate_g_edge_dropout_graph())
                labels.append(LABEL_EDGE_DROPOUT)
            else:
                graphs.append(self.generator.generate_g_graph(make_binary=True))
                labels.append(LABEL_GENERATE)
        return graphs, labels

    def do_train(self, args, valid_edges):
        optimizer_params = {'params': filter(lambda param: param.requires_grad, self.generator.parameters()),
                            'lr': args.lr_g}
        if args.wd_g:
            optimizer_params['weight_decay'] = args.wd_g
        optimizer_g = torch.optim.Adam(**optimizer_params)

        optimizer_params = {'params': self.discriminator.linear.parameters(), 'lr': args.lr_d}
        if args.wd_d:
            optimizer_params['weight_decay'] = args.wd_d
        optimizer_d = torch.optim.Adam(**optimizer_params)

        optimizer_params = {'params': self.gcl.gcn.parameters(), 'lr': args.lr_cl}
        if args.wd_cl:
            optimizer_params['weight_decay'] = args.wd_cl
        optimizer_cl = torch.optim.Adam(**optimizer_params)

        outer_early_stopper = EarlyStopper(
            args.patience,
            lambda: deepcopy(self.state_dict()),
            _print=color_print)
        for epoch in range(1, args.n_epoch + 1):
            color_print('\n# ************** Begin: Train Discriminator ************** #')
            self.generator.eval()
            self.discriminator.linear.train()
            self.gcl.gcn.eval()

            self.discriminator.gcn.embeddings.weight = torch.nn.Parameter(
                self.gcl.gcn.embeddings.weight[self.g_nodes])

            data_iter = progress(range(1, args.n_epoch_d + 1))
            for epoch_d in data_iter:
                loss_list = []
                for _ in range(args.iter_d):
                    with torch.no_grad():
                        graphs, labels = self.prepare_discriminator_data(args.bs_d)
                    graphs, labels = shuffle(graphs, labels)                    
                    loss = self.discriminator.loss(graphs, labels)

                    optimizer_d.zero_grad()
                    loss.backward()
                    optimizer_d.step()

                    loss_list.append(loss.item())

                if epoch_d % args.ri_d == 0:
                    data_iter.write(f'[Iter {epoch}] [Discriminator] [Epoch {epoch_d}] '
                                    f'loss = {np.nanmean(loss_list)}') 
            data_iter.close()

            
            # color_print('# ============== End: Train Discriminator ============== #')

            color_print('\n# ************** Begin: Train CL ************** #')
            self.generator.eval()
            self.discriminator.linear.eval()
            self.gcl.gcn.train()

            data_iter = progress(range(1, args.n_epoch_cl + 1))
            early_stopper = EarlyStopper(args.patience_cl, lambda: deepcopy(self.gcl.state_dict()))

            for epoch_cl in data_iter:
                graph1, g_graph1 = self.generator.generate_adj_graph()
                graph2, g_graph2 = self.generator.generate_edge_dropout_graph()
                with torch.no_grad():
                    score1 = self.discriminator.forward(g_graph1, False).item()
                    score2 = self.discriminator.forward(g_graph2, False).item()
                if abs(score2 - score1) > self.st:
                    continue
                nodes = None
                batch_nodes = None
                if args.bs_cl < self.n_nodes:
                    nodes = self.nodes.copy()
                    random.shuffle(nodes)
                    nodes = torch.tensor(nodes).to(device=self.device)
                loss_list = []
                for i in range(0, self.n_nodes, args.bs_cl):
                    if nodes is not None:
                        batch_nodes = nodes[i:i + args.bs_cl]

                    loss = self.gcl.forward(training=True, graph1=graph1, graph2=graph2, nodes=batch_nodes)

                    optimizer_cl.zero_grad()
                    loss.backward()
                    optimizer_cl.step()
                    loss_list.append(loss.item())

                if epoch_cl % args.ri_cl == 0:
                    if args.task == 'lp':
                        self.gcl.gcn.eval()
                        with torch.no_grad():
                            score = self.gcl.do_valid(
                                valid_edges if args.n_valid >= len(valid_edges) else random.sample(valid_edges, args.n_valid))
                        self.gcl.gcn.train()
                        data_iter.write(f'[Iter {epoch}] [Contrastive Learning] [Epoch {epoch_cl}] '
                                        f'loss = {np.nanmean(loss_list)}, mrr = {score}')
                        if not early_stopper.update(score):
                            break
                    else:
                        data_iter.write(f'[Iter {epoch}] [Contrastive Learning] [Epoch {epoch_cl}] '
                                        f'loss = {np.nanmean(loss_list)}')

            data_iter.close()
            if args.task == 'lp' and early_stopper.best_model:
                self.gcl.load_state_dict(early_stopper.best_model)

            if args.task == 'lp':
                if not outer_early_stopper.update(early_stopper.best_score):
                    break
            # color_print('# ============== End: Train CL ============== #')

            color_print('\n# ************** Begin: Train Generator ************** #')
            self.generator.train()
            self.discriminator.linear.eval()
            self.gcl.gcn.eval()

            data_iter = progress(range(1, args.n_epoch_g + 1))
            for epoch_g in data_iter:
                loss_list, score_list = [], []
                for _ in range(args.iter_g):
                    loss, score = self.generator.loss(self.discriminator, args.bs_g)

                    optimizer_g.zero_grad()
                    loss.backward()
                    optimizer_g.step()

                    loss_list.append(loss.item())
                    score_list.append(score.item())

                if epoch_g % args.ri_g == 0:
                    data_iter.write(f'[Iter {epoch}] [Generator] [Epoch {epoch_g}] '
                                    f'loss = {np.nanmean(loss_list)}')
                
            data_iter.close()

            # color_print('# ============== End: Train Generator ============== #

        if outer_early_stopper.best_model:
            self.load_state_dict(outer_early_stopper.best_model)
        model = self.gcl.get_embeddings()

        if args.task == 'lp':
            print('=======================================================================')
            print(f'Evaluating GACN in link prediction task on {self.dataset}.')
            print('-----------------------------------------------------------------------')
            mrr, recall_50 = evaluate_lp(f'data/{self.dataset}', model)
            color_print(f'MRR: {mrr}')
            color_print(f'Recall_50: {recall_50}')
        elif args.task == 'clf':
            evaluate_clf(self.dataset, model)

    def generate_graph(self):
        with torch.no_grad():
            self.generator.eval()
            g_matrix = self.generator.forward().view(self.generator.n_g_nodes, self.generator.n_g_nodes)
            g_matrix[g_matrix >= 0.5] = 1
            g_matrix[g_matrix != 1] = 0
            g_matrix = g_matrix.detach().cpu().tolist()
            self.generator.train()
        neighbors = {}
        for node, neighbor_prob in zip(self.nodes, g_matrix):
            neighbors[node] = []
            for i, v in enumerate(neighbor_prob):
                if v == 1:
                    neighbors[node].append(self.nodes[i])
        return neighbors
