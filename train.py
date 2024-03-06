import argparse
import os
import time

import torch

from dataloader import (get_sparse_graph, read_edges, read_edges_from_file,
                        read_features)
from model import GACN
from utils import color_print, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for STMNE')
    parser.add_argument('--dataset', type=str, help='the dataset', default='uci')
    parser.add_argument('--task', type=str, default='lp')
    parser.add_argument('--gpu_id', type=int, help='gpu id', default=0)

    parser.add_argument('--wd_g', type=float, help='weight decay for generator', default=0.0001)
    parser.add_argument('--wd_d', type=float, help='weight decay for discriminator', default=0.0001)
    parser.add_argument('--wd_cl', type=float, help='weight decay for contrastive learning', default=0)

    parser.add_argument('--lr_g', type=float, help='learning rate for generator', default=0.001)
    parser.add_argument('--lr_d', type=float, help='learning rate for discriminator', default=0.001)
    parser.add_argument('--lr_cl', type=float, help='learning rate for contrastive learning', default=0.001)

    parser.add_argument('--temperature', type=float, default=0.0001)
    parser.add_argument('--ssl_ratio', type=float, default=0.5)
    parser.add_argument('--st', type=float, help='score threshold', default=0.2)
    parser.add_argument('--n_g_nodes', type=int, default=2000)
    parser.add_argument('--n_valid', type=int, default=800)
    parser.add_argument('--n_dim', type=int, default=64)
    parser.add_argument('--init_rate', type=float, default=0.75)

    parser.add_argument('--n_epoch', type=int, help='number of epoch', default=50)
    parser.add_argument('--n_epoch_g', type=int, help='number of epoch for generator', default=100)
    parser.add_argument('--n_epoch_d', type=int, help='number of epoch for discriminator', default=100)
    parser.add_argument('--n_epoch_cl', type=int, help='number of epoch for contrastive learning', default=200)

    parser.add_argument('--bs_g', type=int, help='batch size for generator', default=1)
    parser.add_argument('--bs_d', type=int, help='batch size for discriminator', default=3)
    parser.add_argument('--bs_cl', type=int, help='batch size for discriminator', default=2048)
    parser.add_argument('--iter_g', type=int, help='iter for generator', default=1)
    parser.add_argument('--iter_d', type=int, help='iter for discriminator', default=1)

    parser.add_argument('--seed', type=int, help='the random seed', default=0)
    parser.add_argument('--lambda_new', type=float, default=0.5)
    parser.add_argument('--lambda_count', type=float, default=1)

    parser.add_argument('--cl_temperature', type=float, default=0.5)
    parser.add_argument('--cl_lambda_ssl', type=float, default=0.0001)
    parser.add_argument('--cl_lambda_bpr', type=float, default=1)
    parser.add_argument('--cl_lambda_reg', type=float, default=0.0001)

    parser.add_argument('--patience', type=int, help='early stopping patience', default=3)
    parser.add_argument('--patience_cl', type=int, help='early stopping patience for contrastive learning', default=2)

    parser.add_argument('--ri_g', type=int, help='report interval for generator', default=20)
    parser.add_argument('--ri_d', type=int, help='report interval for discriminator', default=20)
    parser.add_argument('--ri_cl', type=int, help='report interval for contrastive learning', default=20)

    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None and torch.cuda.is_available() else 'cpu')

    cache_path = os.path.join('cache_cl_gan', args.dataset)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    data_root = './data'
    n_nodes, all_neighbors, nodes = read_edges(os.path.join(data_root, args.dataset, 'train.txt'))
    node_features, n_dim = read_features(os.path.join(data_root, args.dataset, 'node_features.txt'))
    if node_features is not None:
        args.n_dim = n_dim
    valid_edges = read_edges_from_file(os.path.join(data_root, args.dataset, 'valid.txt'))
    graph = get_sparse_graph(n_nodes, all_neighbors, cache_path).to(device)
    
    model = GACN(n_nodes, nodes, all_neighbors, graph, device, args.dataset, args, node_features)

    model.do_train(args, valid_edges)
    color_print('Training Complete!')
    color_print('Cost: %.3fs' % (time.time() - start_time))


if __name__ == '__main__':
    main()
