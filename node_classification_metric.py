import os

import numpy as np
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split


def get_data(dataset):
    node_labels = []
    with open(os.path.join(f'data/{dataset}/node_categories.txt')) as fin:
        for line in fin:
            if line.strip():
                items = line.strip().split()
                node, label = int(items[0]), items[1]
                node_labels.append([node, label])
    return {node: label for node, label in node_labels}


def evaluate_clf(dataset, model):
    node_labels = get_data(dataset)
    nodes = list(node_labels)
    scores = []
    data_iter = tqdm.tqdm(range(5), bar_format='{l_bar}{r_bar}')
    for i in data_iter:
        train_nodes, test_nodes = train_test_split(nodes, train_size=0.5, random_state=i)
        
        clf = LogisticRegression(C=1, solver='liblinear')
        train_embeddings = [model[x] for x in train_nodes]
        train_labels = [node_labels[x] for x in train_nodes]
        test_embeddings = [model[x] for x in test_nodes]
        test_labels = [node_labels[x] for x in test_nodes]
        clf.fit(train_embeddings, train_labels)
        pred_labels = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, pred_labels)
        f1 = f1_score(test_labels, pred_labels, average='macro')
        p = precision_score(test_labels, pred_labels, zero_division=0, average='macro')
        r = recall_score(test_labels, pred_labels, zero_division=0, average='macro')

        data_iter.write(f'[Iter {i}] f1 = {f1}, p = {p}, r = {r}')
        scores.append([acc, f1, p, r])
    print('============================================================')
    print('Training rate:', 0.5)
    acc, macros, p, r = zip(*scores)
    print(f'Precision & Recall & F1 = '
          f'{np.mean(p):.4f} ± {np.std(p):.4f} & '
          f'{np.mean(r):.4f} ± {np.std(r):.4f} & '
          f'{np.mean(macros):.4f} ± {np.std(macros):.4f}')
    print('============================================================\n')
