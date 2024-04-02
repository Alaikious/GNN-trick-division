import argparse
import csv
import re

import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features  # One-Hot torch float tensor
        self.edge_mat = 0
        self.max_neighbor = 0


def trans(str):  # 转换字符串为浮点数
    try:
        # 尝试将清理后的字符串转换为浮点数
        return float(str)
    except ValueError:
        try:
            return float(int(str))
        except ValueError:
            return None


def load_data(dataset, n_g=300):  # n_g是输入节点数量

    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """

    print('loading data')
    g_list = []
    n = 81  # 共81个元素
    l = 81  # label位于第81个元素

    with open('dataset/%s.csv' % dataset, 'r') as f:
        for i in range(n_g):
            cf = f.readline()
            g = nx.Graph()
            node_tags = [trans(cf[len(cf) - 1])]
            node_features = []
            n_edges = 0
            for j in range(n - 1):
                g.add_node(j)
                attr = trans(cf[j])
                node_features.append(attr)
                for k in range(0, n - 1):
                    g.add_edge(j, k)
            node_features = np.stack(node_features)
            node_feature_flag = True
            g_list.append(S2VGraph(g, l, node_tags, node_features))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = g.node_tags

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


parser = argparse.ArgumentParser(
    description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="translate_list",
                    help='name of dataset (default: MUTAG)')
parser.add_argument('--degree_as_tag', action="store_true",
                    help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
args = parser.parse_args()
graphs = load_data(args.dataset)

print(graphs.pop())
