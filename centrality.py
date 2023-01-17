# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 16:30
# @Author  :
# @Email   :
# @File    : centrality.py
# @Software: PyCharm
# @Note    :
from torch_scatter import scatter
import networkx as nx
from torch_geometric.utils import degree, to_undirected, to_networkx
import os
import json
import torch
from torch_geometric.data import Data, Batch
from utils import write_json


def get_root_index(data):
    root_index = scatter(torch.ones((data.num_nodes,)).to(data.x.device).to(torch.long), data.batch, reduce='sum')
    for i in range(root_index.shape[0] - 1, -1, -1):
        root_index[i] = 0
        for j in range(i):
            root_index[i] += root_index[j]
    return root_index


# need normalization
# root centrality == no children 1 level reply centrality
def degree_centrality(data):
    ud_edge_index = to_undirected(data.edge_index)
    # out degree
    centrality = degree(ud_edge_index[1])
    centrality[0] = 1
    centrality = centrality - 1.0 + 1e-8
    return centrality


# need normalization
# root centrality = no children 1 level reply centrality
def pagerank_centrality(data, damp=0.85, k=10):
    device = data.x.device
    bu_edge_index = data.edge_index.clone()
    bu_edge_index[0], bu_edge_index[1] = data.edge_index[1], data.edge_index[0]

    num_nodes = data.num_nodes
    deg_out = degree(bu_edge_index[0])
    centrality = torch.ones((num_nodes,)).to(device).to(torch.float32)

    for i in range(k):
        edge_msg = centrality[bu_edge_index[0]] / deg_out[bu_edge_index[0]]
        agg_msg = scatter(edge_msg, bu_edge_index[1], reduce='sum')
        pad = torch.zeros((len(centrality) - len(agg_msg),)).to(device).to(torch.float32)
        agg_msg = torch.cat((agg_msg, pad), 0)

        centrality = (1 - damp) * centrality + damp * agg_msg

    centrality[0] = centrality.min().item()
    return centrality


# need normalization
# root centrality == no children 1 level reply centrality
def eigenvector_centrality(data):
    bu_data = data.clone()
    bu_data.edge_index = bu_data.no_root_edge_index
    # bu_data.edge_index[0], bu_data.edge_index[1] = data.no_root_edge_index[1], data.no_root_edge_index[0]

    bu_data.edge_index = to_undirected(bu_data.edge_index)

    graph = to_networkx(bu_data)
    centrality = nx.eigenvector_centrality(graph, tol=1e-3)
    centrality = [centrality[i] for i in range(bu_data.num_nodes)]
    centrality = torch.tensor(centrality, dtype=torch.float32).to(bu_data.x.device)
    return centrality


# need normalization
# root centrality == no children 1 level reply centrality
def betweenness_centrality(data):
    bu_data = data.clone()
    # bu_data.edge_index[0], bu_data.edge_index[1] = data.edge_index[1], data.edge_index[0]

    graph = to_networkx(bu_data)
    centrality = nx.betweenness_centrality(graph)
    centrality = [centrality[i] if centrality[i] != 0 else centrality[i] + 1e-16 for i in range(bu_data.num_nodes)]
    centrality = torch.tensor(centrality, dtype=torch.float32).to(bu_data.x.device)
    return centrality


def calculate_centrality(source_path):
    raw_file_names = os.listdir(source_path)
    for filename in raw_file_names:
        filepath = os.path.join(source_path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))

        x = torch.ones(len(post['comment']) + 1, 20)
        row = []
        col = []
        no_root_row = []
        no_root_col = []
        filepath = os.path.join(source_path, filename)
        post = json.load(open(filepath, 'r', encoding='utf-8'))

        for i, comment in enumerate(post['comment']):
            if comment['parent'] != -1:
                no_root_row.append(comment['parent'] + 1)
                no_root_col.append(comment['comment id'] + 1)
            row.append(comment['parent'] + 1)
            col.append(comment['comment id'] + 1)
        edge_index = [row, col]
        no_root_edge_index = [no_root_row, no_root_col]
        edge_index = torch.LongTensor(edge_index)
        no_root_edge_index = torch.LongTensor(no_root_edge_index)
        one_data = Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index)

        if one_data.num_nodes > 1:
            dc = degree_centrality(Batch.from_data_list([one_data])).tolist()
            pc = pagerank_centrality(Batch.from_data_list([one_data])).tolist()
            ec = eigenvector_centrality(Batch.from_data_list([one_data])).tolist()
            bc = betweenness_centrality(Batch.from_data_list([one_data])).tolist()
        else:
            dc = pc = ec = bc = [1]

        post['centrality'] = {}
        post['centrality']['Degree'] = dc
        post['centrality']['Pagerank'] = pc
        post['centrality']['Eigenvector'] = ec
        post['centrality']['Betweenness'] = bc

        write_json(post, filepath)
