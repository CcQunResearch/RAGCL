# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 16:30
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : augmentation.py
# @Software: PyCharm
# @Note    :
import torch
from torch_scatter import scatter
import networkx as nx
from torch_geometric.utils import degree, to_undirected, to_networkx
import torch_geometric.utils as tg_utils
from torch_geometric.data.batch import Batch


def dege_drop_weights(data, aggr='mean', norm=True):
    centrality = data.centrality
    w_row = centrality[data.edge_index[0]].to(torch.float32)
    w_col = centrality[data.edge_index[1]].to(torch.float32)
    s_row = torch.log(w_row) if norm else w_row
    s_col = torch.log(w_col) if norm else w_col
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    weights = (s.max() - s) / (s.max() - s.mean())
    return weights


def drop_edge_weighted(data, edge_weights, p, threshold):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return data.edge_index[:, sel_mask]


def node_aug_weights(centrality, norm=True):
    s = torch.log(centrality) if norm else centrality
    weights = (s.max() - s) / (s.max() - s.mean())
    return weights


def aug_node_weighted(node_weights, p, threshold):
    node_weights = node_weights / node_weights.mean() * p
    node_weights = node_weights.where(node_weights < threshold, torch.ones_like(node_weights) * threshold)
    sel_mask = torch.bernoulli(1. - node_weights).to(torch.bool)
    return sel_mask


def drop_edge(batch_data, aggr, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        if aug_data_list[i].num_nodes > 1:
            edge_weights = dege_drop_weights(aug_data_list[i], aggr=aggr)
            aug_edge_index = drop_edge_weighted(aug_data_list[i], edge_weights, p, threshold)
            aug_data_list[i].edge_index = aug_edge_index
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def drop_node(batch_data, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        node_weights = node_aug_weights(aug_data_list[i].centrality)
        sel_mask = aug_node_weighted(node_weights, p, threshold)
        sel_mask[0] = True
        aug_edge_index, _ = tg_utils.subgraph(sel_mask, aug_data_list[i].edge_index, relabel_nodes=True,
                                              num_nodes=aug_data_list[i].num_nodes)
        aug_data_list[i].x = aug_data_list[i].x[sel_mask]
        aug_data_list[i].edge_index = aug_edge_index
        aug_data_list[i].__num_nodes__ = aug_data_list[i].x.shape[0]
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def mask_attr(batch_data, p, threshold):
    aug_data = batch_data.clone()
    aug_data_list = aug_data.to_data_list()
    for i in range(aug_data.num_graphs):
        node_weights = node_aug_weights(aug_data_list[i].centrality)
        sel_mask = aug_node_weighted(node_weights, p, threshold)
        sel_mask[0] = True
        # mask_token = aug_data_list[i].x.mean(dim=0)
        mask_token = torch.zeros_like(aug_data_list[i].x[0], dtype=torch.float)
        aug_data_list[i].x[sel_mask] = mask_token
    return Batch.from_data_list(aug_data_list).to(aug_data.x.device)


def augment(batch_data, augs):
    first_aug = augs[0]
    first_argu = first_aug.split(',')
    if first_argu[0] == 'DropEdge':
        aug_data = drop_edge(batch_data, first_argu[1], float(first_argu[2]), float(first_argu[3]))
    elif first_argu[0] == 'NodeDrop':
        aug_data = drop_node(batch_data, float(first_argu[1]), float(first_argu[2]))
    elif first_argu[0] == 'AttrMask':
        aug_data = mask_attr(batch_data, float(first_argu[1]), float(first_argu[2]))
    if len(augs) > 1:
        last_aug = augs[1]
        last_argu = last_aug.split(',')
        if last_argu[0] == 'NodeDrop':
            aug_data = drop_node(batch_data, float(last_argu[1]), float(last_argu[2]))
        elif last_argu[0] == 'AttrMask':
            aug_data = mask_attr(batch_data, float(last_argu[1]), float(last_argu[2]))

    return aug_data
