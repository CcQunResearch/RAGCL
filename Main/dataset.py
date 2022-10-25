# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 18:59
# @Author  :
# @Email   :
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset


class WeiboDataset(InMemoryDataset):
    def __init__(self, root, word2vec, centrality_metric, transform=None, pre_transform=None, pre_filter=None):
        self.word2vec = word2vec
        self.centrality_metric = centrality_metric
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        raw_file_names = self.raw_file_names

        for filename in raw_file_names:
            centrality = None
            y = []
            row = []
            col = []
            no_root_row = []
            no_root_col = []

            filepath = os.path.join(self.raw_dir, filename)
            post = json.load(open(filepath, 'r', encoding='utf-8'))
            x = self.word2vec.get_sentence_embedding(post['source']['content']).view(1, -1)
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            for i, comment in enumerate(post['comment']):
                x = torch.cat(
                    [x, self.word2vec.get_sentence_embedding(comment['content']).view(1, -1)], 0)
                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            if self.centrality_metric == "Degree":
                centrality = torch.tensor(post['centrality']['Degree'], dtype=torch.float32)
            elif self.centrality_metric == "PageRank":
                centrality = torch.tensor(post['centrality']['Pagerank'], dtype=torch.float32)
            elif self.centrality_metric == "Eigenvector":
                centrality = torch.tensor(post['centrality']['Eigenvector'], dtype=torch.float32)
            elif self.centrality_metric == "Betweenness":
                centrality = torch.tensor(post['centrality']['Betweenness'], dtype=torch.float32)
            edge_index = [row, col]
            no_root_edge_index = [no_root_row, no_root_col]
            y = torch.LongTensor(y)
            edge_index = torch.LongTensor(edge_index)
            no_root_edge_index = torch.LongTensor(no_root_edge_index)
            one_data = Data(x=x, y=y, edge_index=edge_index, no_root_edge_index=no_root_edge_index,
                            centrality=centrality) if 'label' in post['source'].keys() else \
                Data(x=x, edge_index=edge_index, no_root_edge_index=no_root_edge_index, centrality=centrality)
            data_list.append(one_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])
