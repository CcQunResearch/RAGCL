# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import random
import copy


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_norm (bool, optional): whether or not to normalize adj matrix.
            (default: :obj:`True`)
        gfn (bool, optional): If `True`, only linear transform (1x1 conv) is
            applied to every nodes. (default: :obj:`False`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        edge_index = edge_index[0]
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, dataset=None, num_classes=2, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0,
                 edge_norm=True):
        super(ResGCN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.num_classes = num_classes
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch
        self.collapse = collapse
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        self.use_xg = False
        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)

        hidden_in = dataset.num_features
        if collapse:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, self.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            feat_gfn = True  # set true so GCNConv is feat transform
            self.conv_feat = GCNConv(hidden_in, hidden, gfn=feat_gfn)
            if "gating" in global_pool:
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            if self.res_branch == "resnet":
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GCNConv(hidden, hidden, gfn=feat_gfn))
            else:
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(GConv(hidden, hidden))
            self.bn_hidden = BatchNorm1d(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, self.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.collapse:
            return self.forward_collapse(x, edge_index, batch, xg)
        elif self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(x, edge_index, batch, xg)
        elif self.res_branch == "BNReLUConv":
            return self.forward_BNReLUConv(x, edge_index, batch, xg)
        elif self.res_branch == "ConvReLUBN":
            return self.forward_ConvReLUBN(x, edge_index, batch, xg)
        elif self.res_branch == "resnet":
            return self.forward_resnet(x, edge_index, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_collapse(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNConvReLU(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_BNReLUConv(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_, edge_index)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_ConvReLUBN(self, x, edge_index, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x, edge_index))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, edge_index, batch, xg=None):
        # this mimics resnet architecture in cv.
        x = self.bn_feat(x)
        x = self.conv_feat(x, edge_index)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i * 3 + 0](x_))
            x_ = self.convs[i * 3 + 0](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 1](x_))
            x_ = self.convs[i * 3 + 1](x_, edge_index)
            x_ = F.relu(self.bns_conv[i * 3 + 2](x_))
            x_ = self.convs[i * 3 + 2](x_, edge_index)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class ResGCN_graphcl(ResGCN):
    def __init__(self, **kargs):
        super(ResGCN_graphcl, self).__init__(**kargs)
        hidden = kargs['hidden']
        self.proj_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

    def forward_graphcl(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss


######################################################

class vgae_encoder(ResGCN):
    def __init__(self, **kargs):
        super(vgae_encoder, self).__init__(**kargs)
        hidden = kargs['hidden']
        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                         nn.Softplus())

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_

        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(torch.nn.Module):
    def __init__(self, hidden=128):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')
        self.pool = global_mean_pool
        self.add_pool = global_add_pool

    def forward(self, x, x_mean, x_std, batch, edge_index, edge_index_batch, edge_index_neg, edge_index_neg_batch,
                reward):
        edge_pos_pred = self.sigmoid(self.decoder(x[edge_index[0]] * x[edge_index[1]]))
        edge_neg_pred = self.sigmoid(self.decoder(x[edge_index_neg[0]] * x[edge_index_neg[1]]))

        # for link prediction
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score
        edge_pred = torch.cat((edge_pos_pred, edge_neg_pred)).detach().cpu().numpy()
        edge_auroc = roc_auc_score(np.concatenate((np.ones(edge_pos_pred.shape[0]), np.zeros(edge_neg_pred.shape[0]))),
                                   edge_pred)
        edge_auprc = average_precision_score(
            np.concatenate((np.ones(edge_pos_pred.shape[0]), np.zeros(edge_neg_pred.shape[0]))), edge_pred)
        if True:
            return edge_auroc, edge_auprc
        # end link prediction

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device))
        loss_pos = self.pool(loss_edge_pos, edge_index_batch)
        loss_neg = self.pool(loss_edge_neg, edge_index_neg_batch)
        loss_rec = loss_pos + loss_neg
        if not reward is None:
            loss_rec = loss_rec * reward

        # reference: https://github.com/DaehanKim/vgae_pytorch
        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)
        kl_ones = torch.ones(kl_divergence.shape).to(kl_divergence.device)
        kl_divergence = self.pool(kl_divergence, batch)
        kl_double_norm = 1 / self.add_pool(kl_ones, batch)
        kl_divergence = kl_divergence * kl_double_norm

        loss = (loss_rec + kl_divergence).mean()
        return loss


class vgae(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, reward=None):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, data.batch, data.edge_index, data.edge_index_batch, data.edge_index_neg,
                            data.edge_index_neg_batch, reward)
        return loss

    # for one graph
    def generate(self, data):
        x, _, _ = self.encoder(data)
        prob = torch.einsum('nd,md->nmd', x, x)
        prob = self.decoder.decoder(prob).squeeze()

        prob = torch.exp(prob)
        prob[torch.isinf(prob)] = 1e10
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))

        # sparsify
        prob[prob < 1e-1] = 0
        prob[prob.sum(dim=1) == 0] = 1
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))
        return prob


######################################################


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, tddroprate):
        super(TDrumorGCN, self).__init__()
        self.tddroprate = tddroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        edge_index_list = edge_index.tolist()
        if self.tddroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            tdrow = list(np.array(edge_index_list[0])[poslist])
            tdcol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([tdrow, tdcol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, budroprate):
        super(BUrumorGCN, self).__init__()
        self.budroprate = budroprate
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        device = data.x.device
        x = data.x
        edge_index = data.edge_index.clone()
        edge_index[0], edge_index[1] = data.edge_index[1], data.edge_index[0]

        edge_index_list = edge_index.tolist()
        if self.budroprate > 0:
            length = len(edge_index_list[0])
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            burow = list(np.array(edge_index_list[0])[poslist])
            bucol = list(np.array(edge_index_list[1])[poslist])
            edge_index = torch.LongTensor([burow, bucol]).to(device)

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[index][0]
        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[index][0]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_classes, tddroprate=0.0, budroprate=0.0):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats, tddroprate)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats, budroprate)
        self.proj_head = torch.nn.Linear((out_feats + hid_feats) * 2, out_feats)
        self.fc = torch.nn.Linear((out_feats + hid_feats) * 2, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def forward_graphcl(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.proj_head(x)
        return x

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss
