# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 15:07
# @Author  :
# @Email   :
# @File    : main(semisup).py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from Main.pargs import pargs
from Main.dataset import WeiboDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.model import ResGCN_graphcl
from Main.utils import create_log_dict_semisup, write_log, write_json
from Main.augmentation import augment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def semisup_train(unsup_train_loader, train_loader, aug1, aug2, model, optimizer, device, lamda):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for sup_data, unsup_data in zip(train_loader, unsup_train_loader):
        optimizer.zero_grad()
        sup_data = sup_data.to(device)
        unsup_data = unsup_data.to(device)

        out = model(sup_data)
        sup_loss = F.binary_cross_entropy(out, sup_data.y.to(torch.float32))

        sup_aug_data1 = augment(sup_data, augs1)
        sup_aug_data2 = augment(sup_data, augs2)
        unsup_aug_data1 = augment(unsup_data, augs1)
        unsup_aug_data2 = augment(unsup_data, augs2)

        sup_out1 = model.forward_graphcl(sup_aug_data1)
        sup_out2 = model.forward_graphcl(sup_aug_data2)
        unsup_out1 = model.forward_graphcl(unsup_aug_data1)
        unsup_out2 = model.forward_graphcl(unsup_aug_data2)
        unsup_loss = model.loss_graphcl(sup_out1, sup_out2) + model.loss_graphcl(unsup_out1, unsup_out2)

        # print(sup_loss.item(), (lamda * unsup_loss).item())
        # print()

        loss = sup_loss + lamda * unsup_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sup_data.num_graphs

    # for sup_data, unsup_data in zip(train_loader, unsup_train_loader):
    #     optimizer.zero_grad()
    #     sup_data = sup_data.to(device)
    #     unsup_data = unsup_data.to(device)
    #
    #     out = model(sup_data)
    #     sup_loss = F.binary_cross_entropy(out, sup_data.y.to(torch.float32))
    #
    #     sup_data_list = sup_data.to_data_list()
    #     unsup_data_list = unsup_data.to_data_list()
    #     for item in sup_data_list:
    #         item.y = None
    #     data = Batch.from_data_list(sup_data_list + unsup_data_list).to(device)
    #
    #     norm = True
    #     if centrality_metric == "Degree":
    #         centrality = degree_centrality(data)
    #     elif centrality_metric == "PageRank":
    #         centrality = pagerank_centrality(data)
    #     elif centrality_metric == "Katz":
    #         centrality = katz_centrality(data)
    #     elif centrality_metric == "Betweenness":
    #         centrality = betweenness_centrality(data)
    #
    #     aug_data1 = augment(data, centrality, norm, augs1)
    #     aug_data2 = augment(data, centrality, norm, augs2)
    #
    #     out1 = model.forward_graphcl(aug_data1)
    #     out2 = model.forward_graphcl(aug_data2)
    #     unsup_loss = model.loss_graphcl(out1, out2)
    #
    #     loss = sup_loss + lamda * unsup_loss
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss.item() * sup_data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    batch_size = args.batch_size
    unsup_bs_ratio = args.unsup_bs_ratio
    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict_semisup(args)

    if not osp.exists(model_path):
        sort_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        word2vec = Embedding(model_path)
        unlabel_dataset = WeiboDataset(unlabel_dataset_path, word2vec, args.centrality)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size * unsup_bs_ratio, shuffle=True)

        sort_dataset(label_source_path, label_dataset_path, k_shot=k)

        train_dataset = WeiboDataset(train_path, word2vec, args.centrality)
        val_dataset = WeiboDataset(val_path, word2vec, args.centrality)
        test_dataset = WeiboDataset(test_path, word2vec, args.centrality)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = ResGCN_graphcl(dataset=unlabel_dataset, hidden=args.hidden, num_feat_layers=args.n_layers_feat,
                               num_conv_layers=args.n_layers_conv, num_fc_layers=args.n_layers_fc, gfn=False,
                               collapse=False, residual=args.skip_connection, res_branch=args.res_branch,
                               global_pool=args.global_pool, dropout=args.dropout, edge_norm=args.edge_norm).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            _ = semisup_train(unsup_train_loader, train_loader, args.aug1, args.aug2, model, optimizer, device, lamda)

            train_error, train_acc, _, _, _ = test(model, train_loader, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader, device,
                                                           epoch, lr, train_error, train_acc, log_record)
            write_log(log, log_info)

            scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
