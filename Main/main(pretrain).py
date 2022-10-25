# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:56
# @Author  :
# @Email   :
# @File    : main(pretrain).py
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
from Main.pargs import pargs
from Main.dataset import WeiboDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.model import ResGCN_graphcl
from Main.utils import create_log_dict_pretrain, write_log, write_json
from Main.augmentation import augment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def pre_train(dataloader, aug1, aug2, model, optimizer, device):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)

        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)

        out1 = model.forward_graphcl(aug_data1)
        out2 = model.forward_graphcl(aug_data2)
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataloader.dataset)


def fine_tuning(model, optimizer, dataloader, device):
    model.train()

    total_loss = 0
    y_true = []
    y_pred = []
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)

        loss = F.binary_cross_entropy(out, data.y.to(torch.float32))
        loss.backward()

        out[out >= 0.5] = 1
        out[out < 0.5] = 0
        y_true += data.y.tolist()
        y_pred += out.tolist()

        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataloader.dataset), accuracy_score(y_true, y_pred)


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


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, ft_log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    ft_log_record['val accs'].append(round(val_acc, 3))
    ft_log_record['test accs'].append(round(test_acc, 3))
    ft_log_record['test prec T'].append(round(test_prec[0], 3))
    ft_log_record['test prec F'].append(round(test_prec[1], 3))
    ft_log_record['test rec T'].append(round(test_rec[0], 3))
    ft_log_record['test rec F'].append(round(test_rec[1], 3))
    ft_log_record['test f1 T'].append(round(test_f1[0], 3))
    ft_log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, ft_log_record


def joao(dataloader, model, gamma_joao):
    aug_prob = dataloader.dataset.aug_prob
    # calculate augmentation loss
    loss_aug = np.zeros(25)

    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        dataloader.dataset.set_aug_prob(_aug_prob)

        n_aug1, n_aug2 = n // 5, n % 5

        count, count_stop = 0, len(dataloader.dataset) // (
                dataloader.batch_size * 10) + 1  # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for _, data1, data2 in dataloader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                out1 = model.forward_graphcl(data1, n_aug1)
                out2 = model.forward_graphcl(data2, n_aug2)
                loss = model.loss_graphcl(out1, out2)
                loss_aug[n] += loss.item() * data1.num_graphs
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= (count * dataloader.batch_size)

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()

    return aug_prob


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    unsup_dataset = args.unsup_dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    ft_runs = args.ft_runs

    aug_ratio = args.aug_ratio
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    ft_epochs = args.ft_epochs
    gamma_joao = args.gamma_joao

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
    weight_path = osp.join(dirname, '..', 'Model', f'{log_name}.pt')

    log = open(log_path, 'w')
    log_dict = create_log_dict_pretrain(args)

    if not osp.exists(model_path):
        sort_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    word2vec = Embedding(model_path)

    for run in range(runs):
        unlabel_dataset = WeiboDataset(unlabel_dataset_path, word2vec, args.centrality)
        unsup_train_loader = DataLoader(unlabel_dataset, batch_size, shuffle=True)

        model = ResGCN_graphcl(dataset=unlabel_dataset, hidden=args.hidden, num_feat_layers=args.n_layers_feat,
                               num_conv_layers=args.n_layers_conv, num_fc_layers=args.n_layers_fc, gfn=False,
                               collapse=False, residual=args.skip_connection, res_branch=args.res_branch,
                               global_pool=args.global_pool, dropout=args.dropout, edge_norm=args.edge_norm).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

        write_log(log, f'runs:{run}')
        log_record = {
            'run': run,
            'record': []
        }

        for epoch in range(1, epochs + 1):
            pretrain_loss = pre_train(unsup_train_loader, args.aug1, args.aug2, model, optimizer, device)

            log_info = 'Epoch: {:03d}, Loss: {:.7f}'.format(epoch, pretrain_loss)
            write_log(log, log_info)

        torch.save(model.state_dict(), weight_path)
        write_log(log, '')

        ks = [10, 20, 40, 80, 100, 200, 300, 500, 10000]
        for k in ks:
            for r in range(ft_runs):
                ft_lr = args.ft_lr
                write_log(log, f'k:{k}, r:{r}')

                ft_log_record = {'k': k, 'r': r, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                                 'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

                sort_dataset(label_source_path, label_dataset_path, k_shot=k)

                train_dataset = WeiboDataset(train_path, word2vec, args.centrality)
                val_dataset = WeiboDataset(val_path, word2vec, args.centrality)
                test_dataset = WeiboDataset(test_path, word2vec, args.centrality)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model.load_state_dict(torch.load(weight_path))
                optimizer = Adam(model.parameters(), lr=args.ft_lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

                val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader,
                                                                  device, 0, args.ft_lr, 0, 0, ft_log_record)
                write_log(log, log_info)

                for epoch in range(1, ft_epochs + 1):
                    ft_lr = scheduler.optimizer.param_groups[0]['lr']
                    train_loss, train_acc = fine_tuning(model, optimizer, train_loader, device)

                    val_error, log_info, ft_log_record = test_and_log(model, val_loader, test_loader,
                                                                      device, epoch, ft_lr, train_loss, train_acc,
                                                                      ft_log_record)
                    write_log(log, log_info)

                    scheduler.step(val_error)

                ft_log_record['mean acc'] = round(np.mean(ft_log_record['test accs'][-10:]), 3)
                log_record['record'].append(ft_log_record)
                write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
