# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : main(sup).py.py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import numpy as np
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from Main.pargs import pargs
from Main.dataset import TreeDataset
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_dataset
from Main.model import ResGCN_graphcl, BiGCN_graphcl
from Main.utils import create_log_dict_sup, write_log, write_json
from Main.augmentation import augment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def sup_train(train_loader, aug1, aug2, model, optimizer, device, lamda, use_unsup_loss):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        out = model(data)
        sup_loss = F.nll_loss(out, data.y.long().view(-1))

        if use_unsup_loss:
            aug_data1 = augment(data, augs1)
            aug_data2 = augment(data, augs2)

            out1 = model.forward_graphcl(aug_data1)
            out2 = model.forward_graphcl(aug_data2)
            unsup_loss = model.loss_graphcl(out1, out2)

            loss = sup_loss + lamda * unsup_loss
        else:
            loss = sup_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(model, dataloader, num_classes, device):
    model.eval()
    error = 0

    y_true = []
    y_pred = []

    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        error += F.nll_loss(pred, data.y.long().view(-1)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.max(1).indices.tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = round(accuracy_score(y_true, y_pred), 4)
    precs = []
    recs = []
    f1s = []
    for label in range(num_classes):
        precs.append(round(precision_score(y_true == label, y_pred == label, labels=True), 4))
        recs.append(round(recall_score(y_true == label, y_pred == label, labels=True), 4))
        f1s.append(round(f1_score(y_true == label, y_pred == label, labels=True), 4))
    micro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='micro'), 4)

    macro_p = round(precision_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_r = round(recall_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, labels=range(num_classes), average='macro'), 4)
    return error / len(dataloader.dataset), acc, precs, recs, f1s, \
           [micro_p, micro_r, micro_f1], [macro_p, macro_r, macro_f1]


def test_and_log(model, val_loader, test_loader, num_classes, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_precs, val_recs, val_f1s, val_micro_metric, val_macro_metric = \
        test(model, val_loader, num_classes, device)
    test_error, test_acc, test_precs, test_recs, test_f1s, test_micro_metric, test_macro_metric = \
        test(model, test_loader, num_classes, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val ERROR: {:.7f}, Test ERROR: {:.7f}\n  Train ACC: {:.4f}, Validation ACC: {:.4f}, Test ACC: {:.4f}\n' \
                   .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc) \
               + f'  Test PREC: {test_precs}, Test REC: {test_recs}, Test F1: {test_f1s}\n' \
               + f'  Test Micro Metric(PREC, REC, F1):{test_micro_metric}, Test Macro Metric(PREC, REC, F1):{test_macro_metric}'

    log_record['val accs'].append(val_acc)
    log_record['test accs'].append(test_acc)
    log_record['test precs'].append(test_precs)
    log_record['test recs'].append(test_recs)
    log_record['test f1s'].append(test_f1s)
    log_record['test micro metric'].append(test_micro_metric)
    log_record['test macro metric'].append(test_macro_metric)
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

    word_embedding = 'tfidf' if 'tfidf' in dataset else 'word2vec'
    lang = 'ch' if 'Weibo' in dataset else 'en'
    tokenize_mode = args.tokenize_mode

    split = args.split
    batch_size = args.batch_size
    undirected = args.undirected
    centrality = args.centrality

    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs
    use_unsup_loss = args.use_unsup_loss

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', unsup_dataset, 'dataset')
    model_path = osp.join(dirname, '..', 'Model',
                          f'w2v_{dataset}_{tokenize_mode}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict_sup(args)

    if not osp.exists(model_path) and word_embedding == 'word2vec':
        sentences = collect_sentences(label_source_path, unlabel_dataset_path, unsup_train_size, lang, tokenize_mode)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test precs': [], 'test recs': [], 'test f1s': [],
                      'test micro metric': [], 'test macro metric': []}

        word2vec = Embedding(model_path, lang, tokenize_mode) if word_embedding == 'word2vec' else None

        sort_dataset(label_source_path, label_dataset_path, k_shot=k, split=split)

        train_dataset = TreeDataset(train_path, word_embedding, word2vec, centrality, undirected)
        val_dataset = TreeDataset(val_path, word_embedding, word2vec, centrality, undirected)
        test_dataset = TreeDataset(test_path, word_embedding, word2vec, centrality, undirected)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        num_classes = train_dataset.num_classes
        if args.model == 'ResGCN':
            model = ResGCN_graphcl(dataset=train_dataset, num_classes=num_classes, hidden=args.hidden,
                                   num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                                   residual=args.skip_connection,
                                   res_branch=args.res_branch, global_pool=args.global_pool, dropout=args.dropout,
                                   edge_norm=args.edge_norm).to(device)
        elif args.model == 'BiGCN':
            model = BiGCN_graphcl(train_dataset.num_features, args.hidden, args.hidden, num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader, num_classes,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            _ = sup_train(train_loader, args.aug1, args.aug2, model, optimizer, device, lamda, use_unsup_loss)

            train_error, train_acc, _, _, _, _, _ = test(model, train_loader, num_classes, device)
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader, num_classes, device, epoch,
                                                           lr, train_error, train_acc, log_record)
            write_log(log, log_info)

            if split == '622':
                scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-10:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
