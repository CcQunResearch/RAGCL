# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 20:26
# @Author  :
# @Email   :
# @File    : record.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))

cal_mean = -10

if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset:', log['dataset'])
            print('unsup dataset:', log['unsup dataset'])
            # print('unsup train size:', log['unsup train size'])
            # print('batch size:', log['batch size'])
            # print('unsup bs ratio:', log['unsup_bs_ratio'])
            # print('n layers feat:', log['n layers feat'])
            # print('n layers conv:', log['n layers conv'])
            # print('n layers fc:', log['n layers fc'])
            # print('vector size:', log['vector size'])
            # print('hidden size:', log['hidden'])
            # print('global pool:', log['global pool'])
            # print('skip connection:', log['skip connection'])
            # print('res branch:', log['res branch'])
            # print('dropout:', log['dropout'])
            # print('edge norm:', log['edge norm'])
            # print('lr:', log['lr'])
            # print('epochs:', log['epochs'])
            # print('weight decay:', log['weight decay'])
            # print('lamda:', log['lamda'])
            print('centrality:', log['centrality'])
            print('aug1:', log['aug1'])
            print('aug2:', log['aug2'])
            # print('k:', log['k'])

            acc_list = []
            for run in log['record']:
                # mean_acc = run['mean acc']
                mean_acc = round(np.mean(run['test accs'][cal_mean:]), 3)
                acc_list.append(mean_acc)

            mean = round(sum(acc_list) / len(acc_list), 3)
            sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
            maxx = max(acc_list)
            print('test acc: {:.3f}±{:.3f}'.format(mean, sd))
            print('max acc: {:.3f}'.format(maxx))
            print()
