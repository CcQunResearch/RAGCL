# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 17:49
# @Author  :
# @Email   :
# @File    : record_pretrain.py
# @Software: PyCharm
# @Note    :
import os
import sys
import json
import math

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '..'))


def get_value(acc_list):
    mean = round(sum(acc_list) / len(acc_list), 3)
    sd = round(math.sqrt(sum([(x - mean) ** 2 for x in acc_list]) / len(acc_list)), 3)
    maxx = max(acc_list)
    return 'test acc: {:.3f}±{:.3f}'.format(mean, sd), 'max acc: {:.3f}'.format(maxx)


if __name__ == '__main__':
    log_dir_path = os.path.join(dirname, '..', 'Log')

    for filename in os.listdir(log_dir_path):
        if filename[-4:] == 'json':
            print(f'【{filename[:-5]}】')
            filepath = os.path.join(log_dir_path, filename)

            log = json.load(open(filepath, 'r', encoding='utf-8'))
            print('dataset:', log['dataset'])
            print('unsup dataset:', log['unsup dataset'])
            print('unsup train size:', log['unsup train size'])
            print('batch size:', log['batch size'])
            print('n layers feat:', log['n layers feat'])
            print('n layers conv:', log['n layers conv'])
            print('n layers fc:', log['n layers fc'])
            print('vector size:', log['vector size'])
            print('hidden size:', log['hidden'])
            print('global pool:', log['global pool'])
            print('skip connection:', log['skip connection'])
            print('res branch:', log['res branch'])
            print('dropout:', log['dropout'])
            print('edge norm:', log['edge norm'])
            print('lr:', log['lr'])
            print('ft lr:', log['ft_lr'])
            print('epochs:', log['epochs'])
            print('ft epochs:', log['ft_epochs'])
            print('weight decay:', log['weight decay'])
            print('centrality:', log['centrality'])
            print('aug1:', log['aug1'])
            print('aug2:', log['aug2'])

            record = log['record']
            acc_lists = {10: [], 20: [], 40: [], 80: [], 100: [], 200: [], 300: [], 500: [], 10000: []}

            for run_record in record:
                for re in run_record['record']:
                    acc_lists[re['k']].append(re['mean acc'])
            for key in acc_lists.keys():
                acc, max_acc = get_value(acc_lists[key])
                print(f'k: {key}, {acc}, {max_acc}')
            print()
