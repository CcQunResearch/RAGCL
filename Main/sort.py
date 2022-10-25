# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:40
# @Author  :
# @Email   :
# @File    : sort.py
# @Software: PyCharm
# @Note    :
import os
import json
import random
import time
from Main.utils import write_post, dataset_makedirs


def sort_dataset(label_source_path, label_dataset_path, k_shot=10000):
    train_path, val_path, test_path = dataset_makedirs(label_dataset_path)

    label_file_paths = []
    for filename in os.listdir(label_source_path):
        label_file_paths.append(os.path.join(label_source_path, filename))

    all_post = []
    for filepath in label_file_paths:
        post = json.load(open(filepath, 'r', encoding='utf-8'))
        all_post.append((post['source']['tweet id'], post))

    random.seed(1234)
    random.shuffle(all_post)
    train_post = []
    positive_num = 0
    negative_num = 0
    for post in all_post[:int(len(all_post) * 0.6)]:
        if post[1]['source']['label'] == 1 and positive_num != k_shot:
            train_post.append(post)
            positive_num += 1
        if post[1]['source']['label'] == 0 and negative_num != k_shot:
            train_post.append(post)
            negative_num += 1
        if positive_num == k_shot and negative_num == k_shot:
            break
    val_post = all_post[int(len(all_post) * 0.6):int(len(all_post) * 0.8)]
    test_post = all_post[int(len(all_post) * 0.8):]
    write_post(train_post, train_path)
    write_post(val_post, val_path)
    write_post(test_post, test_path)
