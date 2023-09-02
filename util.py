# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os.path as osp

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120

def get_ntu120_action_classes():
    with open('data/ntu/ntu120_action_classes.txt', 'r') as file:
        lines = file.readlines()
    ntu120_action_classes=[]
    k=0
    for line in lines:
        if k<10:
            action = line[4:-2]
        elif k<100:
            action = line[5:-2]
        else:
            action = line[6:-2]
        ntu120_action_classes.append(action)
        k=k+1
    print(ntu120_action_classes)
    print(len(ntu120_action_classes))
    return ntu120_action_classes
    