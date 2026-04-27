import argparse
import sys
import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset_bipartite, split_bipartite_graph
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, eval_recall, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval_mip import evaluate_mip
from parse import parse_method, parser_add_main_args, parse_method_mip

import time
import pickle

import random
import math

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Testing Pipeline for Edge Identification')
parser_add_main_args(parser)
args = parser.parse_args()

args.data_dir = args.data_dir.split("/")[0] + "_SCIP/"
args.valid_dir = args.valid_dir.split("/")[0] + "_SCIP/"
args.test_dir = args.test_dir.split("/")[0] + "_SCIP/"

if args.all_data:
    args.data_dir = "/ml_nfs/" + args.data_dir
    args.valid_dir = "/ml_nfs/" + args.valid_dir
    args.test_dir = "/ml_nfs/" + args.test_dir

print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_loader = load_dataset_bipartite(args.test_dir, 1)

# Label to the device
# for label in label_loader:
#     label = label.to(device)
#     c = label.values().max().item() + 1

# Extract some basic information
for train_data in dataset_loader:
    var_d = train_data.variable_features.shape[1]
    con_d = train_data.constraint_features.shape[1]
    break

# Label count, 二分类问题
c = 2

### Load method ###
model = parse_method_mip(args, c, var_d, con_d, device)

### Load model
model_save_dir = "Models_SCIP/"
test_epoch = 20
model_path = model_save_dir + "model_params_epoch_" + str(test_epoch) + ".pth"
model.load_state_dict(torch.load(model_path))

### Performance metric (Acc, AUC, F1) ###
# criterion = nn.NLLLoss()  # 负对数似然损失（Negative Log Likelihood Loss）
criterion = nn.CrossEntropyLoss()  # 负对数似然损失（Negative Log Likelihood Loss）

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
elif args.metric == 'acc':
    eval_func = eval_acc
else:
    eval_func = eval_recall

Maximum_batch_size_test = 1000

test_acc_list, test_loss_list, test_acc, test_loss = evaluate_mip(model, dataset_loader, eval_func, criterion, args, device, Maximum_batch_size_test, in_test=True)

# 使用matplotlib绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(test_acc_list, bins=30, density=True, alpha=0.6, color='b', label='Histogram')

# 使用seaborn绘制核密度估计(KDE)图
sns.kdeplot(test_acc_list, color='r', label='KDE')

# 添加标题和标签
plt.title('Probability Distribution of the Data')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

# 保存图形
plt.savefig('probability_distribution.png')

print(test_acc)