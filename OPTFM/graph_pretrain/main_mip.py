import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import CustomDataset, collate_fn
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_recall, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval_mip import evaluate_mip
from parse import parser_add_main_args, parse_method_mip
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning import losses

import time
import pickle

import random
import math

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Edge Identification')
parser_add_main_args(parser)
args = parser.parse_args()

print(args)

fix_seed(args.seed)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
data_dir = '/ml_nfs/samples_1'
dataset = CustomDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1280, shuffle=True, collate_fn=collate_fn)

### Load method ###
in_channels = 32
out_channels = 32
model = parse_method_mip(args, in_channels, out_channels, device)

# criterion = nn.NLLLoss()  # 负对数似然损失（Negative Log Likelihood Loss
loss_func = losses.ProxyAnchorLoss(num_classes=317, embedding_size=out_channels)

model.train()
print('MODEL:', model)

best_acc = -float("inf")
model_save_dir = "Models_SCIP/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

optimizer = torch.optim.Adam([
            {'params': model.params0, 'weight_decay': args.trans_weight_decay},
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)

for epoch in range(args.epochs):
    print("Current epoch: ", epoch)
    running_loss = 0.0
    for batch_idx, (data, labels, lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        start_time = time.time()
        
        labels = torch.tensor(labels, device=device)
        data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)
        embeddings = model(data, lengths)
        loss = loss_func(embeddings, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 1 == 0:  # 每 10 个 batch 打印一次
            print(f'Epoch [{epoch + 1}/{args.epochs}], Step [{batch_idx + 1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    if epoch % args.eval_step == 0:
        save_path = model_save_dir + "model_params_epoch_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), save_path)