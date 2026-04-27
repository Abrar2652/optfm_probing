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
from dataset import load_dataset_bipartite, split_bipartite_graph, BipartiteNodeData
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, eval_recall, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval_mip import generate_embedding
from parse import parser_add_main_args, parse_method_mip, parse_method_global

from generate_graph_scip import bipartite_graph_generation_scip

import time
import pickle
import json

import random
import math

import warnings
from gbdt_regressor import GradientBoostingRegressor_
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Testing Pipeline for Edge Identification')
parser_add_main_args(parser)
args = parser.parse_args()

data_dir = "/ml_nfs/1_item_placement/train"
instance_list = os.listdir(data_dir)

bipartite_graph_generation = bipartite_graph_generation_scip
var_d = 9
con_d = 1

print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# Label to the device
# for label in label_loader:
#     label = label.to(device)
#     c = label.values().max().item() + 1

# Label count, 二分类问题
c = 2

### Load method ###
model = parse_method_mip(args, c, var_d, con_d, device)

in_channels = 32
out_channels = 32
model_global = parse_method_global(args, in_channels, out_channels, device)

### Load model
model_save_dir = "/home/SGFormer-main/large/Models_SCIP/"

test_epoch = 0
model_path = model_save_dir + "model_params_epoch_" + str(test_epoch) + ".pth"
model.load_state_dict(torch.load(model_path))

### Load model GLOBAL
model_save_dir = "/home/SGFormer-main/graph_learning/Models_SCIP/"

test_epoch = 40
model_path = model_save_dir + "model_params_epoch_" + str(test_epoch) + ".pth"
model_global.load_state_dict(torch.load(model_path))

Maximum_batch_size_test = 1000

dataset = "1_item_placement"
config_data_dir = "/home/SGFormer-main/tune/MILPTune/dataset"
output_config_dir = "/ml_nfs/configs"

input_json_path = config_data_dir + "/" + dataset + ".json"
output_json_path = output_config_dir + "/" + dataset + ".json"

with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

path_to_index = {item['path']: index for index, item in enumerate(data)}

del_index_list = []
for instance in instance_list:
    if "mps" not in instance:
        continue
    
    print(instance)
    
    indexed_path = "~/ml4co-competition/instances/1_item_placement/train/" + instance
        
    # Extract the features
    var_feas, cons_feas, edge_indices, edge_features = bipartite_graph_generation(data_dir, instance)
    
    graph = BipartiteNodeData(
            torch.FloatTensor(cons_feas),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(var_feas)
        )
    
    embeddings = generate_embedding(model, graph, args, device, Maximum_batch_size_test)
    
    # append the batch dimension
    embeddings = embeddings.unsqueeze(0)
    
    lengths = torch.tensor([embeddings.shape[1]], device=device)
    
    graph_embeddings = model_global(embeddings, lengths)
        
    if indexed_path in path_to_index:
        index = path_to_index[indexed_path]
        if "configs" in data[index]:
            data[index]['embedding'] = graph_embeddings.squeeze(0).tolist()
        else:
            del_index_list.append(index)
            print(index)

del_set = set(del_index_list)
del_indices = sorted([i for i in del_set if 0 <= i < len(data)], reverse=True)

for idx in del_indices:
    del data[idx]

# 将更新后的数据写入新的 JSON 文件
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)