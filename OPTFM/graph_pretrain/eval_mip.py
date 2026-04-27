import torch
import torch.nn.functional as F

from torch_geometric.utils import subgraph
from pathlib import Path

from dataset import split_bipartite_graph
import numpy as np
import math
import pickle

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, label, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.constraint_features,
                dataset.edge_index,
                dataset.edge_attr,
                dataset.variable_features)
    if len(label.shape) == 1:
        label_ = label.unsqueeze(1)
    train_acc = eval_func(
        label_[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        label_[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        label_[split_idx['test']], out[split_idx['test']])

    out = F.log_softmax(out, dim=1)
    valid_loss = criterion(
        out[split_idx['valid']], label[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_mip(model, valid_loader, eval_func, criterion, args, device, Maximum_batch_size, in_test = False):
    
    model.eval()
    batch_id = 0
    valid_loss_list = []
    valid_acc_list = []
    for dataset in valid_loader:
        if in_test:
            print("Batch ID: ", batch_id)
        dataset.constraint_features, dataset.edge_index, dataset.edge_attr, dataset.variable_features = \
            dataset.constraint_features.to(device), dataset.edge_index.to(device), dataset.edge_attr.to(device), dataset.variable_features.to(device)
        
        node_count = dataset.constraint_features.shape[0] + dataset.variable_features.shape[0]
        
        if node_count > 3000:
            continue
        
        if node_count > Maximum_batch_size:
            subgraphs = split_bipartite_graph(dataset.variable_features, dataset.constraint_features, dataset.edge_index, dataset.edge_attr, Maximum_batch_size, device)
            for mini_batch_id in range(len(subgraphs)):
                constraint_indices, variable_indices, batch_edge_indices = subgraphs[mini_batch_id]
                
                mini_batch_constraint_features = dataset.constraint_features[constraint_indices]
                mini_batch_variable_features = dataset.variable_features[variable_indices]
                
                # Append global variable and constraint features
                mini_batch_constraint_features = torch.cat((mini_batch_constraint_features, torch.mean(mini_batch_constraint_features, dim=0, keepdim=True)), dim=0)
                mini_batch_variable_features = torch.cat((mini_batch_variable_features, torch.mean(mini_batch_variable_features, dim=0, keepdim=True)), dim=0)
                M = mini_batch_constraint_features.shape[0] - 1
                N = mini_batch_variable_features.shape[0] - 1
                
                mini_batch_edge_index = dataset.edge_index[:, batch_edge_indices]
                mini_batch_edge_features = dataset.edge_attr[batch_edge_indices]
                
                con_node_idx = torch.zeros(dataset.constraint_features.shape[0], dtype=torch.long,
                                device=device)
                var_node_idx = torch.zeros(dataset.variable_features.shape[0], dtype=torch.long,
                                device=device)
                con_node_idx[constraint_indices] = torch.arange(len(constraint_indices), device=device)
                var_node_idx[variable_indices] = torch.arange(len(variable_indices), device=device)
                
                mini_batch_edge_index[0] = con_node_idx[mini_batch_edge_index[0]]
                mini_batch_edge_index[1] = var_node_idx[mini_batch_edge_index[1]]
                
                # Append edge indices and edge features
                left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
                right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
                mini_batch_edge_index = torch.cat([mini_batch_edge_index, left_to_right_edges, right_to_left_edges], dim=1)
                
                mini_batch_edge_features = torch.cat([mini_batch_edge_features, torch.ones((M + N, 1), dtype=mini_batch_edge_features.dtype, device=device)], dim=0)
                
                size = (len(mini_batch_constraint_features), len(mini_batch_variable_features))
                value = torch.tensor([1 for i in range(len(mini_batch_edge_index[0]))], device=device)
                mini_batch_label = torch.sparse_coo_tensor(mini_batch_edge_index, value, size)
                mini_batch_label = mini_batch_label.coalesce()
                
                mini_batch_label = mini_batch_label.to(device)
                
                # Random mask some edges
                total_edges = len(mini_batch_edge_features)
                num_edges_to_select = math.ceil(total_edges * (1 - args.masked_edge_ratio))
                random_indices = torch.randperm(total_edges)[:num_edges_to_select]
                
                selected_edge_index = mini_batch_edge_index[:, random_indices]
                selected_edge_features = mini_batch_edge_features[random_indices]
                
                out, _, _ = model(mini_batch_constraint_features,
                            selected_edge_index,
                            selected_edge_features,
                            mini_batch_variable_features)
                
                mini_batch_label = mini_batch_label.to_dense().flatten()
                if len(mini_batch_label.shape) == 1:
                    label_ = mini_batch_label.unsqueeze(1)
                
                valid_acc = eval_func(label_, out)
                print(valid_acc)
                valid_acc_list.append(valid_acc)

                # out = F.log_softmax(out, dim=1)
                valid_loss = criterion(out, mini_batch_label)
                valid_loss_list.append(valid_loss)
        
        else:
            
            # Append global variable and constraint features
            constraint_features = torch.cat((dataset.constraint_features, torch.mean(dataset.constraint_features, dim=0, keepdim=True)), dim=0)
            variable_features = torch.cat((dataset.variable_features, torch.mean(dataset.variable_features, dim=0, keepdim=True)), dim=0)
            M = constraint_features.shape[0] - 1
            N = variable_features.shape[0] - 1
            
            # Append edge indices and edge features
            left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
            right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
            edge_indices = torch.cat([dataset.edge_index, left_to_right_edges, right_to_left_edges], dim=1)
            
            edge_features = torch.cat([dataset.edge_attr, torch.ones((M + N, 1), dtype=dataset.edge_attr.dtype, device=device)], dim=0)
            
            # Random mask some edges
            total_edges = len(edge_features)
            num_edges_to_select = math.ceil(total_edges * (1 - args.masked_edge_ratio))
            random_indices = torch.randperm(total_edges)[:num_edges_to_select]
            
            selected_edge_index = edge_indices[:, random_indices]
            selected_edge_features = edge_features[random_indices]
            
            out, _, _ = model(constraint_features,
                    selected_edge_index,
                    selected_edge_features,
                    variable_features)
            
            label_size = (len(constraint_features), len(variable_features))
            label_value = torch.tensor([1 for i in range(len(edge_indices[0]))], device=device)
            batch_label = torch.sparse_coo_tensor(edge_indices, label_value, label_size)
            batch_label = batch_label.coalesce()
            label = batch_label.to(device)
            
            label = label.to_dense().flatten()
            if len(label.shape) == 1:
                label_ = label.unsqueeze(1)
            valid_acc = eval_func(label_, out)
            print(valid_acc)
            valid_acc_list.append(valid_acc)
            
            # out = F.log_softmax(out, dim=1)
            valid_loss = criterion(out, label)
            valid_loss_list.append(valid_loss)
        
        batch_id += 1
        
    average_acc = np.mean(valid_acc_list)
    average_loss = torch.mean(torch.stack(valid_loss_list), dim=0)

    if not in_test:
        return average_acc, average_loss
    return valid_acc_list, valid_loss_list, average_acc, average_loss


@torch.no_grad()
def generate_graphembedding(model, valid_loader, device, Maximum_batch_size, samples_per_instance, output_dir, in_test = False):
    
    model.eval()
    batch_id = 0
    for num in range(samples_per_instance):
        print("Epoch ID:  ", num)
        for dataset in valid_loader:
            data = []
            if in_test:
                print("Batch ID: ", batch_id)
            
            node_count = dataset.constraint_features.shape[0] + dataset.variable_features.shape[0]
            if node_count > 9000:
                continue
            
            dataset.constraint_features, dataset.edge_index, dataset.edge_attr, dataset.variable_features = \
                dataset.constraint_features.to(device), dataset.edge_index.to(device), dataset.edge_attr.to(device), dataset.variable_features.to(device)
                                    
            label = num
            results = {}
            
            if node_count > Maximum_batch_size:
                subgraphs = split_bipartite_graph(dataset.variable_features, dataset.constraint_features, dataset.edge_index, dataset.edge_attr, Maximum_batch_size, device)
                for mini_batch_id in range(len(subgraphs)):
                    constraint_indices, variable_indices, batch_edge_indices = subgraphs[mini_batch_id]
                    
                    mini_batch_constraint_features = dataset.constraint_features[constraint_indices]
                    mini_batch_variable_features = dataset.variable_features[variable_indices]
                    
                    # Append global variable and constraint features
                    mini_batch_constraint_features = torch.cat((mini_batch_constraint_features, torch.mean(mini_batch_constraint_features, dim=0, keepdim=True)), dim=0)
                    mini_batch_variable_features = torch.cat((mini_batch_variable_features, torch.mean(mini_batch_variable_features, dim=0, keepdim=True)), dim=0)
                    M = mini_batch_constraint_features.shape[0] - 1
                    N = mini_batch_variable_features.shape[0] - 1
                    
                    mini_batch_edge_index = dataset.edge_index[:, batch_edge_indices]
                    mini_batch_edge_features = dataset.edge_attr[batch_edge_indices]
                    
                    con_node_idx = torch.zeros(dataset.constraint_features.shape[0], dtype=torch.long,
                                    device=device)
                    var_node_idx = torch.zeros(dataset.variable_features.shape[0], dtype=torch.long,
                                    device=device)
                    con_node_idx[constraint_indices] = torch.arange(len(constraint_indices), device=device)
                    var_node_idx[variable_indices] = torch.arange(len(variable_indices), device=device)
                    
                    mini_batch_edge_index[0] = con_node_idx[mini_batch_edge_index[0]]
                    mini_batch_edge_index[1] = var_node_idx[mini_batch_edge_index[1]]
                    
                    # Append edge indices and edge features
                    left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
                    right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
                    mini_batch_edge_index = torch.cat([mini_batch_edge_index, left_to_right_edges, right_to_left_edges], dim=1)
                    
                    mini_batch_edge_features = torch.cat([mini_batch_edge_features, torch.ones((M + N, 1), dtype=mini_batch_edge_features.dtype, device=device)], dim=0)
                    
                    out, var_emb, cons_emb = model(mini_batch_constraint_features,
                                mini_batch_edge_index,
                                mini_batch_edge_features,
                                mini_batch_variable_features)
                                        
                    global_var_emb = var_emb[-1, :]  # 形状为 (num_features,)
                    global_cons_emb = cons_emb[-1, :]  # 形状为 (num_features,)
                    
                    # 拼接为一维张量
                    concatenated_tensor = torch.cat([global_var_emb, global_cons_emb], dim=0)  # 形状为 (2 * num_features,)
                    
                    # 将张量转换为 Python 列表并添加到列表中
                    data.append(concatenated_tensor.tolist())
            else:
                # Append global variable and constraint features
                constraint_features = torch.cat((dataset.constraint_features, torch.mean(dataset.constraint_features, dim=0, keepdim=True)), dim=0)
                variable_features = torch.cat((dataset.variable_features, torch.mean(dataset.variable_features, dim=0, keepdim=True)), dim=0)
                M = constraint_features.shape[0] - 1
                N = variable_features.shape[0] - 1
                
                # Append edge indices and edge features
                left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
                right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
                edge_indices = torch.cat([dataset.edge_index, left_to_right_edges, right_to_left_edges], dim=1)
                
                edge_features = torch.cat([dataset.edge_attr, torch.ones((M + N, 1), dtype=dataset.edge_attr.dtype, device=device)], dim=0)
                
                out, var_emb, cons_emb = model(constraint_features,
                        edge_indices,
                        edge_features,
                        variable_features)
                                
                global_var_emb = var_emb[-1, :]  # 形状为 (num_features,)
                global_cons_emb = cons_emb[-1, :]  # 形状为 (num_features,)
                
                # 拼接为一维张量
                concatenated_tensor = torch.cat([global_var_emb, global_cons_emb], dim=0)  # 形状为 (2 * num_features,)
                
                # 将张量转换为 Python 列表并添加到列表中
                data.append(concatenated_tensor.tolist())
            
            results['data'] = np.array(data)
            results['label'] = label
            
            saved_path = output_dir + "/" + "sample_" + str(batch_id) + "_" + str(num) + ".pkl"
            
            with open(saved_path, 'wb') as file:
                pickle.dump(results, file)
            
            batch_id += 1


@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(torch.device(device))
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

def evaluate_batch(model, dataset, split_idx, args, device, n, true_label):
    num_batch = n // args.batch_size + 1
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    train_total, train_correct=0, 0
    valid_total, valid_correct=0, 0
    test_total, test_correct=0, 0

    with torch.no_grad():
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            train_mask_i = train_mask[idx_i]
            valid_mask_i = valid_mask[idx_i]
            test_mask_i = test_mask[idx_i]

            out_i = model(x_i, edge_index_i)

            cur_train_total, cur_train_correct=eval_acc(y_i[train_mask_i], out_i[train_mask_i])
            train_total+=cur_train_total
            train_correct+=cur_train_correct
            cur_valid_total, cur_valid_correct=eval_acc(y_i[valid_mask_i], out_i[valid_mask_i])
            valid_total+=cur_valid_total
            valid_correct+=cur_valid_correct
            cur_test_total, cur_test_correct=eval_acc(y_i[test_mask_i], out_i[test_mask_i])
            test_total+=cur_test_total
            test_correct+=cur_test_correct

            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            # valid_acc = eval_func(
            #     dataset.label[split_idx['valid']], out[split_idx['valid']])
            # test_acc = eval_func(
            #     dataset.label[split_idx['test']], out[split_idx['test']])
        train_acc=train_correct/train_total
        valid_acc=valid_correct/valid_total
        test_acc=test_correct/test_total

    return train_acc, valid_acc, test_acc, 0, None

def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred=torch.max(pred,dim=1,keepdim=True)[1]
    # cmp=torch.eq(true, pred)
    # print(f'pred:{pred}')
    # print(cmp)
    true_cnt=(true==pred).sum()

    return true.shape[0], true_cnt.item()


if __name__=='__main__':
    x=torch.arange(4).unsqueeze(1)
    y=torch.Tensor([[3,0,0,0],
                    [3,2,1.5,2.8],
                    [0,0,2,1],
                    [0,0,1,3]
                    ])
    a, b=eval_acc(x, y)
    print(x)
    print(a,b)
