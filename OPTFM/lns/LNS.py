import os
from pathlib import Path
import time
import torch
import random
import pickle
import argparse

import numpy as np

from gbdt_regressor import GradientBoostingRegressor_

import math
import heapq

import pandas as pd

from pyscipopt import Model

def GBDT_predict(data,
        gbdt_model_path,
        #  gbdt,
         device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         ):
 
    # gbdt model can not share among different instances, it must re-install for each solving
    with open(gbdt_model_path, "rb") as f:
        trained_model = pickle.load(f)[0]
    gbdt = GradientBoostingRegressor_(model=trained_model, random_state=123, n_estimators=30, learning_rate=0.1, max_depth=5, min_samples_split=2)
    predict = gbdt.predict(np.array(data))
    loss =  gbdt.calc(np.array(data))

    # # Get the variable type list
    # var_type = np.array(var_feas)[:, 3]
    loss_array = np.array(loss)

    # For continous variables, set the loss to infinite, avoid fixing
    loss = loss_array.tolist()
    return predict, loss, data, gbdt

class pair: 
    def __init__(self): 
        self.site = 0
        self.loss = 0

def cmp(a, b):
    if a.loss < b.loss: 
        return -1
    else:
        return 1

def cmp2(a, b):
    if a.loss > b.loss: 
        return -1
    else:
        return 1

def get_best_solution(test_instance_path, max_time, set_time, nowX, block_list, loss, predict, variable_ratio):
    '''
    Function Description:
    Solve the problem using an optimization solver based on the provided problem instance.

    Parameters:
    - test_instance_path: Instance path
    - max_time: Left solving time
    - set_time: Maximum solving time
    - nowX: Current best solution
    - block_list: Current partition neighborhood
    - loss: Prediction loss for each variable
    - predict: Prediction value for each variable
    - rate: Maximum variable ratios in sub problem

    Return: 
    The optimal solution of the current neighborhood.
    '''

    block_array = np.array(block_list)
    loss_array = np.array(loss)[block_array]
    inf_mask = np.isinf(loss_array)
    loss_array = np.where(inf_mask, -1, loss_array)
    nowX_array = np.array(nowX)[block_array]
    predict_array = np.array(predict)[block_array]

    variable_scores = list(loss_array * abs(nowX_array - predict_array))
    sorted_indices = find_max_n_indices(variable_scores, len(variable_scores))

    unfixed_count = int(len(predict) * variable_ratio)
    unfixed_variable_set = block_array[np.array(sorted_indices)[:unfixed_count]]

    model = Model()
    model.readProblem(test_instance_path)
    
    vars = model.getVars()
    
    for i in range(len(predict)):
        # assert(col < num_variables)
        if i in unfixed_variable_set:
            continue
        model.fixVar(vars[i], nowX[i])
    
    model.setParam('limits/time', 0.2 * set_time)
    
    model.optimize()
    
    # 输出每个变量的取值
    current_solution = [model.getVal(var) for var in vars]
    
    return current_solution, model.getObjVal()


def find_min_n_indices(lst, n):
    if n < 1:
        return []
    min_indices = heapq.nsmallest(n, range(len(lst)), key=lst.__getitem__)
    return min_indices

def find_max_n_indices(lst, n):
    if n < 1:
        return []
    max_indices = heapq.nlargest(n, range(len(lst)), key=lst.__getitem__)
    return max_indices

def repair(F, U, var_feas, con_feas, graph_feas, predict):

    sorted_tuples = sorted(graph_feas, key=lambda x: (x[1], -abs(x[2])))  # 按照每个三元组的行约束排序，并优先考虑影响较大的entry
    current_row = 0
    lower_activity = 0
    upper_activity = 0
    start_id = 0
    end_id = 0
    for i in range(len(sorted_tuples)):
        row = sorted_tuples[i][1]
        if (row != current_row):
            end_id = i
            row_upper = con_feas[current_row][1]
            row_lower = con_feas[current_row][0]
            update_fixed = False
            if lower_activity > row_upper or upper_activity < row_lower:
                lower_gap = lower_activity - row_upper
                upper_gap = row_lower - upper_activity
                gap = max(lower_gap, upper_gap)
                if gap > 1e-6:
                    update_fixed = True
            
            if update_fixed:
                for j in range(start_id, end_id):
                    col = sorted_tuples[j][0]
                    if col in F:
                        F.remove(col)
                        U.add(col)
                        col_lower = var_feas[col][0]
                        col_upper = var_feas[col][1]
                        col_coef = sorted_tuples[j][2]
                        fix_val = predict[col]
                        if col_coef > 0:
                            loweract_down = col_coef * (fix_val - col_lower)
                            upperact_up = col_coef * (col_upper - fix_val)
                        else:
                            loweract_down = -col_coef * (col_upper - fix_val)
                            upperact_up = -col_coef * (fix_val - col_lower)
                        lower_gap -= loweract_down
                        upper_gap -= upperact_up
                        gap = max(lower_gap, upper_gap)
                        if gap < 1e-6:
                            update_fixed = False
                            break
            
            current_row = row
            lower_activity = 0
            upper_activity = 0
            start_id = i
            end_id = i
        
        col = sorted_tuples[i][0]
        col_lower = var_feas[col][0]
        col_upper = var_feas[col][1]
        col_coef = sorted_tuples[i][2]
        fix_val = predict[col]
        if col in F:
            lower_activity += col_coef * fix_val
            upper_activity += col_coef * fix_val
        else:
            if col_coef > 0:
                lower_activity += col_coef * col_lower
                upper_activity += col_coef * col_upper
            else:
                lower_activity += col_coef * col_upper
                upper_activity += col_coef * col_lower
    
    return F, U


def initial_solution_search(test_instance_path, set_time, predict, loss, variable_ratio, col_bounds, row_bounds, graph_feas):
    '''
    Function Description:
    Perform an initial solution search using fixed-radius neighborhood search based on the given problem instance and the predicted results from GBDT.

    Parameters:
    - test_instance_path: Instance that has to be solved
    - time_limit: Maximum solving time.
    - predict: Predicted results from GBDT.
    - loss: Prediction loss from GBDT.
    - variable_ratio: Maximum free variable size
    - var_feas, con_feas, graph_feas: utilized to iterate each constraint

    Return: 
    The initial feasible solution of the problem and its corresponding objective function value.
    '''

    min_n_indices = find_min_n_indices(loss, len(loss))

    F = set()  # Fixed variable set
    U = set([i for i in range(len(loss))])  # Free variable set

    num_variables = len(loss)

    before_fixed_size = 0
    initial_variable_ratio = variable_ratio
    Maximum_F_length = 0
    successive_fails = 0
    while len(U) > initial_variable_ratio * num_variables:
        print("F:"+str(len(F)))
        if len(F) > 0 or before_fixed_size > 0:
            variable_ratio = variable_ratio * 0.9
        fixed_size = math.ceil((1 - variable_ratio) * num_variables)
        for i in range(fixed_size):
            if min_n_indices[i] not in F and (not math.isinf(loss[min_n_indices[i]])):
                F.add(min_n_indices[i])
                U.remove(min_n_indices[i])
        before_fixed_size = fixed_size
        
        F, U = repair(F, U, col_bounds, row_bounds, graph_feas, predict)
        
        # 超过10次无法固定更多变量，跳出，避免无限循环
        if len(F) > Maximum_F_length:
            Maximum_F_length = len(F)
            successive_fails = 0
        else:
            successive_fails += 1
        
        if successive_fails >= 10:
            print("Warning: Cannot fix so many variables in the initial search")
            break
    
    model = Model()
    model.readProblem(test_instance_path)
    
    vars = model.getVars()
    
    objective_sense = model.getObjectiveSense()
    if objective_sense == 'maximize':
        obj_type = -1
    elif objective_sense == 'minimize':
        obj_type = 1
    
    for col in F:
        # assert(col < num_variables)
        fix_val = predict[col]
        model.fixVar(vars[col], fix_val)
    
    model.setParam('limits/time', 0.4 * set_time)
    
    model.optimize()
    
    # 输出每个变量的取值
    current_solution = [model.getVal(var) for var in vars]
    
    return current_solution, model.getObjVal(), obj_type


def cross_generate_blocks(GBDT, data, required_partition_count):
    '''
    Function Description:
    Obtain the neighborhood partitioning result based on the given problem instance, the predicted results from GBDT, and the current solution.

    Parameters:
    - loss: Prediction loss from GBDT.
    - rate: Neighborhood radius.
    - predict: Predicted results from GBDT.
    - nowX: Current solution of the problem instance.
    - GBDT: Trained Gradient Boosting Decision Tree.
    - data: Neural encoding results of the decision variables.
    - required_partition_count: Neighborhood partition count

    Return: A set of partitioning results of the neighborhood.
    '''

    # the overall tree counts
    num_estimators = GBDT.model.n_estimators

    # TODO: Random select a tree for partition the neighborhood
    randomtree_index = random.randint(0, num_estimators - 1)

    partition_list = GBDT.get_partition_result(randomtree_index, required_partition_count, data)

    return partition_list, len(partition_list)

def cross(test_instance_path, variable_ratio, turnX1, block_list1, turnX2, block_list2, max_time, set_time, col_bounds, row_bounds, graph_feas, loss):
    '''
    Function Description:
    Obtain the crossover solution of two neighborhoods based on the given problem instance, the neighborhood information and search results of neighborhood A, the neighborhood information and search results of neighborhood B.

    Parameters:
    - test_instance_path: The instance that need to be solved
    - variable_ratio: The maximum variable ratios in sub problem
    - turnX1, block_list1: The neighborhood variable list in partition 1 and the optimized solutions
    - turnX2, block_list2: The neighborhood variable list in partition 2 and the optimized solutions
    - max_time: The left solving time
    - set_time: The maximum solving time
    - var_feas, cons_feas, graph_feas: utilized to iterate each constraint rows
    - inactive_constraints: constraints list to disable

    Return: 
    The crossover solution of the two neighborhoods and their corresponding objective function values.
    '''
    num_variables = len(turnX1)
    F = set()  # Fixed variable set
    U = set([i for i in range(num_variables)])  # Free variable set
    fix_val_list = [-1 for i in range(num_variables)] # The value for each potentially fixed variable, default -1

    for i in range(num_variables):
        if i in block_list1 and (not math.isinf(loss[i])):
            F.add(i)
            U.remove(i)
            fix_val_list[i] = turnX1[i]
        elif i in block_list2 and (not math.isinf(loss[i])):
            F.add(i)
            U.remove(i)
            fix_val_list[i] = turnX2[i]
    
    F, U = repair(F, U, col_bounds, row_bounds, graph_feas, fix_val_list)
    
    model = Model()
    model.readProblem(test_instance_path)
    
    vars = model.getVars()
    
    if len(U) < variable_ratio * num_variables:
        for var_idx in F:
            # assert(col < num_variables)
            model.fixVar(vars[var_idx], fix_val_list[var_idx])
        
        model.setParam('limits/time', 0.2 * set_time)
        
        model.optimize()
        
        # 输出每个变量的取值
        current_solution = [model.getVal(var) for var in vars]
        
        return current_solution, model.getObjVal()
    
    else:
        return None, -1


def get_variable_features(var_feas):
    variable_features = []
    for i in range(len(var_feas)):
        now_variable_features = []
        now_variable_features.append(var_feas[i][0]) # obj coef
        if not math.isinf(var_feas[i][1]):
            now_variable_features.append(var_feas[i][1]) # lower bound
        else:
            now_variable_features.append(-10000) # lower bound, infinite to +/-10000
        if not math.isinf(var_feas[i][2]):
            now_variable_features.append(var_feas[i][2]) # upper bound
        else:
            now_variable_features.append(10000) # upper bound, infinite to +/-10000
        now_variable_features.append(var_feas[i][3]) # variable type
        now_variable_features.append(random.random()) # random features
        variable_features.append(now_variable_features)
    return variable_features

def get_constraint_features(cons_feas, graph_feas_array:np.array):
    constraint_features = []
    append_constraint_features = []
    append_mapping_dict = {}
    append_graph_filter_list = []
    for i in range(len(cons_feas)):
        need_append_edges = False
        if cons_feas[i][0] == cons_feas[i][1]:
            now_constraint_features = []
            now_constraint_features.append(cons_feas[i][0])
            now_constraint_features.append(3) # Constraints type, where 1 represents <=, 2 represents >=, and 3 represents =.
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)
            continue

        if not math.isinf(cons_feas[i][0]):
            need_append_edges = True
            now_constraint_features = []
            now_constraint_features.append(cons_feas[i][0])
            now_constraint_features.append(2) # Constraints type, where 1 represents <=, 2 represents >=, and 3 represents =.
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)
        
        if not math.isinf(cons_feas[i][1]):
            now_constraint_features = []
            now_constraint_features.append(cons_feas[i][1])
            now_constraint_features.append(1) # Constraints type, where 1 represents <=, 2 represents >=, and 3 represents =.
            now_constraint_features.append(random.random())
            if not need_append_edges:
                constraint_features.append(now_constraint_features)
            else:
                append_constraint_features.append(now_constraint_features)
            if need_append_edges:
                # graph_feas_array = np.array(feature_data.getBigraph())
                graph_filter = graph_feas_array[np.where(graph_feas_array[:,1] == i)]
                new_index = int(len(cons_feas) + len(append_constraint_features) - 1)
                append_mapping_dict[i] = new_index
                graph_filter[:,1] = new_index
                graph_filter_list_tmp = [list(row) for row in graph_filter]
                for item in range(len(graph_filter_list_tmp)):
                    append_graph_filter_list.append((int(graph_filter_list_tmp[item][0]), int(graph_filter_list_tmp[item][1]), graph_filter_list_tmp[item][2]))
    return constraint_features, append_constraint_features, append_mapping_dict, append_graph_filter_list

def reduction_constraint(solution, cons_feas, graph_feas, activate_num, append_mapping_dict, full_graph_feas):
    """_summary_
        pick out a sub problem to reduce the constraint number for original problem
    Args:
        solution (list[int or float]): solution has been found
        cons_feas (list[list]): _description_
        graph_feas (list[list]): graph without appending edge mapping
        activate_num (int): constraint number expect to be activate
        append_mapping_dict (dict[int,int]): appending edge mapping for constraint with two finite bounds
        full_graph_feas (list[list]): graph with appending edge mapping

    Returns:
        _type_: _description_
        reduction_edge_indices, reduction_edge_features are use for predict for new sub problem
    """
    active_constraints, inactive_constraints = constraint_reduction_filter(solution, cons_feas, graph_feas, activate_num) # not include append_row_index
    append_active_constraints = []
    for row in active_constraints:
        append_row_index = append_mapping_dict.get(row)
        if append_row_index is not None:
            append_active_constraints.append(append_row_index)
    active_constraints = active_constraints + append_active_constraints
    reduction_graph_feas = []
    reduction_edge_indices= [[],[]]
    reduction_edge_features= []
    for i in range(len(full_graph_feas)):
        if full_graph_feas[i][1] in active_constraints:
            reduction_graph_feas.append(full_graph_feas[i])
            # get reduction_edge_indices
            col = graph_feas[i][0]
            row = graph_feas[i][1]
            reduction_edge_indices[0].append(row)
            reduction_edge_indices[1].append(col)
            reduction_edge_features.append([graph_feas[i][2]])
    return inactive_constraints, reduction_graph_feas, reduction_edge_indices, reduction_edge_features



def optimize(
    test_instance_path,
    graph_feas, col_bounds, row_bounds,
    embeddings,
    set_time,
    rate,
    gbdt_model_path,
    compare_with_base,
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    embeddings = np.array(embeddings.cpu())

    required_partition_count = 4

    # model installing time is not included
    begin_time = time.time()
    if(os.path.exists(gbdt_model_path) == False):
        print("No GBDT trained model!")
        return

    model_load_time = time.time() - begin_time
    print("model loading time: " + str(model_load_time))

    # get the GBDT prediction results
    predict, loss, gnn_encoding_data, gbdt = GBDT_predict(embeddings, gbdt_model_path, device)

    # Initial solution search.
    ansTime = []
    ansVal = []

    # Tightmove repair, Currently disabled
    # solution, success = tight_repair(predict, loss, var_feas, cons_feas, graph_feas)

    # for i in range(len(predict)):
    #     predict[i] = solution[i]

    # solution value list and objective value
    nowX, nowVal, obj_type = initial_solution_search(test_instance_path, set_time, predict, loss, rate, col_bounds, row_bounds, graph_feas)
    ansTime.append(time.time() - begin_time)
    ansVal.append(nowVal)
    
    while(time.time() - begin_time < set_time):
        turnX = []
        turnVal = []
        
        # neighborhood cluster
        block_list, block_counts = cross_generate_blocks(gbdt, gnn_encoding_data, required_partition_count)

        # GBDT-guided neighborhood partitioning and neighborhood search.
        for i in range(block_counts):
            max_time = set_time - (time.time() - begin_time)
            if(max_time <= 0):
                break
            newX, newVal = get_best_solution(test_instance_path, max_time, set_time, nowX, block_list[i], loss, predict, rate)
            turnX.append(newX)
            turnVal.append(newVal)
        
        # First-level crossover between neighborhoods.
        if(len(turnX) == block_counts):
            i = 0
            while (i + 1 < block_counts):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = cross(test_instance_path, rate, turnX[i], block_list[i], turnX[i + 1], block_list[i + 1], max_time, set_time, col_bounds, row_bounds, graph_feas, loss)
                if newX:
                    turnX.append(newX)
                    turnVal.append(newVal)
                i += 2
        
        # Second-level crossover between neighborhoods.
        if(len(turnX) == int(block_counts + block_counts/2)):
            max_time = set_time - (time.time() - begin_time)
            if(max_time <= 0):
                break

            i = 0
            while i + 1 < block_counts:
                block_list.append(block_list[i] + block_list[i + 1])
                i += 2
            
            i = block_counts
            while i + 1 < len(block_list):
                newX, newVal = cross(test_instance_path, rate, turnX[i], block_list[i], turnX[i + 1], block_list[i + 1], max_time, set_time, col_bounds, row_bounds, graph_feas, loss)
                if(turnVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
                i += 2
        
        # Update the current solution as the current optimal solution.
        for i in range(len(turnVal)):
            if(obj_type == -1):
                if(turnVal[i] > nowVal):
                    print("Better solution Found: ", turnVal[i])
                    nowVal = turnVal[i]
                    for j in range(len(nowX)):
                        nowX[j] = turnX[i][j]
            else:
                if(turnVal[i] < nowVal):
                    print("Better solution Found: ", turnVal[i])
                    nowVal = turnVal[i]
                    for j in range(len(nowX)):
                        nowX[j] = turnX[i][j]
        
        ansTime.append(time.time() - begin_time)
        ansVal.append(nowVal)
    # print(ansTime)
    # print(ansVal)

    solving_time = ansTime[-1]
    solving_obj = ansVal[-1]
    
    base_solving_time = None
    base_solving_obj = None
    print("\n---------------------------------------------------------\n")

    if compare_with_base:
        begin_time = time.time()
        model = Model()
        model.readProblem(test_instance_path)
        
        model.setParam('limits/time', set_time)
        model.optimize()
        
        base_solving_time = time.time() - begin_time
        base_solving_obj = model.getObjVal()
    
    return solving_time, solving_obj, base_solving_time, base_solving_obj



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", choices = ['IS', 'MVC', 'SC',  'CAT', 'MIPLIB'], default = 'MIPLIB', help = "Problem type selection")
    parser.add_argument("--difficulty_mode", choices = ['easy', 'medium', 'hard'], default = 'easy', help = "Difficulty level.")
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    parser.add_argument("--set_time", type = int, default = 100, help = 'set_time.')  # Maximum solving time
    parser.add_argument("--rate", type = float, default = 0.4, help = 'sub rate.')  # Maximum free variables ratio
    parser.add_argument("--gnn_model_path", type=str, default="GNN_trained_model.pkl")
    parser.add_argument("--gat_model_path", type=str, default="GAT_trained_model.pkl")
    parser.add_argument("--gbdt_model_path", type=str, default="GBDT.pickle")
    parser.add_argument("--method", choices = ['gbdt', 'gat', 'random'], default = 'gbdt', help = "predict method.")
    parser.add_argument("--graph_split_num", type=int, default=1, help="FENNEL cluster class number to split sub graphs to predict, 1 means not split and using full graph, 0 means auto(not implementation)")
    parser.add_argument("--reduction_rate", type=float, default=1.0, help="active constraints rate for srearching, set =1.0 if no need reduction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--save_file_tag", type=str, default='', help="tag for result file")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args)["model_path"])
    # args.difficulty_mode = 'medium'
    args.problem_type = 'SC'
    args.method = "gbdt"
    args.rate = 0.6
    # args.graph_split_num = 10
    # args.reduction_rate = 0.4
    args.save_file_tag = 'test'
    optimize(**vars(args))
    