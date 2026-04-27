from ecole.observation import MilpBipartite
from pyscipopt import Model
from ecole.scip import Model as EcoleModel
import numpy as np
import math
import pickle
import os


def bipartite_graph_generation_scip(data_dir, instance):
    if not os.path.exists(data_dir + "/" + instance):
        return
    
    print(data_dir + "/" + instance)
    
    file_size = os.path.getsize(data_dir + "/" + instance) / (1024*1024)
    if file_size == 0 or file_size > 100:
        return

    model = Model()
    try:
        model.readProblem(data_dir + "/" + instance)
        model = EcoleModel.from_pyscipopt(model)
        features_extractor = MilpBipartite()
        features = features_extractor.extract(model, False)  
    except Exception as e:
        return
    
    var_feas = features.variable_features.astype(np.float32)  # Variable features
    
    cons_feas = features.constraint_features.astype(np.float32)  # Constraint features
        
    edge_indices = features.edge_features.indices.astype(np.int32) # 第一维是约束节点，第二维是变量节点
    edge_features = features.edge_features.values.astype(np.float32)
    edge_features = np.expand_dims(edge_features, axis=1)
    
    def replace_inf(matrix):
        replaced_matrix = [
            [0 if math.isinf(x) else x for x in row]
            for row in matrix
        ]
        
        return replaced_matrix
    
    var_feas = replace_inf(var_feas)
    cons_feas = replace_inf(cons_feas)
    
    return var_feas, cons_feas, edge_indices, edge_features
    
    

def bipartite_graph_generation(data_dir, instance, graph_dir):
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    
    if not os.path.exists(data_dir + "/" + instance):
        return
    
    instance_name = instance.split(".")[0]
    output_dir = graph_dir + '/' + instance_name + '.pickle'
    print(data_dir + "/" + instance)
    if os.path.exists(output_dir):
        return
    
    file_size = os.path.getsize(data_dir + "/" + instance) / (1024*1024)
    if file_size == 0 or file_size > 100:
        return

    model = Model()
    try:
        model.readProblem(data_dir + "/" + instance)
        model = EcoleModel.from_pyscipopt(model)
        features_extractor = MilpBipartite()
        features = features_extractor.extract(model, False)  
    except Exception as e:
        return
    
    var_feas = features.variable_features.astype(np.float32)  # Variable features
    
    cons_feas = features.constraint_features.astype(np.float32)  # Constraint features
        
    edge_indices = features.edge_features.indices.astype(np.int32) # 第一维是约束节点，第二维是变量节点
    edge_features = features.edge_features.values.astype(np.float32)
    edge_features = np.expand_dims(edge_features, axis=1)
    
    def replace_inf(matrix):
        replaced_matrix = [
            [0 if math.isinf(x) else x for x in row]
            for row in matrix
        ]
        
        return replaced_matrix
    
    var_feas = replace_inf(var_feas)
    cons_feas = replace_inf(cons_feas)
    
    with open(graph_dir + '/' + instance_name + '.pickle', 'wb') as f:
        pickle.dump([var_feas, cons_feas, edge_indices, edge_features], f)


def bipartite_graph_generation_scip_for_test(data_dir, instance):
    if not os.path.exists(data_dir + "/" + instance):
        return
    
    print(data_dir + "/" + instance)
    
    file_size = os.path.getsize(data_dir + "/" + instance) / (1024*1024)
    if file_size == 0 or file_size > 100:
        return

    model = Model()
    try:
        model.readProblem(data_dir + "/" + instance)
        model = EcoleModel.from_pyscipopt(model)
        features_extractor = MilpBipartite()
        features = features_extractor.extract(model, False)  
    except Exception as e:
        return
    
    var_feas = features.variable_features.astype(np.float32)  # Variable features

    cons_feas = features.constraint_features.astype(np.float32)  # Constraint features

    edge_indices = features.edge_features.indices.astype(np.int32) # 第一维是约束节点，第二维是变量节点
    edge_features = features.edge_features.values.astype(np.float32)
    
    graph_feas = [(start, end, feature) for start, end, feature in zip(edge_indices[1], edge_indices[0], edge_features)]
    
    edge_features = np.expand_dims(edge_features, axis=1)
    
    def replace_inf(matrix):
        replaced_matrix = [
            [0 if math.isinf(x) else x for x in row]
            for row in matrix
        ]
        
        return replaced_matrix
    
    var_feas = replace_inf(var_feas)
    cons_feas = replace_inf(cons_feas)
    
    col_bounds = [[row[7], row[8]] for row in var_feas]
    row_bounds = [[-float("inf"), row[0]] for row in cons_feas]
    
    return var_feas, cons_feas, edge_indices, edge_features, graph_feas, col_bounds, row_bounds


if __name__=='__main__':
    
    train_valid_data_dir = "/ml_nfs/Research/MILPLIBGen/milp-fbgena/HeuristicGen/milplib2017"
    test_data_dir = "/ml_nfs/Research/MILPLIBGen/ACM-MILP-main/data/milplib2017/train"
    
    # generate_graph_list = ["markshare_4_0", "markshare_5_0", "markshare1", "markshare2", "gen-ip054", "pk1", "gen-ip016", "gen-ip002", "gen-ip021", "supportcase14"]
    generate_graph_list = os.listdir(train_valid_data_dir)

    graph_output_train_dir = "/ml_nfs/Graphs_SCIP/"
    graph_output_valid_dir = "/ml_nfs/Graphs_valid_SCIP/"
    graph_output_test_dir = "/ml_nfs/Graphs_test_SCIP/"
    
    for test_instance_name in generate_graph_list:
        cur_train_valid_dir = train_valid_data_dir + "/" + test_instance_name
        for i in range(4):
            test_instance = test_instance_name + "_" + str(i) + ".mps"
            cur_instance_path = cur_train_valid_dir + "/" + test_instance
            if os.path.exists(cur_instance_path):
                bipartite_graph_generation(cur_train_valid_dir, test_instance, graph_output_train_dir)
        
        for i in range(4,5):
            test_instance = test_instance_name + "_" + str(i) + ".mps"
            cur_instance_path = cur_train_valid_dir + "/" + test_instance
            if os.path.exists(cur_instance_path):
                bipartite_graph_generation(cur_train_valid_dir, test_instance, graph_output_valid_dir)
    
    for test_instance_name in generate_graph_list:
        test_instance = test_instance_name + ".mps"
        bipartite_graph_generation(test_data_dir, test_instance, graph_output_test_dir)