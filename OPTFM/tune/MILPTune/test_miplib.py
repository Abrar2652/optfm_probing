import os
from milptune.configurator.knn import get_configuration_parameters
from milptune.scip.solver import solve_milp
import json

import pandas as pd


def get_no_timeout_instance(thread=8, method='SCIP-spx'):
    if thread == 1:
        name_list_thread = ["CBC", "COPT", "GLPK", "Gurobi", "lpsolve", "MATLAB", "SCIP-spx"]
    else:
        name_list_thread = ["CBC", "COPT", "FiberSCIP-spx ", "Gurobi", "HiGHS ", "SCIP-spx", "SCIP-cpx"]

    method_index = name_list_thread.index(method)

    no_timeout_instance = []
    with open(file='benchmark/mip_result_' + str(thread) + 'thread.txt', mode='rb') as f:
        for line in f:
            tmp_res = str(line).strip().split()
            tmp_name = tmp_res[0][2:]
            val = tmp_res[1:-1]
            if 'fail' not in val and 'abort' not in val and 'timeout' not in val and 'error' not in val and 'mismatch' not in val:
                no_timeout_instance.append(tmp_name)

    print("no_timeout_instance", len(no_timeout_instance))

    return no_timeout_instance


# dataset_name = '3_anonymous'
dataset_name = '1_item_placement'
dataset_dir = '/ml_nfs/' + dataset_name + '/test'

# iterative solve
data = []
methods = ['config', 'default']
for instances in os.listdir(dataset_dir):
    for mds in methods:
        # ins_str = instances[:-7]
        if "json" in instances:
            continue

        instance_path = dataset_dir + '/' + instances

        if mds == 'config':

            configs, distances = get_configuration_parameters(
                instance_file=instance_path,
                dataset_name=dataset_name,
                n_neighbors=1, n_configs=1)

            if configs is None:
                sol, primal, dual, time, gap, nnodes, status = solve_milp(
                    params=None,
                    instance=instance_path)
            else:
                sol, primal, dual, time, gap, nnodes, status = solve_milp(
                    params=configs[0]['params'],
                    instance=instance_path)

            data.append([instances, mds, primal, dual, time, gap, nnodes, status])

            print("-----------------------------Result------------------------------------")
            print(instances, primal, dual, time, gap, nnodes, status)
            print("-------------------------------End-------------------------------------")

        if mds == 'default':
            sol, primal, dual, time, gap, nnodes, status = solve_milp(
                params=None,
                instance=instance_path)

            data.append([instances, mds, primal, dual, time, gap, nnodes, status])

            print("-----------------------------Result------------------------------------")
            print(instances, primal, dual, time, gap, nnodes, status)
            print("-------------------------------End-------------------------------------")

        df = pd.DataFrame(data,
                          columns=['Instances', 'Methods', 'Primal Bound', 'Dual Bound', 'Solving Time', 'Gap', 'Nodes',
                                   'Status'])

        df.to_csv("config_comparison_simple.csv", index=False)
