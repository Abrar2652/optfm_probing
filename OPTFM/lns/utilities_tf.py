import pickle
import gzip
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def stan(data):
    max_ = np.max(abs(data), axis=0)

    # sigma[np.where(sigma == 0)] = 1
    max_[np.where(max_ < 1e-6)] = 1

    return data/max_


def stan_min_max(data):
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)

    diff = max_ - min_

    # sigma[np.where(sigma == 0)] = 1
    diff[np.where(diff < 1e-6)] = 1

    return (data-min_)/diff


def load_batch_gcnn(states_list):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []

    # Features for tripatite graph
    variable_objective_features = []
    variable_objective_indices = [] 
    constraint_objective_features = [] 
    constraint_objective_indices = [] 
    objective_features = []

    # load samples
    for current_state in states_list:

        sample_state = current_state

        c, e, v, v_o, v_o_indices, c_o, c_o_indices, o = sample_state
        c_features.append(stan(c['values']))
        e_indices.append(e['indices'])
        e_features.append(stan(e['values']))
        v_features.append(stan(v['values']))

        variable_objective_features.append(v_o)
        variable_objective_indices.append(v_o_indices)
        constraint_objective_features.append(c_o)
        constraint_objective_indices.append(c_o_indices)
        objective_features.append(o)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_objs_per_sample = [objs.shape[0] for objs in objective_features]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)

    variable_objective_features = np.concatenate(variable_objective_features, axis=0)
    constraint_objective_features = np.concatenate(constraint_objective_features, axis=0)
    objective_features = np.concatenate(objective_features, axis=0)

    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
        for j, e_ind in enumerate(e_indices)], axis=1)
    
    vo_shift = np.cumsum([
            [0] + n_vs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    variable_objective_indices = np.concatenate([e_ind + vo_shift[:, j:(j+1)]
        for j, e_ind in enumerate(variable_objective_indices)], axis=1)
    
    co_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    constraint_objective_indices = np.concatenate([e_ind + co_shift[:, j:(j+1)]
        for j, e_ind in enumerate(constraint_objective_indices)], axis=1)

    # convert to tensors
    # c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    # e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    # e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    # v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    # n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    # n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, variable_objective_features, variable_objective_indices, constraint_objective_features, constraint_objective_indices, objective_features


def load_batch_gcnn_branching(states_list):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []

    # Features for tripatite graph
    variable_objective_features = []
    variable_objective_indices = [] 
    constraint_objective_features = [] 
    constraint_objective_indices = [] 
    objective_features = []

    # load samples
    for current_state in states_list:

        sample_state = current_state

        c, e, v, v_o, v_o_indices, c_o, c_o_indices, o = sample_state
        c_features.append((c['values']))
        e_indices.append(e['indices'])
        e_features.append((e['values']))
        v_features.append((v['values']))

        variable_objective_features.append(v_o)
        variable_objective_indices.append(v_o_indices)
        constraint_objective_features.append(c_o)
        constraint_objective_indices.append(c_o_indices)
        objective_features.append(o)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_objs_per_sample = [objs.shape[0] for objs in objective_features]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)

    variable_objective_features = np.concatenate(variable_objective_features, axis=0)
    constraint_objective_features = np.concatenate(constraint_objective_features, axis=0)
    objective_features = np.concatenate(objective_features, axis=0)

    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
        for j, e_ind in enumerate(e_indices)], axis=1)
    
    vo_shift = np.cumsum([
            [0] + n_vs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    variable_objective_indices = np.concatenate([e_ind + vo_shift[:, j:(j+1)]
        for j, e_ind in enumerate(variable_objective_indices)], axis=1)
    
    co_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    constraint_objective_indices = np.concatenate([e_ind + co_shift[:, j:(j+1)]
        for j, e_ind in enumerate(constraint_objective_indices)], axis=1)

    # convert to tensors
    # c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    # e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    # e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    # v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    # n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    # n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, variable_objective_features, variable_objective_indices, constraint_objective_features, constraint_objective_indices, objective_features


def load_batch_gcnn_integer(states_list, integer_list, integer_length, integer_idx):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []

    # Features for tripatite graph
    variable_objective_features = []
    variable_objective_indices = [] 
    constraint_objective_features = [] 
    constraint_objective_indices = [] 
    objective_features = []

    # load samples

    integer_idx_copy = integer_idx.copy()

    for i in range(len(states_list)):

        integer_idx = integer_idx_copy.copy()

        sample_state = states_list[i]

        c, e, v, v_o, v_o_indices, c_o, c_o_indices, o = sample_state
        ori_v_features = v['values']
        integer_idx = np.array(integer_idx).copy()

        add_features_list = []
        add_e_features = []
        add_v_o_features = []
        add_e_index = []
        add_vo_index = []
        # for j in range(len(integer_idx[i])):
        for j in range(len(integer_idx[i])):
            print(j)
            # print(integer_idx[i][j])
            add_features_list.append(ori_v_features[integer_idx[i][j]])
            add_e_features.append(e['values'][integer_idx[i][j]])
            add_v_o_features.append(v_o[integer_idx[i][j]])

            tmp = []
            for k in range(len(e['indices'][0])):
                target = e['indices'][0][k]
                cur_var = e['indices'][1][k]
                if integer_idx[i][j] == cur_var:
                    tmp.append(target)
            add_e_index.append(tmp)

            tmp = []
            for k in range(len(v_o_indices[0])):
                target = v_o_indices[0][k]
                cur_var = v_o_indices[1][k]
                if integer_idx[i][j] == cur_var:
                    tmp.append(target)
            add_vo_index.append(tmp)

        for j in range(len(integer_idx[i])):
            cur_interger_id = integer_idx[i][j]

            for times in range(integer_length[i][j] - 1):
                ori_v_features = np.insert(ori_v_features, cur_interger_id + 1,
                                           add_features_list[j], axis=0)

            for edges in range(len(add_e_index[j])):
                cur_target = add_e_index[j][edges]
                for times in range(integer_length[i][j] - 1):
                    e_0 = np.insert(e['indices'][0], 0,
                                    cur_target)
                    e_1 = np.insert(e['indices'][1], 0,
                                                [cur_interger_id + times])
                    e['indices'] = np.array([e_0, e_1])
                    e['values'] = np.insert(e['values'], 0,
                                            add_e_features[j], axis=0)

            for edges in range(len(add_vo_index[j])):
                cur_target = add_vo_index[j][edges]
                for times in range(integer_length[i][j] - 1):
                    v_0 = np.insert(v_o_indices[0], 0,
                                    [cur_target], axis=0)
                    v_1 = np.insert(v_o_indices[1], 0,
                                                [cur_interger_id + times], axis=0)
                    v_o_indices = np.array([v_0, v_1])
                    v_o = np.insert(v_o, 0,
                                    add_v_o_features[j], axis=0)

            integer_idx = integer_idx + integer_length[i][j] - 1
        
        c_features.append((c['values']))
        e_indices.append(e['indices'])
        e_features.append((e['values']))
        # v_features.append((v['values']))
        v_features.append(ori_v_features)

        variable_objective_features.append(v_o)
        variable_objective_indices.append(v_o_indices)
        constraint_objective_features.append(c_o)
        constraint_objective_indices.append(c_o_indices)
        objective_features.append(o)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_objs_per_sample = [objs.shape[0] for objs in objective_features]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)

    variable_objective_features = np.concatenate(variable_objective_features, axis=0)
    constraint_objective_features = np.concatenate(constraint_objective_features, axis=0)
    objective_features = np.concatenate(objective_features, axis=0)

    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
        for j, e_ind in enumerate(e_indices)], axis=1)
    
    vo_shift = np.cumsum([
            [0] + n_vs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    variable_objective_indices = np.concatenate([e_ind + vo_shift[:, j:(j+1)]
        for j, e_ind in enumerate(variable_objective_indices)], axis=1)
    
    co_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_objs_per_sample[:-1]
        ], axis=1)
    constraint_objective_indices = np.concatenate([e_ind + co_shift[:, j:(j+1)]
        for j, e_ind in enumerate(constraint_objective_indices)], axis=1)

    # convert to tensors
    # c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    # e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    # e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    # v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    # n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    # n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, variable_objective_features, variable_objective_indices, constraint_objective_features, constraint_objective_indices, objective_features