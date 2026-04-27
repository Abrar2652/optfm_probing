import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import csv
import pandas as pd

import pyscipopt as scip
import utilities

import time
from collections import deque
import pickle
import random

from ddpg.ddpg_learner import DDPG
from ddpg.models import MLPPolicy, MLPPolicy_critic
from ddpg.memory import Memory
from ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from ddpg.common import set_global_seeds
import ddpg.common.tf_util as U

import shutil
import networkx as nx

from pyscipopt import Model

from warnings import simplefilter
from parse import parser_add_main_args, parse_method_mip
from utilities import load_batch_states

simplefilter(action='ignore', category=FutureWarning)


def stan(data):
    # mu = np.mean(data, axis=0)
    # sigma = np.std(data, axis=0)

    max_ = np.max(abs(data), axis=0)

    # sigma[np.where(sigma == 0)] = 1
    max_[np.where(max_ < 1e-6)] = 1

    return data/max_

def make_samples(in_queue, is_maximum):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    episode, instance, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    # print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance)) 

    if eval_flag==1:
        seed=0
    else:
        seed=0

    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    # utilities.init_scip_paramsR(m, seed=seed)
    utilities.init_scip_paramsR_setcover(m, seed=seed)

    m.setIntParam('timing/clocktype', 1)
    m.setRealParam('limits/time', time_limit)   # 设定求解时间，避免训练时间无限延长

    varss = [x for x in m.getVars()]   # 取出全部变量

    minimum_k = np.where(np.array(actions.squeeze())<0.5)
    min_k = minimum_k[0]

    if eval_flag == 1:
        m.addCons(sum(abs(varss[i] - obs[i]) for i in min_k) <= 5)
    else:
        for i in min_k:
            a,b = m.fixVar(varss[i],obs[i])  
   
    m.optimize()

    # print(m.getPrimalbound())

    if abs(m.getPrimalbound()) > 1e15:
        K = obs   # 未得到可行解的情况下，各变量的取值保持不变
        if is_maximum == 1:
            obj = -abs(m.getPrimalbound())  # 若最大化问题，目标值赋予负无穷小
        else:
            obj = abs(m.getPrimalbound())  # 若最小化问题，则目标值赋予无穷大
    else:
        K = [m.getVal(x) for x in m.getVars()]   #获取各变量的取值
        obj = m.getObjVal()

    # obj = m.getPrimalbound()  # 获取当前最优解
        
    m.freeProb() 
        
    out_queue = {
        'type': 'solution',
        'episode': episode,
        'instance': instance,
        'sol' : np.array(K),
        'obj' : obj,
        'seed': seed,
        'mask': min_k[0],
    }               
    
    return out_queue


def send_orders(instances, epi, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    orders_queue = []
    for i in range(len(instances)):
        seed = rng.randint(2**32)
        orders_queue.append([epi[i], instances[i], obs[i], actions[i], seed, exploration_policy, eval_flag, time_limit, out_dir])

    return orders_queue


def collect_samples(instances, epi, obs, actions, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, eval_flag, time_limit, is_maximum):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)
    
    pars = send_orders(instances, epi, obs, actions, rng.randint(2**32), exploration_policy, eval_flag, time_limit, tmp_samples_dir) 
    
    out_Q = []
    for n in range(len(pars)):
        out_queue = make_samples(pars[n], is_maximum)
        out_Q.append(out_queue)        

    # record answers 
    i = 0
    collecter=[]
    epi=[]
    obje=[]
    instances=[]
    mask=[]

    for sample in out_Q:
        
        collecter.extend(sample['sol'])
        
        epi.append(sample['episode'])
        
        if is_maximum == 1:
            obje.append(sample['obj'])
        else:
            obje.append(-sample['obj'])

        instances.append(sample['instance'])

        mask.append(sample['mask'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
   
    return np.array(collecter), np.stack(epi), np.stack(obje), instances, mask

##########  initial formulation features    
class SamplingAgent0(scip.Branchrule):

    def __init__(self, episode, instance, seed, exploration_policy, out_dir):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.exploration_policy = exploration_policy
        self.out_dir = out_dir

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        # custom policy branching           
        if self.model.getNNodes() == 1:
            
            # extract formula features
            self.state = utilities.extract_state(self.model, self.state_buffer)              

            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
                               
        elif self.model.getNNodes() != 1:
               
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)   
            
        else:
            raise NotImplementedError

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


def make_samples0(in_queue, node_limit, initial_solution_heu, pre_solve, conflict):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

#     while True:
    episode, instance, seed, exploration_policy, time_limit, out_dir = in_queue
    # print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance))
    
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    # utilities.init_scip_paramsH(m, seed=seed)
    utilities.init_scip_paramsH_setcover(m, heuristics=initial_solution_heu, seed=seed)
    m.setIntParam('timing/clocktype', 2)
    if node_limit:
        m.setLongintParam('limits/nodes', 1)  # 仅处理当前节点
    else:
        m.setParam('limits/solutions', 1)
        m.setRealParam('limits/time', 100) 

    abc=time.time()
    # print(m)
    # print("------------------------------------------------------------------------")
    m.optimize()       
    # print(time.time()-abc)

    if abs(m.getPrimalbound()) > 1e15:
        return None

    b_obj = m.getObjVal()  # 得到当前最优解

    K = [m.getVal(x) for x in m.getVars()]       # 取出每个变量取值 

    out_queue = {
        'type': 'formula',
        'episode': episode,
        'instance': instance,
        'seed': seed,
        'b_obj': b_obj,
        'sol' : np.array(K),        
    }   

    # print(b_obj)
       
    m.freeTransform()  
        
    obj = [x.getObj() for x in m.getVars()]  
    
    out_queue['obj'] = sum(obj) 
    
    m.freeProb() 
        
    # print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue


def send_orders0(instances, n_samples, seed, exploration_policy, batch_id, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    st = batch_id*n_samples
    orders_queue = []
    for i in instances[st:st+n_samples]:     
        seed = rng.randint(2**32)
        orders_queue.append([episode, i, seed, exploration_policy, time_limit, out_dir])
        episode += 1
    return orders_queue



def collect_samples0(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, batch_id, time_limit, is_maximum, node_limit, initial_solution_heu, presolve, conflict):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)
    
    pars = send_orders0(instances, n_samples, rng.randint(2**32), exploration_policy, batch_id, time_limit, tmp_samples_dir)  
    
    out_Q = []
    for n in range(len(pars)):
        out_queue = make_samples0(pars[n], node_limit, initial_solution_heu, presolve, conflict)
        if out_queue is None:
            continue
        out_Q.append(out_queue)        
        

    # record answers and write samples
    i = 0

    epi=[]
    instances=[]
    obje=[]
    bobj=[]
    ini_sol=[]

    
    for sample in out_Q:

        # print("-------------------------------------------------------")
        
        ini_sol.append(np.array(sample['sol']))
                
        epi.append(sample['episode'])
        
        instances.append(sample['instance'])
        
        obje.append(sample['obj'])

        if is_maximum == 1:
            bobj.append(sample['b_obj'])
        else:
            bobj.append(-sample['b_obj'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)

    return np.stack(epi), np.stack(obje), np.stack(bobj), instances, np.concatenate(ini_sol, axis=0)  # 需要将初始解concat至大图上


def pad_output(output, n_vars_per_sample, pad_value=-1e8):

    new_output = []

    start_index = 0
    for cur_length in n_vars_per_sample:
        end_index = start_index + cur_length
        new_output.append(output[0][start_index:end_index])
        start_index = end_index
    
    return new_output

    # n_vars_max = tf.reduce_max(n_vars_per_sample)

    # output = tf.split(
    #     value=output,
    #     num_or_size_splits=n_vars_per_sample,
    #     axis=1,
    # )
    # output = tf.concat([
    #     tf.pad(
    #         x,
    #         paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
    #         mode='CONSTANT',
    #         constant_values=pad_value)
    #     for x in output
    # ], axis=0)


    # return output


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


def learn(args, is_maximum=1,network='mlp',
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=25,
          nb_rollout_steps=20,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type=None, #'normal_0.01',
          normalize_returns=False,
          normalize_observations=False,
          critic_l2_reg=1e-2,
          actor_lr=1e-5,  # default:1e-5
          critic_lr=1e-5,  # default:1e-5
          popart=False,
          gamma=0.99,  #0.9 #0.96
          clip_norm=None,
          nb_train_steps=40, # default:6
          nb_eval_steps=50, # default:50 TODO:50
          batch_size=3, # default:10
          tau=0.01,
          eval_env=None,
          load_path=None,
          param_noise_adaption_interval=30):

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/RL_model', exist_ok=True)
    os.makedirs('models/RL_model/' + args.problem, exist_ok=True)
    
    emb_size = 16

    # save_path = 'models/RL_model/' + args.problem + "/model_graph.joblib"
    save_path = 'models/RL_model/' + args.problem

    print("seed {}".format(args.seed))

    batch_sample = 8  #8  #TODO:10
    batch_sample_eval = 1 # default:20  TODO:10 for setcover,20 for maxcut
    exploration_strategy = 'relpscost'  # reliability branching on pseudo cost values
    time_limit = 5  # 5  #2
    nb_rollout_steps = 50   # default:50 TODO:50

    pre_solve = True
    conflict = True

    fix_ratio = 0.3

    instances_valid = []
    instances_train = []
    
    if args.problem == 'maxcut':
        instances_train = glob.glob('data/instances/train_4950_2975/*.lp')
        instances_valid += ["data/instances/valid_4950_2975/instance_{}.lp".format(i+1) for i in range(20)]
        out_dir = 'data/samples/tmp'

        is_maximum = 1
        initial_solution_heu = True  # initial feasible solution
        node_limit = True  # limited to one node/ limit to one solution

        is_branching = False  # whether branching in training
        time_limit = 2

        sub_mip_ratio = 1  # limited size ratio for sub-mip problems

    ### number of epochs, cycles, steps
    nb_epochs = 500
    nb_epoch_cycles = len(instances_train)//batch_sample

    print("{} train instances for {} samples".format(len(instances_train),nb_epoch_cycles*nb_epochs*batch_sample))

    # define parameters
    memory_size = 100  # Total memory size, 超出范围后按照从前往后的顺序替换

    memory = Memory(memory_size, batch_size)
    critic = MLPPolicy_critic(batch_size, emb_size)
    actor = MLPPolicy(emb_size)
    
    action_noise = None  # 对采样action的噪声扰动
    param_noise = None

    # TODO:noise type
    agent = DDPG(actor, critic, memory, gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    # ### TENSORFLOW SETUP ###
    # if args.gpu == -1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # else:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)                    
   
    sess = U.get_session()
    # Prepare everything.
    agent.initialize(sess)

    if load_path is not None:  # 加载已经训练好的模型
        agent.load(load_path)

    # sess.graph.finalize()

    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))

    agent.reset()
    nenvs = batch_sample

    num_samples = batch_sample

    # create output directory, throws an error if it already exists
    t = 0 # scalar

    max_obj = -np.inf
    model_index = 0
    #### start train
    for epoch in range(nb_epochs):
        print("Current epoch: ", epoch)
        random.shuffle(instances_train)

        fieldnames = [
            'instance',
            'obj',
            'initial',
            'bestroot',
            'imp',
            'mean',
            'time',
        ]
        result_file = "{}_{}.csv".format(args.problem,time.strftime('%Y%m%d-%H%M%S'))    
        os.makedirs('ddpg_mean_results', exist_ok=True)
        with open("ddpg_mean_results/{}".format(result_file), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()    

            for cycle in range(nb_epoch_cycles):        # nb_epoch_cycles
                print("Current batch: ", cycle)
                ### initial formulation features, 按顺序对各batch进行处理
                epi, ori_objs, best_root, instances, ini_sol = collect_samples0(instances_train, out_dir + '/train', rng, batch_sample,
                                args.njobs, exploration_policy=exploration_strategy,
                                batch_id=cycle,
                                time_limit=None,
                                is_maximum=is_maximum,
                                node_limit = node_limit,
                                initial_solution_heu=initial_solution_heu, 
                                presolve = pre_solve,
                                conflict = conflict)
                
                num_samples = len(ori_objs)

                ### initial solution
                init_sols = ini_sol

                ori_objs=np.copy(best_root)   # 当前最优解

                cur_sols=init_sols

                # Perform rollouts.
                if nenvs > 1:
                    # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                    # of the environments, so resetting here instead
                    agent.reset()

                rec_inc = []  #保存各sample的feasible solution list

                rec_inc_obj = []  #保存各sample对应feasible solution list的objective value

                rec_inc.append(init_sols)  # 每个元素的长度为batch内全部variable num之和

                rec_inc_obj.append(best_root)  # 每个元素的长度为batch内全部variable num之和


                rec_best = np.copy(best_root)                 #当前最优解
                inc_val = rec_inc[-1].copy()  #  当前时刻最优解对应各变量取值

                avg_inc_val = np.array(rec_inc).mean(0)  # 各variable平均solution取值

                pre_sols = np.ones([2, len(rec_inc[0])])  # variable feature

                # Get the pretrained embeddings
                v_features, n_vs_per_sample = load_batch_states(args, instances)
                
                # Static features normalization
                # c_features = stan(c_features)
                # e_features = stan(e_features)
                # v_features = stan(v_features)

                dynamic_variable_features = np.concatenate((inc_val[np.newaxis,:], avg_inc_val[np.newaxis,:], pre_sols), axis=0)

                # print(v_features.shape)
                # print(dynamic_variable_features.shape)
                updated_v_features = np.concatenate((v_features, dynamic_variable_features.transpose(1,0)), axis=1)
                
                # current observations
                cur_obs = [updated_v_features, n_vs_per_sample]
                mask = None

                count_array = np.array([0 for i in range(sum(n_vs_per_sample))])
                
                for t_rollout in range(nb_rollout_steps):      #在每个epoch，对每个sample迭代多少轮
                    
                    action, q, _, _ = agent.step(cur_obs, apply_noise=True, compute_Q=True)

                    pre_sols = np.concatenate((pre_sols,cur_sols[np.newaxis,:]), axis=0) 

                    action = np.random.binomial(1, action)   #一次伯努利采样

                    sample = np.where(action[0] == 1.0)[0]  # 当前step需要求解的变量集合
                                        
                    sub_mip_length = int(len(action[0]) * sub_mip_ratio)

                    count_prob_array = (max(count_array) + 1) - count_array
                    count_prob_array = count_prob_array[sample]
                    count_prob = count_prob_array/sum(count_prob_array)

                    if len(sample) > sub_mip_length:
                        idx = np.random.choice(sample, len(sample) - sub_mip_length, replace=False, p=count_prob)
                    else:
                        idx = []
                    
                    solve_ids = np.array(list(set(sample) - set(idx)))
                    count_array[solve_ids] += 1
                    
                    action = np.where(action > 0.5, action, 0.)
                    action = np.where(action == 0., action, 1.)

                    # action[0] = action[0] * sample
                    action[0][idx] = 0

                    # action=np.where(action > 0.5, action, 0.)  # 小于0.5置0
                    # action=np.where(action == 0., action, 1.)

                    action = pad_output(action, n_vs_per_sample)  # 还原到每个sample

                    # action = sess.run(action)

                    # for i in range(len(n_vs_per_sample)):
                    #     action[i] = action[i][:n_vs_per_sample[i]]  # 删掉补0的部分

                    sample_cur_sols = pad_output(cur_sols[np.newaxis,:], n_vs_per_sample)

                    a=time.time()
                    # Execute next action. derive next solution(state)
                    next_sols, epi, cur_objs, instances, mask = collect_samples(instances, epi, sample_cur_sols, action, out_dir + '/train', rng, batch_sample,
                                    args.njobs, exploration_policy=exploration_strategy,
                                    eval_flag=0,
                                    time_limit=time_limit,
                                    is_maximum=is_maximum)
                                        
                    # print(cur_objs)
                    
                    # print(time.time()-a)
                    
                    cur_sols=next_sols.copy()  # 获取优化后的solution

                    start_index = 0
                    
                    if t_rollout>0:  # store transition
                        for ids in range(num_samples):
                            end_index = start_index + n_vs_per_sample[ids]
                            # if abs(r_s[ids]) * 2000 >= 1e-5 or random.random() > 0.6:  # 仅保留reward有意义的样本
                            #     agent.store_transition(current_states[ids], action_s[ids], r_s[ids],cur_obs_s[:,start_index:end_index], next_obs_s[:,start_index:end_index], action[ids], epi[ids], n_vs_per_sample[ids], n_cs_per_sample[ids])
                            agent.store_transition(v_features[start_index:end_index], action_s[ids], r_s[ids],cur_obs_s[:,start_index:end_index], next_obs_s[:,start_index:end_index], action[ids], epi[ids], n_vs_per_sample[ids])
                            start_index = end_index

                    inc_ind = np.where(cur_objs > rec_best)[0]     #确定当前解取得优化的位置
                    rec_inc.append(rec_inc[-1])

                    for inds in inc_ind:
                        start_index = sum(n_vs_per_sample[:inds])
                        end_index = start_index + n_vs_per_sample[inds]
                        rec_inc[-1][start_index:end_index] = cur_sols[start_index:end_index]

                    rec_best[inc_ind] = cur_objs[inc_ind]  #ADD

                    # compute rewards
                    if is_maximum:
                        r = cur_objs - ori_objs
                    else:
                        r = ori_objs - cur_objs
                    # print(r)
                    # note these outputs are batched from vecenv
                    t += 1
                    
                    inc_val = rec_inc[-1].copy() #ADD
                    avg_inc_val = np.array(rec_inc).mean(0)#ADD                     
                    
                    next_dynamic_variable_features = np.concatenate((inc_val[np.newaxis,:], avg_inc_val[np.newaxis,:], pre_sols[-2:]), axis=0)

                    next_updated_v_features = np.concatenate((v_features, next_dynamic_variable_features.transpose(1,0)), axis=1)

                    # current observations
                    next_obs = [next_updated_v_features, n_vs_per_sample]

                    cur_obs_s = dynamic_variable_features.copy()

                    fla_action = []
                    for acs in action:
                        fla_action.extend(list(acs))

                    fla_action = np.array(fla_action)
                    # fla_action[diff_idx] = 1

                    action_s = pad_output(fla_action[np.newaxis,:], n_vs_per_sample)

                    action_s = action.copy()

                    r_s = np.array([min(t_r/2000., 1) for t_r in r])
                    r_s = np.array([max(t_r, -1) for t_r in r_s])
                    # r_s = r
                    next_obs_s = next_dynamic_variable_features.copy()

                    cur_obs = next_obs
                    # ori_objs = np.maximum(ori_objs, cur_objs)
                    ori_objs = cur_objs.copy()  # TODO

                # Train，从memory中随机抽取样本执行训练
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                print("Start Training...................................")
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()
                
                # Evaluate  
                if cycle%1==0:
                    episodes = 0 #scalar
                    t = 0 # scalar
                    ave_list = []

                    for cyc in range(len(instances_valid)//batch_sample_eval):
                        obj_lis = []
                        ### initial formulation features
                        abcd=time.time()

                        epi, ori_objs, best_root, instances, ini_sol = collect_samples0(instances_valid, out_dir + '/train', rng, batch_sample_eval,
                                args.njobs, exploration_policy=exploration_strategy,
                                batch_id=cyc,
                                time_limit=None,
                                is_maximum=is_maximum,
                                node_limit=node_limit,
                                initial_solution_heu=initial_solution_heu, 
                                presolve = pre_solve,
                                conflict = conflict)

                        ### initial solution
                        init_sols = ini_sol

                        ori_objs=np.copy(best_root)  

                        cur_sols=init_sols                                

                        record_ini=np.copy(ori_objs)

                        rec_inc = []  #保存各sample的feasible solution list
                        rec_inc.append(init_sols)  # 每个元素的长度为batch内全部variable num之和

                        rec_best = np.copy(best_root)                 #ADD
                        
                        inc_val = rec_inc[-1].copy()  #  当前时刻最优解对应各变量取值
                        avg_inc_val = np.array(rec_inc).mean(0)  # 各variable平均solution取值

                        pre_sols = np.ones([2, len(rec_inc[0])])  # variable feature

                        # Get the pretrained embeddings
                        v_features, n_vs_per_sample = load_batch_states(args, instances)

                        # Static features normalization
                        # c_features = stan(c_features)
                        # e_features = stan(e_features)
                        # v_features = stan(v_features)

                        dynamic_variable_features = np.concatenate((inc_val[np.newaxis,:], avg_inc_val[np.newaxis,:], pre_sols), axis=0)
                        updated_v_features = np.concatenate((v_features, dynamic_variable_features.transpose(1,0)), axis=1)

                        # current observations
                        cur_obs = [updated_v_features, n_vs_per_sample]    

                        mask = None
              
                        # Perform rollouts.                
                        for t_rollout in range(nb_eval_steps):

                            action, q, _, _ = agent.step(cur_obs, apply_noise=True, compute_Q=True)

                            pre_sols = np.concatenate((pre_sols,cur_sols[np.newaxis,:]), axis=0) 

                            action = np.random.binomial(1, action)

                            action[0][np.where(v_features[:,2] == 1)] = 1

                            action=np.where(action > 0.5, action, 0.)  
                            action=np.where(action == 0., action, 1.)

                            action = pad_output(action, n_vs_per_sample)  # 还原到每个sample

                            # action = sess.run(action)

                            # for i in range(len(n_vs_per_sample)):
                            #     action[i] = action[i][:n_vs_per_sample[i]]  # 删掉补0的部分

                            sample_cur_sols = pad_output(cur_sols[np.newaxis,:], n_vs_per_sample)
                            # sample_cur_sols = sess.run(sample_cur_sols)

                            # for i in range(len(n_vs_per_sample)):
                            #     sample_cur_sols[i] = sample_cur_sols[i][:n_vs_per_sample[i]]

                            st_time = time.time()
                            # Execute next action. derive next solution(state)
                            next_sols, epi, cur_objs, instances, mask = collect_samples(instances, epi, sample_cur_sols, action, out_dir + '/train', rng, batch_sample_eval,
                                    args.njobs, exploration_policy=exploration_strategy,
                                    eval_flag=0,
                                    time_limit=time_limit,
                                    is_maximum=is_maximum)
                            
                            # print(time.time() - st_time)

                            cur_sols=next_sols.copy()  # 获取优化后的solution

                            inc_ind = np.where(cur_objs > rec_best)[0]
                            rec_inc.append(rec_inc[-1])

                            for inds in inc_ind:
                                start_index = sum(n_vs_per_sample[:inds])
                                end_index = start_index + n_vs_per_sample[inds]
                                rec_inc[-1][start_index:end_index] = cur_sols[start_index:end_index]

                            rec_best[inc_ind] = cur_objs[inc_ind]          

                            # compute rewards
                            if is_maximum:
                                r = cur_objs - ori_objs
                            else:
                                r = ori_objs - cur_objs
                                    
                            # print(r)
                            # note these outputs are batched from vecenv
                            t += 1

                            inc_val = rec_inc[-1].copy()
                            avg_inc_val = np.array(rec_inc).mean(0)

                            next_dynamic_variable_features = np.concatenate((inc_val[np.newaxis,:], avg_inc_val[np.newaxis,:], pre_sols[-2:]), axis=0)
                            next_updated_v_features = np.concatenate((v_features, next_dynamic_variable_features.transpose(1,0)), axis=1)

                            next_obs = [next_updated_v_features, n_vs_per_sample]

                            cur_obs = next_obs
                            ori_objs = cur_objs

                            # Book-keeping.
                            obj_lis.append(cur_objs)

                        print('time___________________________________')         
                        tt = time.time()-abcd      
                        print(time.time()-abcd)
                        if is_maximum:             
                            miniu = np.stack(obj_lis).max(axis=0)
                        else:
                            miniu = np.stack(obj_lis).min(axis=0)
                        ave = np.mean(miniu)
                        for j in range(batch_sample_eval):                 
                            writer.writerow({
                                'instance': instances[j],
                                'obj':miniu[j],
                                'initial':record_ini[j],
                                'bestroot':best_root[j],
                                'imp':miniu[j]-best_root[j],
                                'mean':ave,
                                'time':tt,
                            })
                            csvfile.flush()
                        
                        ave_list.append(ave)
                
                ave = np.mean(np.array(ave_list))

                if save_path is not None and ave>max_obj:  # 取得更优解时，保存结果
                    s_path = os.path.expanduser(save_path)
                    agent.save(s_path + "/model_graph_" + str(model_index) + ".joblib")
                    max_obj = ave
                    model_index += 1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Testing Pipeline for Edge Identification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    
    learn(args)  # is_maximum参数，1表示最大问题，0表示最小问题