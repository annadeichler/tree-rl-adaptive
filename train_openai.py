# -*- coding: utf-8 -*-
"""

runs episodic RL training with given hyperparameters for specified openai environment
executes one episode of training as subprocess for managing gym environment memory leak

example:
python train_openai.py --environment CartPole-v0 --hyperparameters hypes/alphago_racecar.json 
                            --outdir /home/../../ 
                            --output  example_output_name
                            --continue_training True

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import time
import pickle
import collections
from collections import deque
import tensorflow as tf
import gym
import numpy as np
import multiprocessing
import pickle
import random
from copy import deepcopy
import logging
gym.undo_logger_setup()
logging.basicConfig(filename='training.log',level=logging.WARNING)

from utils.utils_tf import *
from utils.utils import run_as_process
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_episode(
    d,          # multiprocessing dictionary, save information between episodes
    hypes,      # dict,dict of hyeperparameters for experiment
    modules,    # dict,dict of used modules
    outdir,     # string,output directory path for results
    ckpdir,     # string,directory path for tf saver checkpoints
    treedir,    # string, directory path for tree information built in MCTS  
    ep):        # current episode
    """ 
    runs one episode of training and trains agent's network 

    """
    # get data from previous episode
    env = d['env']
    t_all = d['t_all']
    experience = d['agent_experience']
    experience_new = d['agent_experience_new']
    agent_nu = d['agent_nu']
    agent_sigma = d['agent_sigma']
    # initialization
    root_index = env.reset()
    env_reset = False
    scores = deque(maxlen=100)  
    R, t = 0.0,0
    states =[]
    states.append(root_index)
    ep_data = {}
    mcts_data = {}
    compare_data = {}

    with tf.Session() as sess:
        # load session, graph and agent data
        tf_graph = build_training_graph(hypes, modules)
        tf_sess = start_session(hypes,outdir)
        load_weights(ckpdir,tf_sess)
        agent = tf_graph['agent']
        agent._experience = experience
        agent._experience_new = experience_new
        agent.nu = agent_nu
        agent.sigma = agent_sigma
        agent.clear_history()

        start_time=time.time()
        actions, states, states_mismatch_all = [],[],[]
        root_states,pi_network, pi_tree = [],[],[]
        pi_network,pi_tree = [], []
        ep_data = {}
        mcts_data = {}
        compare_data = {}
        # run episode
        while True:
        # for j in range(100):
            mcts_start = time.time()
            try:
                root_states.append(root_index)
            except:AttributeError
            a, stats = agent.select_action(hypes,modules,env,tf_sess['sess'],root_index, env_reset, t, t_all, ep, treedir)           
            s1, r ,terminal, _ = env.step(a)
            actions.append(a)
            states.append(s1)
            pi_network.append(agent.pi_NN)
            pi_tree.append(agent.pi_MCTS)
            mcts_data[t] = stats
            R += r 
            t += 1
            t_all += 1
            if t >= hypes['max_ep_steps'] or terminal:
                root_index = env.reset()
                end_time = time.time()
                print("episode: "+ str(ep) + ", reward " + str(R))
                break
            else:
                root_index = s1
                env_reset = False
        # train network at end of each episode
        if agent._experience.size > hypes['nn']['batch_size']:
            loss, grads_norm, grads_norm_clipped, grads_sum = agent.train_network(tf_graph, tf_sess)
            ep_data['loss_d1'] = loss
            ep_data['grads_norm_d1'] = grads_norm
        if agent._experience_new.size > hypes['nn']['batch_size']:
            loss, grads_norm, grads_norm_clipped, grads_sum = agent.train_network(tf_graph, tf_sess,'recent')
            ep_data['loss_d2'] = loss
            ep_data['grads_norm_d2'] = grads_norm
        nn_time = time.time()      
        # save network weights after training
        session_saver(tf_sess,hypes,ckpdir)

    # store information for next episode 
    d['agent_experience'] = agent._experience
    d['agent_experience_new'] = agent._experience
    d['agent_nu'] = agent.nu
    d['agent_sigma'] = agent.sigma
    d['env'] = env

    # save results   
    ep_data['actions'] = actions
    ep_data['rewards_sum'] = R
    ep_data['states'] = states
    ep_data['steps_count'] = t
    ep_data['wallclock'] = end_time - start_time
    ep_data['wallclock_nn'] = nn_time - start_time
    compare_data['pi_NN'] = pi_network
    compare_data['pi_MCTS'] = pi_tree
    compare_data['root_indices'] = root_states
    compare_data['states'] = states
    compare_data['picked_action'] = actions

    d['ep_data'] = ep_data
    d['compare_data'] = compare_data
    d['mcts_data'] = mcts_data 
    d['t_all'] = t_all
    

def init_train(data, hypes,modules,ckpdir):
    with tf.Session() as sess:
      
        graph = build_training_graph(hypes, modules)
        print(graph['agent'].V_hat.__dict__)
        # print([node.name for node in tf.get_default_graph().as_graph_def().node])
        tf_sess = start_session(hypes,ckpdir)
        hypes['solver']['global_step'] = graph['global_step']

        env = modules['environment']
        root_index = env.reset()

        data['env'] = env
        data['root_index'] = root_index
        # data['agent_state'] = None
        data['agent_experience'] = graph['agent']._experience
        data['agent_experience_new'] = graph['agent']._experience_new    
        data['agent_nu'] =  graph['agent'].nu
        data['agent_sigma'] =  graph['agent'].sigma

        session_saver(tf_sess, hypes, ckpdir)

def do_training(
    hyperparams,  # dict of hyeperparameters for experiment
    modules,    # optimization, tree_search, model
    outdir,     # output directory for results
    ckpdir,     # directory for tf saver checkpoints
    treedir,
    output):
    """
    creates tensorflow session, graph, runs and saves experiment results
    """
    print("Starting training, calling session")
    # dictionary to keep information from processes
    manager = multiprocessing.Manager()
    d = manager.dict()
    # logging initialization
    t_all = 0
    ep = 0
    d['t_all'] = t_all
    scores = deque(maxlen=100)
    results = collections.defaultdict(dict)
    results_mcts = collections.defaultdict(dict)
    results_compare = collections.defaultdict(dict)
    # training initialization
    run_as_process(init_train,d,hyperparams,modules,ckpdir)
    env = get_env(d,modules,hyperparams)

    print("max episode steps: " + str(hyperparams['max_ep_steps']))

    # start training loop
    t_end = time.time() + hyperparams['max_wallclock_time']

    while (time.time() < t_end):
    # for k in range(1):
        # run episode and train network as subprocess (created env objects deleted)
        env_copy = deepcopy(d['env'])
        run_as_process(run_episode,d,hyperparams,modules,outdir,ckpdir,treedir,ep)
        results[ep]= d['ep_data']
        results_mcts[ep] = d['mcts_data']
        results_compare[ep] = d['compare_data']
        ep += 1
        scores.append(d['ep_data']['rewards_sum'])
        mean_score = np.mean(scores)

        if ep%20 == 0:
            print('[Episode {}] - Mean survival time over last 20 episodes was {} ticks'.format(ep, mean_score))
        # save partial results
        if ep%5 == 0:
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_main.pkl'), 'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_mcts.pkl'), 'wb') as f:
                pickle.dump(results_mcts, f)
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_compare.pkl'), 'wb') as f:
                pickle.dump(results_compare, f)

            results_mcts = {}
            results = {}
            results_compare = {}


    with open(os.path.join(outdir, str(output)+'_main.pkl'), 'wb') as f:
      pickle.dump(results,f)
    with open(os.path.join(outdir, str(output) + '_mcts.pkl'), 'wb') as f:
      pickle.dump(results_mcts,f)
    with open(os.path.join(outdir, str(output) + '_compare.pkl'), 'wb') as f:
      pickle.dump(results_compare,f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hyperparameters',help = 'path to configuration json file')
    parser.add_argument(
        '--environment',help = 'rl environment agent')
    parser.add_argument(
        '--output',help = 'results pkl name')
    parser.add_argument(
        '--outdir_hype', help = 'put results to same folder as hype json file',default = True)
    parser.add_argument(
        '--outdir', help = 'put results to  folder',default = True)
    parser.add_argument(
        '--continue_training', help = 'continue training from last weight',default = True)       
    args = parser.parse_args()
        
    return args

def main(arg):
    args = parse_args()
    continue_training = args.continue_training
    hyperparam_path = args.hyperparameters
    file_name = args.output
    script_path = (os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(script_path,hyperparam_path), 'r') as f:
        hyperparams = json.load(f)
    hyperparams['base_path'] = script_path
    env = args.environment

    if args.outdir: 
        outdir = args.outdir
    else:
        outdir = os.path.dirname(hyperparam_path)
    # create dir for checkpoint files
    print(outdir)
    outdir = os.path.join(outdir,file_name)
    if not os.path.exists(outdir): os.makedirs(outdir)
    ckpdir = os.path.join(outdir,'checkpoints')
    treedir = os.path.join(outdir,'trees')
    print("saving checkpoints to " + str(ckpdir))
    if not os.path.exists(ckpdir): os.makedirs(ckpdir)
    if not os.path.exists(treedir): os.makedirs(treedir)
    if continue_training:
            base_dir = '/'.join(outdir.rsplit('/')[:-1])
            name = outdir.rsplit('/')[-1]
            hyperparam_path = os.path.join(base_dir, str(name) + '.json')
            with open(os.path.join(outdir,hyperparam_path), 'r') as f:
                hyperparams = json.load(f)               
            hyperparams['base_path'] = (os.path.dirname(os.path.realpath(__file__)))        
    modules = load_modules(hyperparams, env)
    do_training(hyperparams,modules,outdir,ckpdir,treedir,file_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

