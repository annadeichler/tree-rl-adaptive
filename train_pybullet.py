# -*- coding: utf-8 -*-
"""

runs episodic RL training with given hyperparameters for specified pybullet environment

example:
python train_pybullet.py --environment Racecar-v0 --hyperparameters hypes/alphago_racecar.json 
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
import logging
import resource
from PIL import Image
import matplotlib.pyplot as plt


from utils.utils_tf import *
from utils.utils import run_as_process

gym.undo_logger_setup()
logging.basicConfig(filename='training.log',level=logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_training(
    hypes,      # dict,dict of hyeperparameters for experiment
    modules,    # dict,dict of used modules
    graph,      # TF graph
    tf_sess,    # TF session
    outdir,     # string,output directory path for results
    ckpdir,     # string,directory path for tf saver checkpoints
    treedir,    # string, directory path for tree information built in MCTS   
    output):    # # string,base name of result files
    """ 
    runs episodic RL training
    """
    # initialization
    t_all = 0
    ep = 0
    results = collections.defaultdict(dict)
    results_mcts = collections.defaultdict(dict)
    results_compare = collections.defaultdict(dict)

    # setup learning environment
    agent = graph['agent']
    agent_type = hypes['model']['agent_type']
    env = modules['environment']
    
    print("max episode steps: " + str(hypes['max_ep_steps']))
    t_end = time.time() + hypes['max_wallclock_time']
    root_index = env.reset()
    env.save_checkpoint()
    env_reset = False
    scores = deque(maxlen=100)  
    R, t = 0.0,0
    states =[]
    actions = []
    print("start training")
    while (time.time() < t_end):
        terminal = False
        t,R = 0, 0.0
        env_reset = True
        agent.clear_history()
        start_time=time.time()
        actions, states = [],[]
        root_states,pi_network, pi_tree = [],[],[]
        mus, sigmas, etas = [],[],[]
        pi_network,pi_tree = [], []
        ucbs = []
        currvars = []
        avgvars = []
        niterations = []
        ep_data = {}
        mcts_data = {}
        compare_data = {}
        decision_types = []
        child_priors = []
        # run episode until maximum number of time steps or terminal state
        while True:
            mcts_start = time.time()
            try:
                root_states.append(root_index)
            except:AttributeError
            a, stats = agent.select_action(hypes,modules,env,tf_sess['sess'],root_index, env_reset, t, t_all,ep,treedir)
            s1, r ,terminal, _ = env.step(a)
            child_priors.append(agent.child_priors)
            env.save_checkpoint()
            actions.append(a)
            states.append(s1)
            mus.append(agent.nu)
            sigmas.append(agent.sigma)
            etas.append(agent.eta)
            ucbs.append(agent._ucb_param)
            pi_network.append(agent.pi_NN)
            pi_tree.append(agent.pi_MCTS)
            mcts_data[t] = stats
            R += r 
            t += 1
            t_all += 1

            #save rendered image
            # if ep%25==
            # if t%5    ==0:
            #   rgb = env.save_render()
            #   im = Image.fromarray(rgb)
            #   im.save("/home/anna/"+str(t)+"your_file.jpeg")
            #   print(rgb)

            if t >= hypes['max_ep_steps']:
                root_index = env.reset()
                env.save_checkpoint()
                end_time = time.time()
                break

            else:
                root_index = s1
                env_reset = False

        # train network at end of each episode
        if agent._experience.size > hypes['nn']['batch_size']:
            loss, grads_norm, grads_norm_clipped, grads_sum = agent.train_network(graph, tf_sess)
	    ep_data['loss_d1'] = loss
	    ep_data['grads_norm_d1'] = grads_norm
        if agent._experience_new.size > hypes['nn']['batch_size']:
            loss, grads_norm, grads_norm_clipped, grads_sum = agent.train_network(graph, tf_sess,'recent')
            ep_data['loss_d2'] = loss
            ep_data['grads_norm_d2'] = grads_norm
        nn_time = time.time()


        # store information for next episode
        ep_data['actions'] = actions
        ep_data['niterations'] = niterations
        ep_data['currvars'] = currvars
        ep_data['avgvars'] = avgvars
        ep_data['rewards_sum'] = R
        ep_data['states'] = states
        ep_data['mus'] = mus
        ep_data['sigmas'] = sigmas
        ep_data['etas'] = etas
        ep_data['ucb_params'] = ucbs
        ep_data['steps_count'] = t
        ep_data['wallclock'] = end_time - start_time
        ep_data['wallclock_nn'] = nn_time - start_time
        compare_data['pi_NN'] = pi_network
        compare_data['network_decisions'] = decision_types
        compare_data['pi_MCTS'] = pi_tree
        compare_data['root_indices'] = root_states
        compare_data['states'] = states
        compare_data['child_priors'] = child_priors

        compare_data['picked_action'] = actions

        results_compare[ep] = compare_data
        results[ep] = ep_data
        results_mcts[ep] = mcts_data 

        ep += 1
        print("episode: " +str(ep) + ", return: "  +str(R))

        if ep%5 == 0:
            #save training data
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_main.pkl'), 'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_mcts.pkl'), 'wb') as f:
                pickle.dump(results_mcts, f)
            with open(os.path.join(outdir, str(output) + '_' + str(ep) + '_compare.pkl'), 'wb') as f:
                pickle.dump(results_compare, f)
            results_mcts = {}
            results = {}
        #save network weights
        session_saver(tf_sess, hypes, ckpdir)

    return results, results_mcts

def do_training(
    hyperparams,  # dict,dict of hyeperparameters for experiment
    modules,      # dict,optimization, tree_search, model
    env,          # RL environment gym
    outdir,       # string,output directory path for results
    ckpdir,       # string,directory path for tf saver checkpoints
    treedir,      # string, directory path for tree information built in MCTS  
    output,       # string,base name of result files
    continue_training
    ):
    """
    creates tensorflow session, graph, runs and saves experiment results
    """
    print("Starting training, calling session")
    print("Saving results to " +str(outdir))
    with tf.Session() as sess:
        graph  = build_training_graph(hyperparams, modules)
        tf_sess = start_session(hyperparams,outdir)
        if continue_training:
            if check_int(continue_training):
                load_weights_v2(ckpdir,tf_sess,continue_training)
                print("loaded weights from " +str(continue_training))
            else:
                load_weights(ckpdir,tf_sess)
            new_dir = os.path.join(outdir,'new_results')
            try: new_dir = os.makedirs(new_dir)
            except: OSError
            outdir =  new_dir
            # modules = load_modules(hyperparams, env)
        # hyperparams['solver']['global_step'] = graph['global_step'] 
        summary = tf.Summary()
        loop_start = time.time()
        print(hyperparams)
        results, results_mcts  = run_training(hyperparams, modules, graph, tf_sess, outdir, ckpdir,treedir,output)
        print("ended in " + str(time.time() - loop_start))
        with open(os.path.join(outdir, str(output)+'_main.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(outdir, str(output) + '_mcts.pkl'), 'wb') as f:
            pickle.dump(results_mcts, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hyperparameters',help = 'path to configuration json file')
    parser.add_argument(
        '--environment',help = 'string specifying RL environment')
    parser.add_argument(
        '--output',help = 'results pkl name')
    parser.add_argument(
        '--outdir', help = 'put results to  folder',default = True)
    parser.add_argument(
        '--continue_training', help = 'put results to  folder',default = False)       
    args = parser.parse_args()
    return args

def main(arg):
    # parse script arguments
    args = parse_args()
    continue_training = args.continue_training
    hyperparam_path = args.hyperparameters
    file_name = args.output

    # load hyperparameter configuration from json file
    script_path = (os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(script_path,hyperparam_path), 'r') as f:
        hyperparams = json.load(f)
    hyperparams['base_path'] = script_path
    env = args.environment
    
    # create directories for results
    if args.outdir: 
        outdir = args.outdir
    else:
        outdir = os.path.dirname(hyperparam_path)
    # create dir for checkpoint files
    outdir = os.path.join(outdir,file_name) 
    if not os.path.exists(outdir): os.makedirs(outdir)
    ckpdir = os.path.join(outdir,'checkpoints')
    treedir = os.path.join(outdir,'trees')
    print("saving checkpoints to " + str(ckpdir))
    if not os.path.exists(ckpdir):
        os.makedirs(ckpdir)
    if not os.path.exists(treedir): os.makedirs(treedir)

    if continue_training:
            base_dir = '/'.join(outdir.rsplit('/')[:-1])
            name = outdir.rsplit('/')[-1]
            hyperparam_path = os.path.join(base_dir, str(name) + '.json')
            with open(os.path.join(outdir,hyperparam_path), 'r') as f:
                hyperparams = json.load(f)               
            hyperparams['base_path'] = (os.path.dirname(os.path.realpath(__file__)))        
    # load agent, optimizer, tree search modules, create gym environment
    modules = load_modules(hyperparams, env)
    do_training(hyperparams,modules,env,outdir,ckpdir,treedir,file_name,continue_training)

if __name__ == '__main__':
    sys.exit(main(sys.argv))


