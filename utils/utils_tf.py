# -*- coding: utf-8 -*-
"""

helper functions for gym,tensorflow

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import random
import tensorflow as tf
import gym
import sys,os,glob
import imp
import pybullet_envs
from .env_wrappers import *
import tensorflow.contrib.slim as slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def build_training_graph(
	hyperparams, 		# dict of hyperparameters
	modules):			# dict of modules in algorithm
	"""
	helper function for building tensorflow graph and loading modules
	"""

	network = modules['agent']
	optimizer = modules['optimizer']
	Env = modules['environment']
	scope_main = 'agent'

	agent = network.agent(Env, hyperparams, scope=scope_main )
	with tf.name_scope('inputs'):   
		input_data = {k: tf.placeholder(shape=v['shape'],dtype=v['dtype'],name=k) for k,v in agent._input_dict.iteritems()}
	agent.build_model(input_data)

	if hyperparams['use_target_network'] == True:
		target_network = network.agent(Env, hyperparams, scope='target_network')
		with tf.name_scope('target_inputs'):    
			input_data_target = {k: tf.placeholder(shape=v['shape'],dtype=v['dtype'],name=k) for k,v in target_network._input_dict.iteritems()}
		target_network.build_model(input_data_target)

	else: 
		target_network = None
		input_data_target = None
		target = None

	with tf.name_scope("loss"):
		loss = agent._loss()
		tf.summary.scalar('loss', loss)

	with tf.name_scope("optimizer"):
		global_step = tf.Variable(0, trainable=False)
		train_op, grads_norm, grads_norm_clipped, grads_sum_op = optimizer.train(hyperparams, loss, global_step, scope_main)

	summary_op = tf.summary.merge_all()

	graph = {}
	graph['loss'] = loss
	graph['loss_pi'] = agent.pi_loss
	graph['inputs'] = input_data
	# graph['inputs_target'] = input_data_target
	graph['agent'] = agent
	graph['target'] = target_network
	graph['train_op'] = train_op
	graph['grads_norm'] = grads_norm
	graph['grads_norm_clipped'] = grads_norm_clipped
	graph['grads_sum'] = grads_sum_op
	graph['global_step'] = global_step
	graph['summary_op'] = summary_op

	return graph


def start_session(
	hyperparams,		# dict of hyperparameters
	ckpdir):			# str, checkpoint directory
	"""
	starts tensofrlow session
	"""

	tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
	tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)

	for variable in slim.get_model_variables():
	    tf.summary.histogram(variable.op.name, variable)

	summary_op = tf.summary.merge_all()

	kc = 10000.0
	saver = tf.train.Saver(max_to_keep=5000)


	sess = tf.get_default_session()
	sess.run(tf.global_variables_initializer())

	# summary_writer = tf.summary.FileWriter(ckpdir, sess.graph)
	tf_sess = {}
	tf_sess['sess'] = sess
	tf_sess['summary_op'] = summary_op
	# tf_sess['writer'] = summary_writer
	tf_sess['saver'] = saver

	return tf_sess

def create_environment(	
	env_str, 			# str indicating which environment to use
	hypes):				# sdict of hyperparameters
	"""
	creates openai gym environment based on script argument
	"""
	print("creating environment")
	env_seed = hypes['env_seed']
	if env_str == 'hypes':
		env_str = hypes['env_str']
		print("creating  " + hypes['env_str'])
	if 'CartPole' in env_str:
		return CartPoleWrapper(gym.make('CartPole-v0'),env_str, env_seed)
		# return gym.make('CartPole-v0')
	elif 'LunarLander' in env_str:
		# return  LunarLanderWrapper(gym.make('LunarLander-v2'),env_str,env_seed)
		return LunarLanderWrapper(gym.make('LunarLander-v2'),'LunarLander-vr',env_seed)
	elif 'MountainCar' in env_str:
		return MountainCarWrapper(gym.make('MountainCar-v0'),env_str, env_seed)
	# assuming discrete action version is registered in pybullet envs __init__ file
	elif 'Taxi' in env_str:
		return OpenAIWrapper(gym.make('Taxi-v2'),env_str,env_seed)
	elif 'Acrobot' in env_str:
		return OpenAIWrapper(gym.make('Acrobot-v1'),env_str,env_seed)
	elif 'Racecar-v0' in env_str:
		return RaceCarEnvWrapper(gym.make('RacecarBulletDiscreteEnv-v0'),env_str)
	elif 'Kuka-v0' in env_str:
		return KukaEnvWrapper(gym.make('KukaDiscreteEnv-v0'),env_str)
	elif 'Reacher-v0' in env_str:
		return ReacherEnvWrapper(DiscreteActionWrapper(gym.make('ReacherBulletEnv-v0'),5),env_str)		

def get_env(
	d,					# multiprocessing dictionary
	modules,			# dict of modules
	hypes):				# dict of hyperparameters
	"""
	environment from multiprocessing dict for unpickable environments
	"""
    if 'LunarLander' in type(modules['environment']).__name__:
        env = LunarLanderWrapper(gym.make('LunarLander-v2'),modules['environment'].name,hypes['env_seed'])
        d_env, d_envenv, np_rand = d['env']
        [setattr(env.env,key,d_env[key]) for key in d_env.keys()]
        [setattr(env.env.env,key,d_envenv[key]) for key in d_envenv.keys()]
        env.reset()
        return env
    else:
        return d['env']

def load_modules(
	hypes, 				# dict of hyperparameters of experiment
	env_name):			# str indicating which environment to use
	"""
	load algorithm modules / optimizer, tree search, neural network, environment
	"""
	modules = {}
	print(hypes['base_path'])
	print(hypes['model']['network_file'])
	net_source = os.path.join(hypes['base_path'],hypes['model']['network_file'])
	try:
		modules['agent'] = imp.load_source('network', net_source)
	except IOError: print("file \n " + str(net_source) + " does not exist")
	opt_source = os.path.join(hypes['base_path'],hypes['model']['optimizer_file'])
	modules['optimizer'] = imp.load_source('optimizer', opt_source)
	tree_source = os.path.join(hypes['base_path'],hypes['model']['tree_file'])
	modules['tree_search'] = imp.load_source('MCTS', tree_source)
	modules['environment'] = create_environment(env_name, hypes)
	print("modules created")
	return modules

def load_weights(
	checkpoint_dir,		# str, checkpoint directory path
	tf_sess):			# tensorflow session
    """
    load the weights of a model stored in saver, returns training step of checkpoint
    """
    sess = tf_sess['sess']
    saver = tf_sess['saver']
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # logging.info(ckpt.model_checkpoint_path)
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        print(checkpoint_path)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])

    # CHECKPOINT_NAME = path to save file
    # restored_vars  = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_NAME)
    # tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
    # loader = tf.train.Saver(tensors_to_load)
    # loader.restore(sess, CHECKPOINT_NAME)


def load_weights_v2(
	ckpdir,				# str, checkpoint directory path
	tf_sess,			# tensorflow session
	ckp_key):			# int id of checkpoint
	"""
	load specified weights of model stored in saver
	"""
	sess = tf_sess['sess']
	saver = tf_sess['saver']
	ckp_name = 'model.ckpt-'+str(ckp_key)
	checkpoint_path = os.path.join(ckpdir, ckp_name) 
	print(checkpoint_path)
	saver.restore(sess, checkpoint_path)
	return int(ckp_key)

def session_saver(
	tf_sess,			# tensorflow sessin
	hypes,				# dict of hyperparameters
	ckpdir):			# str, checkpoint directory path
    sess = tf_sess['sess']
    saver = tf_sess['saver']
    #[os.remove(filename) for filename in glob.glob(str(ckpdir) + '/model*')]
    checkpoint_path = os.path.join(ckpdir,
                                           'model.ckpt')
    tf_sess['saver'].save(sess, checkpoint_path, hypes['solver']['global_step'])
