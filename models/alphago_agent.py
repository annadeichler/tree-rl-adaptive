# -*- coding: utf-8 -*-
"""

implementation of two-head neural network module in AlphaZero two for single player deterministic environments

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import copy
import tensorflow.contrib.slim as slim
import pickle
import collections
import time
from collections import deque
from copy import deepcopy

from utils.utils import display_info, Database, check_space
from utils.utils_tf import load_weights, build_training_graph

class agent():
	""" class for two-head neural network of AlphaZero """ 
	def __init__(self, 	
		env, 			# environment emulator object
		hypes, 			# dict of hyperparameters
		scope):			# string for TF scope
		"""
		creates an instance of model
		"""
		self._scope = scope
		self._hidden_1 = hypes['nn']['hidden_1']
		if hypes['same_hidden']:
			self._hidden_2 = hypes['nn']['hidden_1']
		else:   
			self._hidden_2 = hypes['nn']['hidden_2']
		self._action_dim, self._action_discrete  = check_space(env.action_space)
		if not self._action_discrete: 
			raise ValueError('Continuous action space not implemented')
		self._state_dim, self._state_discrete  = check_space(env.observation_space)
		if not self._state_discrete:
			s_dict =  {'shape':np.append(None,self._state_dim), 'dtype':tf.float32 }
		else:
			s_dict =  {'shape':np.append(None,1),'dtype':tf.int32 }
		self._input_dict = {'state': s_dict, 
							'V':{ 'shape':[None,1],'dtype':tf.float32 },
							'pi' : { 'shape':[None,self._action_dim],'dtype':tf.float32 }}

		self._experience = Database(max_size = hypes['db_size'], 
									batch_size=hypes['nn']['batch_size'])
		self._experience_new = Database(max_size = hypes['max_ep_steps'], 
									batch_size=hypes['nn']['batch_size'])
		self.avg_returnVar = 1.
		self._temp =  hypes['tree_search']['temp']
		self._root_count_type = hypes['tree_search']['root_count_type']
		self._verbose = hypes['tree_search']['verbose_tree']
		self._n_epoch = self.get_n_epochs(hypes)
		self._ucb_params =  np.linspace( hypes['tree_search']['ucb_max'],  hypes['tree_search']['ucb_min'],  hypes['tree_search']['ucb_decay_steps'])
		self._ucb_decay = hypes['tree_search']['ucb_decay_steps']
		self._beta_POP = hypes['nn']['beta_POP']
		self.current_state = None
		self.nu = 0.
		self.eta = hypes['tree_search']['eta']
		self.sigma = 1.

	def build_model(self, 
		inputs):		# dict of TF graph inputs
		"""
		create neural network layers and activations
		"""
		self._inputs = inputs
		with tf.variable_scope(self._scope):
			# feedforward
			x = inputs['state']
			if  self._state_discrete:
				x =  tf.squeeze(tf.one_hot(x, self._state_dim, axis = 1), axis = 2)
			x = slim.fully_connected(x, self._hidden_1, activation_fn = tf.nn.elu)
			x = slim.fully_connected(x, self._hidden_2, activation_fn = tf.nn.elu)
			# output
			self.f = x
			self.log_pi_hat = slim.fully_connected(x, self._action_dim,activation_fn = None) 
			self.pi_hat = tf.nn.softmax(self.log_pi_hat)           
			self.V_hat = slim.fully_connected(x, 1, activation_fn = None,scope='V_hat')
			self.network_out = [self.pi_hat, self.V_hat]

	def predict_V(self, 
		state, 			# current state
		sess):			# TF session
		"""
		run inference on given input state, return value network output
		"""
		return sess.run(self.V_hat, feed_dict = {self._inputs['state']:state})

	def predict_pi(self, 
		state, 			# current state
		sess):			# TF session
		"""
		run inference on given input state, return policy network output
		"""
		return sess.run(self.pi_hat, feed_dict = {self._inputs['state']:state})

	def _loss(self):
		"""
		additive loss definition for neural network
		"""
		self.V_loss = 0.5 * tf.losses.mean_squared_error(labels = self._inputs['V'], 
														  predictions = self.V_hat)
		self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self._inputs['pi'],
																  logits = self.log_pi_hat)
		self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)
		return self.loss

	def update_scale_shift(self,
		beta):			# exponential decay parameter
		"""
		update return normalization parameters for MCTS
		"""
		returns  = self.current_state.returns
		returns_2 = [x**2 for x in returns]
		self.sigma_old = deepcopy(self.sigma)
		self.nu_old = deepcopy(self.nu)
		self.nu = (1-beta)*self.nu + beta*np.mean(returns)
		self.eta = (1-beta) * self.eta + beta*np.mean(returns_2)
		self.sigma = np.sqrt(self.eta - self.nu**2)

	def update_POP(self,
		sess,			# TF session
		t):				# int, time step 
		"""
		update decay parameter and call return normalization update
		"""
		beta = self._beta_POP*(1-(1-self._beta_POP)**(t+1))**(-1)
		self.update_scale_shift(beta)

	def select_action(self,
		hypes,			# dict of hyperparameters
		modules, 		# dict of modules
		env, 			# environment emulator object
		sess, 			# TF session
		root_index,		# current state
		env_reset, 		# bool, reset environments
		t, 				# int, episode time step
		t_all,			# int, training time steps 
		ep,				# int, episode count
		outdir=""):
		"""
		run MCTS tree search and select action for root current state
		"""
		if env_reset == True: self.current_state = None
		self.get_ucb_param(t_all)
		self.current_state = modules['tree_search'].MCTS(self, root_index, self.current_state,
															 env, sess, hypes['tree_search'],ep,t_all,t,outdir)
		self.update_POP(sess,t)
		priors = np.squeeze(self.predict_pi(self.current_state.index[None,],sess))
		pi,V =  self.current_state.return_results(self._root_count_type, self._temp)
		self._experience.store((self.current_state.index,V,pi))    
		self._experience_new.store((self.current_state.index,V,pi))  
		a = np.random.choice(len(pi), p = pi)
		self.current_state = self.current_state.forward(a)
		self.pi_NN = priors
		self.pi_MCTS = pi
		return a, copy.copy(self.current_state)

	def clear_history(self):
		"""
		 clear previous episode training data
		"""
		self._experience_new.clear()

	def get_ucb_param(self, 
		t):				# int, time step
		"""
		get decayed ucb parameter for MCTS
		"""
		self._ucb_param = self._ucb_params[min(t, self._ucb_decay-1)]

	def get_n_epochs(self,
		hypes):			# dict of hyperparameters
		"""
		get number of training epochs based on MCTS iterations
		"""
		mcts_min = hypes['tree_search']['n_mcts_min']
		n_mcts = float(hypes['tree_search']['n_mcts'])
		c_epoch = hypes['c_epoch']
		n_epoch = int(np.rint(((n_mcts/mcts_min-1)*c_epoch+1)))
		return n_epoch

	def train_network(self, 
		graph, 			# TF graph
		tf_sess, 		# TF session
		database = 'main'):
		"""
		train neural network
		"""
		if database == 'recent':
			db = self._experience_new
			self._experience_new.reshuffle()
			n_epoch = 1
		else: 
			db = self._experience
			self._experience.reshuffle()
			n_epoch = self._n_epoch
		sess = tf_sess['sess']
		merge = tf.summary.merge_all()
		counter = 0
		losses, grads_norms, grads_clipped_norms = [], [], []
		for epoch in range(n_epoch):
			for sb,Vb,pib in db:
				counter +=1
				batch_size = len(sb)
				feed_dict = {graph['inputs']['state']: sb,
							graph['inputs']['V']: Vb,
							graph['inputs']['pi']: pib}
				_, loss_value, grads_norm, grads_norm_clipped, grads_sum, summary = sess.run([graph['train_op'], 
																					 graph['loss'],
																					 graph['grads_norm'],
																					 graph['grads_norm_clipped'],
																					 graph['grads_sum'],
																					 merge],
																					 feed_dict = feed_dict)
				#tf_sess['writer'].add_summary(summary, counter)
				losses.append(loss_value/batch_size)
				grads_norms.append(grads_norm)
				# print(np.mean(grads_norm))
				grads_clipped_norms.append(grads_norm_clipped)
			mean_loss = np.mean(losses)
		return mean_loss, grads_norms, grads_clipped_norms, grads_sum
