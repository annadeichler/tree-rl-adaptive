# -*- coding: utf-8 -*-
"""

implementation of MCTS module in the AlphaZero algorithm for pybullet emulators
optional root return variance based adaptive MCTS
	increased number of MCTS iterations if variance is high
	policy temperature based on root return variance
based on  @tmoer

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import random
from PIL import Image
import time
import pickle
import os
from utils.utils import *
	 
def MCTS(
	model, 			# AlphaZero neural network
	root_index, 	# np vector, current state index
	root,			# State object, current state
	env,			# openai environment emulator
	sess,			# tensorflow session
	tree_hypes, 	# dict, tree search hyperparameters 
	ep, 			# int, current episode
	t_all,			# int, all steps
	t_current, 		# int, episode steps
	outdir):		# str, result output directory
	''' Monte Carlo Tree Search function '''
	# read tree hyperparameters from dictionary
	N = tree_hypes['n_mcts']
	pure_mcts = tree_hypes['pure_mcts']
	bootstrap = tree_hypes['bootstrap']
	gamma = tree_hypes['gamma']
	c = model._ucb_param
	use_expl_bias = tree_hypes['use_expl_bias']
	use_prior = tree_hypes['use_prior']
	init_value = tree_hypes['action_init_value']
	# make sure environment restored to save state in pybullet emulator
	env.restore_checkpoint()	
	if root is None:
		root = State(root_index,r=0.0,terminal=False,parent_action=None,na=env.action_space.n,model=model,sess=sess) # initialize the root node
		root.evaluate(model,env,sess,bootstrap,gamma)
		root.add_child_actions(init_value)
	else:
		root.parent_action = None # continue from current root

	# perform N MCTS iterations
	for i in range(int(N)):  
		MCTS_step(Env,root,sess,model,c,g,use_prior,use_expl_bias,bootstrap,init_value)
	env.restore_checkpoint()

	# adaptive variance-based MCTS, increas MCTS iterations if variance high	
	if tree_hypes['adaptive'] == True:
		# return variance of current tree node
		model.curr_var = np.var(root.returns[-N_base:])
		ratio = model.curr_var/model.avg_returnVar
		# additional MCTS iterations
		N_adt = max(N_min,min(math.ceil(1.5*N_min*ratio),N_max))
		model.N_adt = N_adt
		# update baseline variance estimate
		if model.avg_returnVar == 1.0:
			model.avg_returnVar = model.curr_var
		else:
			model.avg_returnVar = model.avg_returnVar * (t_all - 1)/t_all + model.curr_var/t_all
		# execute additional MCTS iterations
		if N_adt>N_min: 
			for i in range(int(N_adt)): 
				MCTS_step(Env,root,sess,model,c,g,use_prior,use_expl_bias,bootstrap,init_value)

	# get policy temperature parameter based on variance-based
	if ratio < 0.5 and ep>25:
		(model.get_policy_temp(model.curr_var/model.avg_returnVar))
	else:
		model._temp = 1.

	# save tree information for visualization
	# if ep % 20 == 0:
	# 	try: pickle.dump(root, open(os.path.join(outdir, 'root_' + str(ep)+'_'+str(t_current)+"_tree.p"), "wb" ))
	# 	except RuntimeError: print("failed pickling, recursion")            

	# restore environment state
	env.restore_checkpoint()
	# return the estimates at the root
	return root 

def MCTS_step(
	env,			# openai environment emulator
	root,			# current root node in search tree
	tf_sess,		# tensorflow session
	network,		# AlphaZero neural network
	c_ucb,			# exploration-exploitation parameter in UCB
	gamma,			# decay rate
	use_prior,		# use prior in UCB of AlphaZero
	use_expl_bias,	# use expl bias in UCB of AlphaZero
	bootstrap,		# type of leaf node evaluation, true: neural network
	init_value):	# child nodes initialization value
	''' 
	execute one MCTS iteration (select,expand,backup)
	'''
	state = root 
	env.restore_checkpoint()
	while True:            
		# select
		action = state.select(network,c_ucb,tf_sess,use_prior,use_expl_bias)
		s1,r,t,_ = env.step(action.index)
		if hasattr(action,'child_state'):
			state = action.child_state
		else:
		# expand
			state = action.add_child(s1,r,t,network,tf_sess)
			state.evaluate(network,env,tf_sess,bootstrap,gamma)
			state.add_child_actions(init_value)
			break            
		if state.terminal:
			break
	# backup
	R = np.squeeze(state.V)
	while state.parent_action is not None:
		R = state.r + gamma * R 
		action = state.parent_action
		action.update(R)
		state = action.parent_state
		state.update()
		# log root returns for root return variance based adaptation
		state.update_returns(R)

class State(object):
	''' State object '''
	__slots__ =  'index', 'r','terminal', 'parent_action', 'na', 'priors', 'child_actions', 'n', 'V', 'Qa', 'returns'
	# rewrite copy operator to save memory
	def __copy__(self):
		stats  = self.__class__.__new__(self.__class__)
		stats.index = self.index
		stats.terminal = self.terminal
		stats.n = self.n
		stats.na = [child_action.n for child_action in self.child_actions]
		stats.Qa = [child_action.Q for child_action in self.child_actions]
		stats.priors = self.priors
		stats.returns = self.returns
		return stats

	def __init__(self,index,r,terminal,parent_action,na,model,sess):
		''' Initialize a new state '''
		self.index = index # state
		self.r = r # reward upon arriving in this state
		self.terminal = terminal
		self.parent_action = parent_action
		self.na = na
		self.returns = []
		self.priors = np.squeeze(model.predict_pi(index[None,],sess))
		self.child_actions = [Action(a,parent_state=self) for a in range(na)]
		self.n = 0

	def __getstate__(self):
		return dict([(k, getattr(self,k,None)) for k in self.__slots__])

	def __setstate__(self,data):
		for k,v in data.items():
			setattr(self,k,v)

	def add_child_actions(
		self, 
		value_init):    # if 'value' initialize actionvalues to parent state value
		if value_init == 'value':
			self.child_actions = [Action(a,parent_state=self,Q_init=self.V) for a in range(self.na)]
		else:
			self.child_actions = [Action(a,parent_state=self) for a in range(self.na)]
		   
	def select(
		self,
		model,
		c,				# exploration-exploitatoin trade-off parameter
		sess,           # tensorflow session
		use_prior,      # use policy network
		use_expl_bias): # use +1 in denominator
		''' 
		select one of the child actions based on UCT rule 
		'''
		# use adaptive normalization in returns
		Q = [(child_action.Q-model.nu)/model.sigma for child_action in self.child_actions]
		if use_expl_bias:
			U = [c * (np.sqrt(self.n)/(1 + child_action.n)) for child_action in self.child_actions]
		else:
			U = [c * (np.sqrt(self.n)/(child_action.n)) if child_action.n!=0 else np.inf for child_action in self.child_actions] 
		scores = np.squeeze(np.array([Q]) + self.priors * np.array([U]))
		winners = np.argwhere(scores == np.max(scores)).flatten()        
		winner = random.choice(winners)
		return self.child_actions[winner]
		
	def return_results(
		self,
		count_type,     # 'default': only counts count - original alphago
		temp = 1,       # temperature parameter (inf one hot action selection)
		sample_size=15):#   

		counts = np.array([child_action.n for child_action in self.child_actions],dtype='float32')
		Q = np.array([child_action.Q for child_action in self.child_actions])
		if count_type == 'thompson':
			counts_thompson = thompson_sample(Q, counts, sample_size)
			probs = stable_normalizer(counts_thompson,temp)
		elif count_type == 'random_ucb_sample':
			counts_random_ucb = random_ucb_sample(Q, counts, sample_size)
			probs = stable_normalizer(counts_random_ucb, temp)       
		else: 

			probs = stable_normalizer(counts,temp)
		#probs = normalizer(counts,temp)

		V = np.sum((counts/np.sum(counts))*Q)[None]
		return probs,V
	
	def evaluate(self,model,env,sess,bootstrap,gamma):
		if self.terminal:
			self.V = 0.0
		else:
			if bootstrap:
				self.V = model.predict_V( self.index[None,], sess)     
				# set also child action to self.V      
			else:
				self.V = rollout(self.index,env,sess,policy='random',model=model,gamma=gamma)                
				
	def update(self):
		''' update count on backward pass '''
		self.n += 1
		
	def forward(self,a):
		return self.child_actions[a].child_state
	def update_returns(self,R):
		self.returns.append(R)

def thompson_sample(Q_vec,n_vec,n):
	draws = [np.argmax(np.random.normal(Q_vec,1.0/np.sqrt(n_vec))) for i in range(n)]
	counts,_ = np.histogram(draws,bins=len(Q_vec),range=[-0.5,len(Q_vec)-0.5])
	return counts 

def random_ucb_sample(Q_vec,n_vec,n):
	draws = [np.argmax(Q_vec + np.random.uniform(0.0,2.0,size=len(Q_vec)) * (np.sqrt(np.sum(n_vec))/np.array(n_vec))) for i in range(n)]
	counts,_ = np.histogram(draws,bins=len(Q_vec),range=[-0.5,len(Q_vec)-0.5])
	return counts 

class Action():
	''' Action object '''
	__slots__ =  'index', 'parent_state', 'W', 'n', 'Q', 'child_state'

	def __init__(self,index,parent_state,Q_init=0):
		self.index = index # index of action e.g. move right = 1, move left = 0
		self.parent_state = parent_state
		self.W = 0
		self.n = 0
		self.Q = Q_init
				
	def add_child(self,s1,r,terminal,model, sess):
		self.child_state = State(s1,r,terminal,self,self.parent_state.na,model,sess)
		return self.child_state
		
	def update(self,R):
		self.n += 1
		self.W += R
		self.Q = self.W/self.n
