# -*- coding: utf-8 -*-
"""

implementation of optimizer module

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle

def train(hyperparams, loss, global_step, scope):
	""" Set up training operation 
		create optimizer, that calculates gradients of neural network
		inputs
			loss: defined in neural network class
			hyperparams: hyperparameters containing learning rate
	"""
	lr = hyperparams['solver']['learning_rate']
	hyperparams['solver']['global_step'] = global_step

	solver = hyperparams['solver']
	with tf.name_scope('training'):
		if solver['optimizer'] == 'RMS':
			optimizer = tf.train.RMSPropOptimizer(learning_rate = lr, decay = 0.9, epsilon = solver['epsilon'])

		elif solver['optimizer'] == 'Adam':
			optimizer = tf.train.AdamOptimizer(learning_rate = lr, epsilon = solver['epsilon'])

		elif solver['optimizer'] == 'SGD':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
		else:
			raise ValueError("optimizer")

			
		gradients_variables = optimizer.compute_gradients(loss)
		
		#select main network, exclude target network variables (used in DQN network
		grad_vars_selected = [(g,v) for g,v, in gradients_variables if v.name.startswith(scope)]
		grads, tvars = zip(*grad_vars_selected)
		grads_norm = [tf.norm(item) for item in grads]
		grads_norm_clipped = []

		if hyperparams['solver']['clip_norm'] > 0.0:
			clip_norm = hyperparams['solver']['clip_norm']
			clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)
			grad_vars_selected = zip(clipped_grads, tvars)
			grads = clipped_grads
			grads_norm_clipped = [tf.norm(item) for item in clipped_grads]
			grads_norm = grads_norm_clipped
		grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in grad_vars_selected])

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.apply_gradients(grad_vars_selected,
										   global_step=global_step)


	return train_op, grads_norm, grads_norm_clipped, grad_summ_op

