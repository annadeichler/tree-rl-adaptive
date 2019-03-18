#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''

runs episodic RL training with given hyperparameters for specified pybullet/openai environment
hyperparameters are assumed to be in json file
experiments can be in parallel in thread_count number of threads

example:
python loop_hyper_iters.py --environment 'hypes' --base_json hypes/alphago_racecar.json --loop_json hypes/loop_config_racecar.json --train_file /home/asus/code/adaptive_tree_rl/train_pybullet_2.py --outdir /home/asus/ --thread_count 3

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import sys
import os
import shutil
import subprocess
import json
import pickle
import itertools
import numpy as np
import pandas as pd
import math
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

from utils.utils import *

sys.setrecursionlimit(10000)

def get_combinations(
	jsonLoop,		# str, path to json file defining parameter values to loop ove
	jsonBase):		# str, path to json file defining rest of hyperparamers
	"""
	returns pandas dataframe of hyperparameter combinations to run
	"""
	data = json.load(open(jsonLoop))
	df = pd.DataFrame(list(itertools.product(*[data[key]['values'] 
								for key in data.keys()])),columns = data.keys())
	return df, data
		
def create_json_files(
	params, 		# list of hyperparameters
	df, 			# DataFrame, looped hyperparameter settings
	jsonBase,		# str, path to json of all hyperparameters
	base_dir):		# str, path to json configuration
	"""
	create json files with hyperparemeters in output directory, returns output names 
	"""
	jsonFiles = []
	data = json.load(open(jsonBase))
	# print(df)
	for i,row in df.iterrows():
		_ = copy.deepcopy(data)
		#pid = mp.current_process()._identity[0]
		#randst = np.random.mtrand.RandomState(pid)
		#rand_int = randst.randint(0,100, size=1)[0]
		_['env_seed'] = random.randint(0,100)
		_['numpy_seed'] = random.randint(0,100)
		_['random_seed'] = random.randint(0,100)
		for param in params:
			keys = get_keys(data, param)
			if (type(keys)==list): setInDict(_, keys, row[param])
			else: _[param] = row[param]
		outFile = os.path.join(base_dir , str(row['fnames'])+'.json')
		with open(outFile, 'wa') as outfile:
			json.dump(_, outfile, indent=4)
			outfile.close()
		jsonFiles.append(outFile)
	return jsonFiles

def create_dirs(df, 
	outdir,			# str, path to output directory
	n_runs):		# int, number of threads
	"""
	create results dir
	"""
	dirs,fnames = [],[]
	try: dirId = (np.max(([int(item) for item in os.walk(outdir).next()[1]]))+1)
	except ValueError: 	dirId = 1
	sDir = os.path.join(outdir,str(dirId))
	[os.makedirs(os.path.join(sDir, str(i+1))) for i in range(n_runs)]
	for index,row in df.iterrows():
		fName = "_".join(["_".join([key,str(row[key])]) for key in df.keys()])
		fnames.append(fName)
	df['fnames'] = fnames
	return df, sDir

def create_loop_jsons(_dlin, _base_dir):
	""" 
		create json file describing loop settings in output directory
	"""
	with open(os.path.join(_base_dir ,'loop_config.json'), 'wa') as outfile:
		json.dump(_dlin, outfile,indent=4)
		outfile.close()

def run_experiments(
	entry,			# cintains training file, jsonFiles, env string
	 **kwargs):
	"""
	call training on all hyperparamer settings sequentially

	"""
	base_json = args.base_json
	env = args.environment
	train_file = args.train_file
	json_path = entry['dir']
	jsonFiles = create_json_files(entry['params'], entry['df'], base_json,json_path)
	for item in jsonFiles:
		file_name = os.path.splitext(os.path.basename(item))[0]
		print(file_name)
		subprocess.call(['python', train_file, '--hyperparameters=' + item,'--environment=' + env,
							'--output=' + file_name, '--outdir=' + json_path])

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--environment',help = 'environment')
	parser.add_argument(
		'--outdir',help = 'results output directory')
	parser.add_argument(
		'--base_json',help = 'base json file to be modified')
	parser.add_argument(
		'--loop_json',help = 'grid of hyperparameters')
	parser.add_argument(
		'--train_file',help = 'path to training file')
	parser.add_argument(
		'--thread_count', help = 'number of iterations per setting')

	args = parser.parse_args()  
	return args

def main(**kwargs):
	thread_count = int(args.thread_count)
	pool = Pool(int(thread_count))
	base_json = args.base_json
	loop_json = args.loop_json
	outdir = args.outdir
	train_file = args.train_file
	env = args.environment
	# get hyperparameter combinations specified in loop json files
	comb, comb_loop_data = get_combinations(args.loop_json, args.base_json)
	# looped hyperparameters
	params = comb.keys()
	df, base_dir = create_dirs(comb,outdir,thread_count)
	create_loop_jsons(comb_loop_data,base_dir)
	# copy hyperparameter loop configuration json file to output location
	shutil.copyfile(loop_json, os.path.join(base_dir, 'loop_config.json'))
	loop_data = [{'params':params, 'df':df, 'dir':os.path.join(base_dir, str(i+1))} for  i in range(thread_count)]
	# run experiments parallel
	pool.map(partial(run_experiments, **kwargs), loop_data)

if __name__ == '__main__':
	args = parse_args()
	kwargs = vars(args)
	main(**kwargs)