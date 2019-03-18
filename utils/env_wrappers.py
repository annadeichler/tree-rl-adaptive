# -*- coding: utf-8 -*-
"""

openai and pybullet gym environment wrappers

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
#from .box2d_utils import *
from copy import deepcopy
from operator import add
import itertools
import pybullet as pb
from .utils import *
from .utils_tf import *


class DiscreteActionWrapper(gym.ActionWrapper):
	# contiunous action discretizer
	def __init__(self,env,steps):
		super(DiscreteActionWrapper, self).__init__(env)
		self._actions = []
		self.env = env.env
		space = self.env.action_space
		bins = ({i: np.linspace(space.low[i],space.high[i],steps) for i in range(space.shape[0])})
		self._actions = [np.array(l) for l in list(itertools.product(*bins.values()))]
		self.action_space = gym.spaces.Discrete(len(self._actions))

	def action(self,a):
 		return self._actions[a].copy()

class CartPoleWrapper(gym.Wrapper):
	"""
		openai gym EnvWrapper for CartPole-v1 environment, normalized rewards
	"""
	def __init__(self, env, name, env_seed):
		super(CartPoleWrapper, self).__init__(env)

		self.name = name
		self.env_name = self.env.env.__class__.__name__
		self.env_seed = env_seed
		self.env = env.env
		self.seed(self.env_seed)

	def seed(self,seed):
		return self.env.seed(self.env_seed)

	def step(self, action):
		s, r, terminal, _ = self.env.step(action)
		if self.name == 'CartPole-vr':
			if terminal:
				r = -1
			else:
				r = 0.005
		if self.name == 'CartPole-vn':
			r = r/200.0
			
		return s, r, terminal, _

	def reset(self):
		return self.env.reset()

	def render(self):
		return self.env.render()

	def close(self):										
		return self.env.close()

class OpenAIWrapper(gym.Wrapper):
	"""
		openai gym EnvWrapper, normalized rewards
	"""
	def __init__(self, env, name, env_seed):
		super(OpenAIWrapper, self).__init__(env)

		self.name = name
		self.env_name = self.env.env.__class__.__name__
		self.env_seed = env_seed
		self.env = env.env
		self.seed(self.env_seed)

	def seed(self,seed):
		return self.env.seed(self.env_seed)

	def step(self, action):
		s, r, terminal, _ = self.env.step(action)		
		return s, r, terminal, _

	def reset(self):
		return self.env.reset()

	def render(self):
		return self.env.render()

	def close(self):										
		return self.env.close()

class RaceCarEnvWrapper(gym.Wrapper):
	"""
		openai gym EnvWrapper for CartPole-v1 environment, normalized rewards
	"""
	def __init__(self, env, name):
		super(RaceCarEnvWrapper, self).__init__(env)
		# d_env = {key: getattr(env,key) for key in env.__dict__.keys()}    
		# _=[setattr(self,key,d_env[key]) for key in d_env.keys()]
		# self.env = env

		self.name = name
		self.env_name = self.env.env.__class__.__name__
		# self.env_seed = env_seed
		# remove time limit wrapper
		self.env = env.env
		# self._p=self.env._p
		self.checkpoint_counter = None
		self.checkpoint = None
		# self._envStepCounter = self.env._envStepCounter
		# self.seed()

		print("making new  " + str(self.env_name))

	def seed(self,seed):
		return self.env._seed()

	def step(self, action):
	  s, r, terminal, _ = self.env.step(action)	
	  
	  return s, r, terminal, _

	def reset(self):
	  return self.env.reset()

	def render(self):
	  return self.env.render()

	 # option to return rendered environment image
	 # option to return rendered environment image
	def save_render(self, mode="rgb_array", close=False):
		print("opoWTF")
		print(mode)
		mode ="rgb_array"
		if mode != "rgb_array":
		    return np.array([])
		base_pos = [0,0,0]
		_cam_dist = 10  #.3
		_cam_yaw = 50
		_cam_pitch = -35
		_render_width=480
		_render_height=480

		view_matrix = pb.computeViewMatrixFromYawPitchRoll(
		    cameraTargetPosition=base_pos,
		    distance=_cam_dist,
		    yaw=_cam_yaw,
		    pitch=_cam_pitch,
		    roll=0,
		    upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(
		    fov=90, aspect=float(_render_width)/_render_height,
		    nearVal=0.01, farVal=100.0)
		#proj_matrix=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
		(_, _, px, _, _) = pb.getCameraImage(
		    width=_render_width, height=_render_height, viewMatrix=view_matrix,
		    projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
	  return self.env.close()

	def save_checkpoint(self):
		# save current environment state
		self.checkpoint = self.env._p.saveState()
		self.checkpoint_counter = deepcopy(self.env._envStepCounter)

	def restore_checkpoint(self):	
		# restored saved environment state
		self.env._p.restoreState(self.checkpoint)
		self.env._envStepCounter = self.checkpoint_counter

class ReacherEnvWrapper(gym.Wrapper):
	"""
		openai gym EnvWrapper for CartPole-v1 environment, normalized rewards
	"""
	def __init__(self, env, name):
		super(ReacherEnvWrapper, self).__init__(env)
		self.name = name
		self.env_name = self.env.env.__class__.__name__
		# self.env = env.env
		self.checkpoint = None
		print("making new  " + str(self.env_name))

	def seed(self,seed):
		return self.env._seed()

	def step(self, action):
	  s, r, terminal, _ = self.env.step(action)	
	  return s, r, terminal, _

	def reset(self):
	  return self.env.reset()

	def render(self):
	  return self.env.render()

	def save_render(self, mode="rgb_array", close=False):
		# save rendered image of environment
		print("opoWTF")
		print(mode)
		mode ="rgb_array"
		if mode != "rgb_array":
		    return np.array([])
		base_pos = [0,0,0]
		_cam_dist = 10  #.3
		_cam_yaw = 50
		_cam_pitch = -35
		_render_width=480
		_render_height=480

		view_matrix = pb.computeViewMatrixFromYawPitchRoll(
		    cameraTargetPosition=base_pos,
		    distance=_cam_dist,
		    yaw=_cam_yaw,
		    pitch=_cam_pitch,
		    roll=0,
		    upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(
		    fov=90, aspect=float(_render_width)/_render_height,
		    nearVal=0.01, farVal=100.0)
		#proj_matrix=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
		(_, _, px, _, _) = pb.getCameraImage(
		    width=_render_width, height=_render_height, viewMatrix=view_matrix,
		    projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
	  return self.env.close()

	def save_checkpoint(self):
		self.checkpoint = self.env.env._p.saveState()

	def restore_checkpoint(self):	
		self.env.env._p.restoreState(self.checkpoint)


class KukaEnvWrapper(gym.Wrapper):
	"""
		pybullet gym EnvWrapper for Kuka environment
	"""
	def __init__(self, env, name):
		super(KukaEnvWrapper, self).__init__(env)
		# d_env = {key: getattr(env,key) for key in env.__dict__.keys()}    
		# _=[setattr(self,key,d_env[key]) for key in d_env.keys()]
		# self.env = env

		self.name = name
		self.env_name = self.env.env.__class__.__name__
		# self.env_seed = env_seed
		# remove time limit wrapper
		self.env = env.env
		# self._p=self.env._p
		self.checkpoint_counter = None
		self.checkpoint = None
		# self._envStepCounter = self.env._envStepCounter
		# self.seed()

		print("making new  " + str(self.env_name))

	def seed(self,seed):
		return self.env._seed()

	def step(self, action):
	  s, r, terminal, _ = self.env.step(action)	
	  
	  return s, r, terminal, _

	def reset(self):
	  return self.env.reset()

	def render(self):
	  return self.env.render()

	 # option to return rendered environment image
	def save_render(self, mode="rgb_array", close=False):
		print("opoWTF")
		print(mode)
		mode ="rgb_array"
		if mode != "rgb_array":
		    return np.array([])
		base_pos = [0,0,0]
		_cam_dist = 10  #.3
		_cam_yaw = 50
		_cam_pitch = -35
		_render_width=480
		_render_height=480

		view_matrix = pb.computeViewMatrixFromYawPitchRoll(
		    cameraTargetPosition=base_pos,
		    distance=_cam_dist,
		    yaw=_cam_yaw,
		    pitch=_cam_pitch,
		    roll=0,
		    upAxisIndex=2)
		proj_matrix = pb.computeProjectionMatrixFOV(
		    fov=90, aspect=float(_render_width)/_render_height,
		    nearVal=0.01, farVal=100.0)
		#proj_matrix=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
		(_, _, px, _, _) = pb.getCameraImage(
		    width=_render_width, height=_render_height, viewMatrix=view_matrix,
		    projectionMatrix=proj_matrix, renderer=pb.ER_TINY_RENDERER) #ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (_render_height, _render_width, 4))
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def close(self):
	  return self.env.close()

	def save_checkpoint(self):
		self.checkpoint = self.env._p.saveState()
		self.checkpoint_counter = deepcopy(self.env._envStepCounter)

	def restore_checkpoint(self):	
		self.env._p.restoreState(self.checkpoint)
		self.env._envStepCounter = self.checkpoint_counter

class MountainCarWrapper(gym.Wrapper):
	"""
		openai gym EnvWrapper for CartPole-v1 environment, normalized rewards
	"""
	def __init__(self, env, name, env_seed):
		super(MountainCarWrapper, self).__init__(env)
		# d_env = {key: getattr(env,key) for key in env.__dict__.keys()}	
		# _=[setattr(self,key,d_env[key]) for key in d_env.keys()]
		# self.env = env

		self.name = name
		self.env_name = self.env.env.__class__.__name__
		self.env_seed = env_seed
		self.env = env.env
		self.seed(self.env_seed)
		print("making new  " + str(self.env_name))

	def seed(self,seed):
		return self.env.seed(seed)

	def step(self, action):
		s, r, terminal, _ = self.env.step(action)
		if self.name == 'MountainCar-vr':
			if terminal:
				r = 1
			else:
				r = -0.005
		# if self.name == 'MountainCar-vn':
		# 	r = r/200.0
			
		return s, r, terminal, _

	def reset(self):
		return self.env.reset()

	def render(self):
		return self.env.render()

	def close(self):
		return self.env.close()
