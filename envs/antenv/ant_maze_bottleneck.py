import gym
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from gym import spaces
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

class AntMazeBottleneckEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    xml_filename = 'ant_maze_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)


        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15],
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    
    def rand_goal(self):
        while True:
            self.goal = np.random.uniform(low=-4., high=20., size=2)
            if not ((self.goal[0] < 12) and (self.goal[1] > 4) and (self.goal[1] < 12)):
                break


    def reset_model(self):
        self.rand_goal()
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist




class AntMazeBottleneckEvalEnv(mujoco_env.MujocoEnv, utils.EzPickle): 
    xml_filename = 'ant_maze_bottleneck.xml'
    goal = np.random.uniform(low=-4., high=20., size=2)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets', xml_filename)
    objects_nqpos = [0]
    objects_nqvel = [0]
    reward_type = 'sparse'
    distance_threshold = 0.5
    action_threshold = np.array([30., 30., 30., 30., 30., 30., 30., 30.])
    init_xy = np.array([0,0])

    def __init__(self, file_path=None, expose_all_qpos=True,
                expose_body_coms=None, expose_body_comvels=None, seed=0):
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.rng = np.random.RandomState(seed)
        self.max_step = 600
        self.nb_step = 0

        mujoco_env.MujocoEnv.__init__(self, self.mujoco_xml_full_path, 5)
        utils.EzPickle.__init__(self)
        self._check_model_parameter_dimensions()

    def _check_model_parameter_dimensions(self):
        '''overridable method'''
        assert 15 == self.model.nq, 'Number of qpos elements mismatch'
        assert 14 == self.model.nv, 'Number of qvel elements mismatch'
        assert 8 == self.model.nu, 'Number of action elements mismatch'

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model



    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        done = False
        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        dist = self.compute_reward(ob['achieved_goal'], self.goal, sparse=False)
        success = (self.goal_distance(ob['achieved_goal'], self.goal) <= 5)
        self.nb_step = 1 + self.nb_step
        #done = bool((self.nb_step>self.max_step) or success)
        info = {
            'is_success': success,
            'success': success,
            'dist': dist
        }
        return ob, reward, done, info

    def compute_reward(self, achieved_goal, goal, info = None, sparse=False):
        dist = self.goal_distance(achieved_goal, goal)
        if sparse:
            rs = (np.array(dist) > self.distance_threshold)
            return - rs.astype(np.float32)
        else:
            return - dist

    def low_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def low_dense_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=False)

    def high_reward_func(self, achieved_goal, goal, info, ob=None):
        return self.compute_reward(achieved_goal, goal, info, sparse=True)

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.flat[:15], 
            self.data.qvel.flat[:14],
        ])
        achieved_goal = obs[:2]
        return {
            'observation': obs.copy(),
            'achieved_goal': deepcopy(achieved_goal),
            'desired_goal': deepcopy(self.goal),
        }
    

    def reset_model(self):
        self.goal = np.array([0., 16.])
        self.set_goal("goal_point")
        qpos = self.init_qpos + self.rng.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.rng.randn(self.model.nv) * .1
        self.init_qpos[:2] = self.init_xy
        qpos[:2] = self.init_xy

        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.
        self.set_state(qpos, qvel)
        self.nb_step = 0

        return self._get_obs()

    def set_goal(self, name):
        body_ids = self.model.body_name2id(name)
        

        self.model.body_pos[body_ids][:2] = self.goal
        self.model.body_quat[body_ids] = [1., 0., 0., 0.]
    
        
    def goal_distance(self, achieved_goal, goal):
        if(achieved_goal.ndim == 1):
            dist = np.linalg.norm(goal - achieved_goal)
        else:
            dist = np.linalg.norm(goal - achieved_goal, axis=1)
            dist = np.expand_dims(dist, axis=1)
        return dist



