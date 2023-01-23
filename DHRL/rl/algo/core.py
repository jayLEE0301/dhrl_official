import numpy as np
import torch
import datetime
import os
import os.path as osp
import sys

from rl.utils.run_utils import Timer, log_config, merge_configs
from rl.replay.core import sample_her_transitions

import time
class BaseAlgo:
    def __init__(
            self,
            env, env_params, args,
            low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
            low_reward_func, high_reward_func,
            name='algo',
    ):
        self.env = env
        self.env_params = env_params
        self.args = args
        
        self.low_agent = low_agent
        self.high_agent = high_agent
        self.low_replay = low_replay
        self.high_replay = high_replay
        self.monitor = monitor
        self.low_learner = low_learner
        self.high_learner = high_learner

        
        self.low_reward_func = low_reward_func
        self.high_reward_func = high_reward_func
        
        self.timer = Timer()
        self.start_time = self.timer.current_time
        self.total_timesteps = 0
        
        self.env_steps = 0
        self.low_opt_steps = 0
        self.high_opt_steps = 0
        self.num_envs = 1
        self.curr_subgoal = None
        self.curr_highpolicy_obs = None
        self.way_to_subgoal = 0
        self.subgoal_freq = args.subgoal_freq
        self.subgoal_scale = np.array(args.subgoal_scale)
        self.subgoal_offset = np.array(args.subgoal_offset)

        if hasattr(self.env, 'num_envs'):
            self.num_envs = getattr(self.env, 'num_envs')
        
        self._save_file = str(name) + '.pt'
        
        if len(args.resume_ckpt) > 0:
            resume_path = osp.join(
                osp.join(self.args.save_dir, self.args.env_name),
                osp.join(args.resume_ckpt, 'state'))
            self.load_all(resume_path)
        
        self.log_path = osp.join(osp.join(self.args.save_dir, self.args.env_name), args.ckpt_name)
        self.model_path = osp.join(self.log_path, 'state')
        os.makedirs(self.model_path, exist_ok=True)
        self.monitor.set_tb(self.log_path)
        config_list = [env_params.copy(), args.__dict__.copy()]
        log_config(config=merge_configs(config_list), output_dir=self.log_path)
    
    def run_eval(self, use_test_env=False, render=False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            ag_origin = ag.copy()
            #print("final goal:", bg)
            for timestep in range(env._max_episode_steps):
                act = self.eval_get_actions(ob, bg)
                if render:
                    env.render(subgoal=self.curr_subgoal)
                observation, _, _, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
            if self.num_envs > 1:
                for per_env_info in info:
                    total_trial_count += 1
                    if per_env_info['is_success'] == 1.0:
                        total_success_count += 1
            else:
                total_trial_count += 1
                if info['is_success'] == 1.0:
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        return success_rate

    def eval_get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False):
        #get subgoal
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)) :
            self.curr_highpolicy_obs == ob
            sub_goal = self.high_agent.get_actions(ob, bg)
            sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq

        act = self.low_agent.get_actions(ob, self.curr_subgoal)
        self.way_to_subgoal -= 1
        return act

    def state_dict(self):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def save(self, path):
        state_dict = self.state_dict()
        save_path = osp.join(path, self._save_file)
        torch.save(state_dict, save_path)
    
    def load(self, path):
        load_path = osp.join(path, self._save_file)
        try:
            state_dict = torch.load(load_path)
        except RuntimeError:
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
    
    def save_all(self, path):
        self.save(path)
        self.low_agent.save(path)
        self.low_replay.save(path)
        self.low_learner.save(path)
        self.high_agent.save(path)
        self.high_replay.save(path)
        self.high_learner.save(path)
    
    def load_all(self, path):
        self.load(path)
        self.low_agent.load(path)
        self.low_replay.load(path)
        self.low_learner.load(path)
        self.high_agent.load(path)
        self.high_replay.load(path)
        self.high_learner.load(path)

