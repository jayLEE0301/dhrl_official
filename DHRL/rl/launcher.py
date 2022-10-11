import gym
import random
import numpy as np
import torch
import time
from rl.utils.run_utils import Monitor
from rl.replay.planner import LowReplay, HighReplay
from rl.learn.dhrl import HighLearner, LowLearner
from rl.agent.agent import LowAgent, HighAgent
from rl.algo.dhrl import Algo
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from envs.antenv import EnvWithGoal, GatherEnv
from envs.antenv.create_maze_env import create_maze_env
from envs.antenv.create_gather_env import create_gather_env
def get_env_params(env, args):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'sub_goal': args.subgoal_dim,
              'l_action_dim': args.l_action_dim,
              'h_action_dim': args.h_action_dim,
              'action_max': args.action_max,
              'max_timesteps': args.max_steps}
    return params


def launch(args):
    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    elif args.env_name in ["AntMaze", "AntMazeSmall-v0", "AntMazeComplex-v0", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        test_env.evaluate = True
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.test_env_name)
        if args.env_name == "Reacher3D-v0":
            test_env.evaluate = True
    seed = args.seed

    env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    assert np.all(env.action_space.high == -env.action_space.low)
    env_params = get_env_params(env, args)
    low_reward_func = env.low_reward_func
    high_reward_func = env.high_reward_func
    monitor = Monitor()


    ckpt_name = args.ckpt_name
    if len(ckpt_name) == 0:
        data_time = time.ctime().split()[1:4]
        ckpt_name = data_time[0] + '-' + data_time[1]
        time_list = np.array([float(i) for i in data_time[2].split(':')], dtype=np.float32)
        for time_ in time_list:
            ckpt_name += '-' + str(int(time_))
        args.ckpt_name = ckpt_name
    
    low_agent = LowAgent(env_params, args)
    high_agent = HighAgent(env_params, args)
    low_replay = LowReplay(env_params, args, low_reward_func)
    high_replay = HighReplay(env_params, args, high_reward_func, monitor)
    low_learner = LowLearner(low_agent, monitor, args)
    high_learner = HighLearner(high_agent, monitor, args)

    
    algo = Algo(
        env=env, env_params=env_params, args=args,
        test_env=test_env,
        low_agent=low_agent, high_agent = high_agent, low_replay=low_replay, high_replay=high_replay, monitor=monitor, 
        low_learner=low_learner, high_learner=high_learner,
        low_reward_func=low_reward_func, high_reward_func=high_reward_func
    )
    return algo