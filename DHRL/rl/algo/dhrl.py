import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rl.algo.core import BaseAlgo
from rl.algo.graph import GraphPlanner

class Algo(BaseAlgo):
    def __init__(
        self,
        env, env_params, args,
        test_env,
        low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
        low_reward_func, high_reward_func,
        name='algo',
    ):
        super().__init__(
            env, env_params, args,
            low_agent, high_agent, low_replay, high_replay, monitor, low_learner, high_learner,
            low_reward_func, high_reward_func,
            name=name,
        )
        self.test_env = test_env
        self.fps_landmarks = None
        self.curr_subgoal = None
        self.curr_high_act = None
        self.curr_highpolicy_obs = None
        self.way_to_subgoal = 0
        self.subgoal_freq = args.subgoal_freq
        self.subgoal_scale = np.array(args.subgoal_scale)
        self.subgoal_offset = np.array(args.subgoal_offset)
        self.subgoal_dim = args.subgoal_dim
        self.low_replay = low_replay
        self.high_replay = high_replay
        
        self.graphplanner = GraphPlanner(args, low_replay, low_agent, env)
        self.waypoint_subgoal = None

    def get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False, graph=False):
        #get subgoal
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)) :
            self.curr_highpolicy_obs = ob

            if random_goal:
                sub_goal = np.random.uniform(low=-1, high=1, size=self.env_params['sub_goal'])
                sub_goal = sub_goal * self.subgoal_scale + self.subgoal_offset
            else:
                sub_goal = self.high_agent.get_actions(ob, bg)
                if self.args.subgoal_noise_eps > 0.0:
                    subgoal_low_limit = self.subgoal_offset - self.subgoal_scale
                    subgoal_high_limit = self.subgoal_offset + self.subgoal_scale
                    sub_goal_noise = self.args.subgoal_noise_eps * np.random.randn(*sub_goal.shape)
                    sub_goal = sub_goal + sub_goal_noise
                    sub_goal = np.clip(sub_goal, subgoal_low_limit, subgoal_high_limit)

            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            #graph search
            if (self.graphplanner.graph is not None):
                self.graphplanner.find_path(ob, self.curr_subgoal)

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, self.curr_subgoal)[:self.subgoal_dim]

        #find low level policy action
        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=self.env_params['l_action_dim'])
        else:
            act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
            if self.args.noise_eps > 0.0:
                act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
                act = np.clip(act, -a_max, a_max)
            if self.args.random_eps > 0.0:
                a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
                mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
                if self.num_envs > 1:
                    mask = np.expand_dims(mask, -1)
                act += mask * (a_rand - act)
        self.way_to_subgoal -= 1
        return act
    
    def low_agent_optimize(self):
        self.timer.start('low_train')
        
        for n_train in range(self.args.n_batches):
            batch = self.low_replay.sample(batch_size=self.args.batch_size)
            self.low_learner.update_critic(batch, train_embed=True)
            batch_g = self.low_replay.sample_g(batch_size=self.args.batch_size)
            self.low_learner.update_critic_g(batch_g, train_embed=True)
            if self.low_opt_steps % self.args.actor_update_freq == 0:
                self.low_learner.update_actor(batch, train_embed=True)
            self.low_opt_steps += 1
            if self.low_opt_steps % self.args.target_update_freq == 0:
                self.low_learner.target_update()
        
        self.timer.end('low_train')
        self.monitor.store(LowTimePerTrainIter=self.timer.get_time('low_train') / self.args.n_batches)


    def high_agent_optimize(self):
        self.timer.start('high_train')
        
        for n_train in range(self.args.n_batches):
            batch = self.high_replay.sample(batch_size=self.args.batch_size, graphplanner = self.graphplanner)
            self.high_learner.update_critic(batch, train_embed=True)
            if self.high_opt_steps % self.args.actor_update_freq == 0:
                self.high_learner.update_actor(batch, train_embed=True)
            self.high_opt_steps += 1
            if self.high_opt_steps % self.args.target_update_freq == 0:
                self.high_learner.target_update()
        
        self.timer.end('high_train')
        self.monitor.store(HighTimePerTrainIter=self.timer.get_time('high_train') / self.args.n_batches)

    
    def collect_experience(self, random_goal= False, act_randomly=False, train_agent=True, graph=False):
        low_ob_list, low_ag_list, low_bg_list, low_a_list = [], [], [], []
        high_ob_list, high_ag_list, high_bg_list, high_a_list = [], [], [], []
        self.monitor.update_episode()
        observation = self.env.reset()
        self.curr_subgoal = None
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']
        ag_origin = ag.copy()
        a_max = self.env_params['action_max']

        if (self.graphplanner.graph is not None) and (self.args.FGS):
            FGS = self.graphplanner.check_easy_goal(ob, bg)
            if FGS is not None:
                bg = FGS
        
        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, bg, a_max=a_max, random_goal= random_goal, act_randomly=act_randomly, graph=graph)
            low_ob_list.append(ob.copy())
            low_ag_list.append(ag.copy())
            low_bg_list.append(self.waypoint_subgoal.copy())
            low_a_list.append(act.copy())
            if ((self.way_to_subgoal == 0) or (timestep == self.env_params['max_timesteps'] - 1)):
                high_ob_list.append(self.curr_highpolicy_obs.copy())
                high_ag_list.append(self.curr_highpolicy_obs[:self.args.subgoal_dim].copy())
                high_bg_list.append(bg.copy())
                high_a_list.append(self.curr_subgoal.copy())

            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            self.total_timesteps += self.num_envs
            for every_env_step in range(self.num_envs):
                self.env_steps += 1
                if train_agent:
                    self.low_agent_optimize()
                    if self.env_steps % self.args.high_optimize_freq == 0:
                        self.high_agent_optimize()
        low_ob_list.append(ob.copy())
        low_ag_list.append(ag.copy())
        high_ob_list.append(ob.copy())
        high_ag_list.append(ag.copy())
        
        low_experience = dict(ob=low_ob_list, ag=low_ag_list, bg=low_bg_list, a=low_a_list)
        high_experience = dict(ob=high_ob_list, ag=high_ag_list, bg=high_bg_list, a=high_a_list)
        low_experience = {k: np.array(v) for k, v in low_experience.items()}
        high_experience = {k: np.array(v) for k, v in high_experience.items()}
        if low_experience['ob'].ndim == 2:
            low_experience = {k: np.expand_dims(v, 0) for k, v in low_experience.items()}
        else:
            low_experience = {k: np.swapaxes(v, 0, 1) for k, v in low_experience.items()}
        if high_experience['ob'].ndim == 2:
            high_experience = {k: np.expand_dims(v, 0) for k, v in high_experience.items()}
        else:
            high_experience = {k: np.swapaxes(v, 0, 1) for k, v in high_experience.items()}
        low_reward = self.low_reward_func(ag, self.waypoint_subgoal.copy(), None)


        high_reward = self.high_reward_func(ag, bg, None, ob)
        

        self.monitor.store(LowReward=np.mean(low_reward))
        self.monitor.store(HighReward=np.mean(high_reward))
        self.monitor.store(Train_GoalDist=((bg - ag) ** 2).sum(axis=-1).mean())
        self.low_replay.store(low_experience)
        self.high_replay.store(high_experience)
    
    def run(self):
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(random_goal= True, act_randomly=True, train_agent=False, graph=False)
        
        for epoch in range(self.args.n_epochs):
            print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ')
            sys.stdout.flush()
            
            for n_iter in range(self.args.n_cycles):
                print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n')
                sys.stdout.flush()
                self.timer.start('rollout')
                
                self.collect_experience(train_agent=True, graph=True)
                
                self.timer.end('rollout')
                self.monitor.store(TimePerSeqRollout=self.timer.get_time('rollout'))
            if epoch > self.args.start_planning_epoch :
                self.graphplanner.graph_construct(epoch)
            self.monitor.store(env_steps=self.env_steps)
            self.monitor.store(low_opt_steps=self.low_opt_steps)
            self.monitor.store(high_opt_steps=self.high_opt_steps)
            self.monitor.store(low_replay_size=self.low_replay.current_size)
            self.monitor.store(high_replay_size=self.high_replay.current_size)
            self.monitor.store(low_replay_fill_ratio=float(self.low_replay.current_size / self.low_replay.size))
            self.monitor.store(high_replay_fill_ratio=float(self.high_replay.current_size / self.high_replay.size))
            
            her_success = self.run_eval(epoch, use_test_env=True, render=self.args.eval_render)
            print('Epoch %d her eval %.3f'%(epoch, her_success))
            print('Log Path:', self.log_path)
            # logger.record_tabular("Epoch", epoch)
            self.monitor.store(Success_Rate=her_success)
            self.save_all(self.model_path)




    def run_eval(self, epoch, use_test_env=False, render=False):
        env = self.env
        if use_test_env and hasattr(self, 'test_env'):
            print("use test env")
            env = self.test_env
        total_success_count = 0
        total_trial_count = 0
        for n_test in range(self.args.n_test_rollouts):
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            for timestep in range(self.env_params['max_timesteps']):
                act = self.eval_get_actions(ob, bg)
                if render:
                    env.render()
                observation, _, _, info = env.step(act)
                ob = observation['observation']
                ag = observation['achieved_goal']
            TestEvn_Dist = env.goal_distance(ag, bg)
            self.monitor.store(TestEvn_Dist=np.mean(TestEvn_Dist))
            
            total_trial_count += 1
            if(self.args.env_name == "AntMazeSmall-v0"):
                if (TestEvn_Dist <= 2.5):
                    total_success_count += 1
            elif(self.args.env_name == "Reacher3D-v0"):
                if (TestEvn_Dist <= 0.25):
                    total_success_count += 1
            else:
                if (TestEvn_Dist <= 5):
                    total_success_count += 1
        success_rate = total_success_count / total_trial_count
        return success_rate

    

    def eval_get_actions(self, ob, bg, a_max=1, random_goal=False, act_randomly=False, graph=False):
        if ((self.curr_subgoal is None) or (self.way_to_subgoal == 0)) :
            self.curr_highpolicy_obs = ob
            sub_goal = self.high_agent.get_actions(ob, bg)

            self.curr_subgoal = sub_goal
            self.way_to_subgoal = self.subgoal_freq
            if (self.graphplanner.graph is not None):
                self.graphplanner.find_path(ob, self.curr_subgoal)

        # which waypoint to chase
        self.waypoint_subgoal = self.graphplanner.get_waypoint(ob, self.curr_subgoal)
        act = self.low_agent.get_actions(ob, self.waypoint_subgoal)
        self.way_to_subgoal -= 1 
        return act


    
    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)
    
    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']


