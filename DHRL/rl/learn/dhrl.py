import torch
from torch.optim import Adam

import os.path as osp
from rl.utils import net_utils
import numpy as np
from rl.learn.core import dict_to_numpy

class HighLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim1 = Adam(agent.critic1.parameters(), lr=args.lr_critic)
        self.q_optim2 = Adam(agent.critic2.parameters(), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        

    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            Loss_q1=loss_q1.item(),
            Loss_q2=loss_q2.item(),
            Loss_critic_1=loss_critic['critic_1'].item(),
            Loss_critic_2=loss_critic['critic_2'].item(),
        )
        monitor_log = dict(
            q_targ=q_targ,
            offset=offset,
            r=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic
    
    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        

        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg)
        subgoal_scale = torch.as_tensor(self.args.subgoal_scale, dtype=torch.float32).cuda(device=self.args.cuda_num)
        subgoal_offset = torch.as_tensor(self.args.subgoal_offset, dtype=torch.float32).cuda(device=self.args.cuda_num)
        action_l2 = ((pi - subgoal_offset) / subgoal_scale).pow(2).mean()
        loss_actor = (- q_pi).mean()
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            Loss_actor=loss_actor.item(),
            Loss_action_l2=action_l2.item(),
            Loss_bc=loss_bc.item(),
        )
        monitor_log = dict(q_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    
    
    def update_critic(self, batch, train_embed=True):
        loss_critic1 = self.critic_loss(batch)['critic_1']
        loss_critic2 = self.critic_loss(batch)['critic_2']
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(gradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim1.step()
        self.q_optim2.step()
            
    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim1=self.q_optim1.state_dict(),
            q_optim2=self.q_optim2.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim1.load_state_dict(state_dict['q_optim1'])
        self.q_optim2.load_state_dict(state_dict['q_optim2'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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


class LowLearner:
    def __init__(
        self,
        agent,
        monitor,
        args,
        name='learner',
    ):
        self.agent = agent
        self.monitor = monitor
        self.args = args
        
        self.q_optim1 = Adam(agent.critic1.parameters(), lr=args.lr_critic)
        self.q_optim2 = Adam(agent.critic2.parameters(), lr=args.lr_critic)
        self.q_optim1_g = Adam(agent.critic1_g.parameters(), lr=args.lr_critic)
        self.q_optim2_g = Adam(agent.critic2_g.parameters(), lr=args.lr_critic)
        self.pi_optim = Adam(agent.actor.parameters(), lr=args.lr_actor)
        
        self._save_file = str(name) + '.pt'
    
    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        
        q_bg1 = self.agent.get_qs(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs(o, bg, a, net = 2)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()
        
    
        loss_critic = {'critic_1' : loss_q1, 'critic_2' : loss_q2}
        
        self.monitor.store(
            Loss_q1=loss_q1.item(),
            Loss_q2=loss_q2.item(),
            Loss_critic_1=loss_critic['critic_1'].item(),
            Loss_critic_2=loss_critic['critic_1'].item(),
        )
        monitor_log = dict(
            q_targ=q_targ,
            offset=offset,
            r=r,
        )
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_critic

    def critic_loss_g(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            noise = np.random.randn(*a.shape) *0.05 
            noise = self.agent.to_tensor(noise)
            n_a = self.agent.get_pis(o2, bg, pi_target=True) + noise
            q_next = self.agent.get_qs_g(o2, bg, n_a, q_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, max = 0.0)

        q_bg1 = self.agent.get_qs_g(o, bg, a, net = 1)
        q_bg2 = self.agent.get_qs_g(o, bg, a, net = 2)
        q_bg = self.agent.get_qs_g(o, bg, a)
        loss_q1 = (q_bg1 - q_targ).pow(2).mean()
        loss_q2 = (q_bg2 - q_targ).pow(2).mean()


        loss_critic = {'critic_1_g' : loss_q1, 'critic_2_g' : loss_q2}
        return loss_critic
    
    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']
        
        a = self.agent.to_tensor(a)
        
        q_pi, pi = self.agent.forward1(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2
        
        pi_future = self.agent.get_pis(o, future_ag)
        loss_bc = (pi_future - a).pow(2).mean()
        
        self.monitor.store(
            Loss_actor=loss_actor.item(),
            Loss_action_l2=action_l2.item(),
            Loss_bc=loss_bc.item(),
        )
        monitor_log = dict(q_pi=q_pi)
        self.monitor.store(**dict_to_numpy(monitor_log))
        return loss_actor
    
    
    def update_critic(self, batch, train_embed=True):
        loss_critic1 = self.critic_loss(batch)['critic_1']
        loss_critic2 = self.critic_loss(batch)['critic_2']
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic1=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic2=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_critic1=net_utils.mean_grad_norm(self.agent.critic1.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1.parameters(), self.args.grad_value_clipping)
            self.monitor.store(gradnorm_mean_critic2=net_utils.mean_grad_norm(self.agent.critic2.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2.parameters(), self.args.grad_value_clipping)
        self.q_optim1.step()
        self.q_optim2.step()

    def update_critic_g(self, batch, train_embed=True):
        loss_critic1 = self.critic_loss_g(batch)['critic_1_g']
        loss_critic2 = self.critic_loss_g(batch)['critic_2_g']
        self.q_optim1_g.zero_grad()
        self.q_optim2_g.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        if self.args.grad_norm_clipping > 0.:
            c_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1_g.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic1_g=c_norm1)
            c_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2_g.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_critic2_g=c_norm2)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_critic1_g=net_utils.mean_grad_norm(self.agent.critic1_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic1_g.parameters(), self.args.grad_value_clipping)
            self.monitor.store(gradnorm_mean_critic2_g=net_utils.mean_grad_norm(self.agent.critic2_g.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.critic2_g.parameters(), self.args.grad_value_clipping)
        self.q_optim1_g.step()
        self.q_optim2_g.step()
            
    def update_actor(self, batch, train_embed=True):
        loss_actor = self.actor_loss(batch)
        self.pi_optim.zero_grad()
        loss_actor.backward()
        
        if self.args.grad_norm_clipping > 0.:
            a_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.args.grad_norm_clipping)
            self.monitor.store(gradnorm_actor=a_norm)
        if self.args.grad_value_clipping > 0.:
            self.monitor.store(gradnorm_mean_actor=net_utils.mean_grad_norm(self.agent.actor.parameters()).item())
            torch.nn.utils.clip_grad_value_(self.agent.actor.parameters(), self.args.grad_value_clipping)
        self.pi_optim.step()
    
    def target_update(self):
        self.agent.target_update()
    
    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    
    def state_dict(self):
        return dict(
            q_optim1=self.q_optim1.state_dict(),
            q_optim2=self.q_optim2.state_dict(),
            pi_optim=self.pi_optim.state_dict(),
        )
    
    def load_state_dict(self, state_dict):
        self.q_optim1.load_state_dict(state_dict['q_optim1'])
        self.q_optim2.load_state_dict(state_dict['q_optim2'])
        self.pi_optim.load_state_dict(state_dict['pi_optim'])
    
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

