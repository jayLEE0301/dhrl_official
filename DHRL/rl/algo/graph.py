import numpy as np
import sys
import time
import networkx as nx
import torch
import random
class GraphPlanner:
    def __init__(self, args, low_replay, low_agent, env):
        self.low_replay = low_replay
        self.low_agent = low_agent
        self.env = env
        self.dim = args.subgoal_dim
        self.args = args

        self.graph = None
        self.n_graph_node = 0
        self.cutoff = args.cutoff
        self.landmarks = None
        self.states = None
        self.waypoint_vec = None
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0
        self.edge_lengths = None
        self.initial_sample = args.initial_sample
        random.seed(self.args.seed)


    def fps_selection(
            self,
            landmarks,
            states,
            n_select: int,
            inf_value=1e6,
            low_bound_epsilon=10, early_stop=True,
    ):
        n_states = landmarks.shape[0]
        dists = np.zeros(n_states) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if (np.max(dists) < low_bound_epsilon) and early_stop and (len(chosen) > self.args.n_graph_node/10):
                break
            idx = np.argmax(dists)  # farthest point idx
            farthest_state = states[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            if self.args.use_oracle_G:
                new_dists = self._get_dist_from_start_oracle(farthest_state, landmarks)
            else:
                new_dists = self.low_agent._get_dist_from_start(farthest_state, landmarks)
            new_dists[idx] = 0
            dists = np.minimum(dists, new_dists)
        return chosen
        
    def graph_construct(self, iter):
        replay_data = self.low_replay.sample_regular_batch(self.initial_sample)
        landmarks = replay_data['ag']
        states = replay_data['ob']


        idx = self.fps_selection(
            landmarks=landmarks, 
            states=states, 
            n_select=self.args.n_graph_node,
            early_stop=True,
            low_bound_epsilon=self.args.low_bound_epsilon,
        )
        self.n_graph_node = len(idx)
        self.landmarks = landmarks[idx]
        self.states = states[idx]


        #get pairwise dist
        if self.args.use_oracle_G:
            pdist = self._get_pairwise_dist_oracle(self.states)
        else:
            pdist = self.low_agent._get_pairwise_dist(self.states, self.landmarks)
        

        self.graph = nx.DiGraph()

        for i in range(self.n_graph_node):
            for j in range(self.n_graph_node):
                if i == 0:
                    min_from_i = min(pdist[i,(i+1):])
                elif i == (self.n_graph_node-1):
                    min_from_i = min(pdist[i,:i])
                else:
                    min_from_i = min(min(pdist[i,:i]), min(pdist[i,(i+1):]))
                if (pdist[i][j] < self.cutoff) or (pdist[i][j] < min_from_i * 1.2):
                    self.graph.add_edge(i, j, weight = pdist[i][j])
        return self.landmarks, self.states
    
    def find_path(self, ob, subgoal, inf_value=1e6,):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        if self.args.use_oracle_G:
            start_edge_length = self._get_dist_from_start_oracle(ob, self.landmarks)
        else:
            start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks)
        if self.args.use_oracle_G:
            goal_edge_length = self._get_dist_to_goal_oracle(self.states, subgoal)
        else:
            goal_edge_length = self.low_agent._get_dist_to_goal(self.states, subgoal)
        for i in range(self.n_graph_node):
            if(start_edge_length[i] < self.cutoff):
                expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            if(goal_edge_length[i] < self.cutoff):
                expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])

        if self.args.use_oracle_G:
            start_to_goal_length = np.squeeze(self._get_point_to_point_oracle(ob, subgoal))
        else:
            start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal))
        if start_to_goal_length < self.cutoff:
            expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
        
        self.edge_lengths = [] 
        if((not expanded_graph.has_node('start')) or (not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal'))):
            #if no edge from start point, force to make edge from start point
            if(not expanded_graph.has_node('start')):
                adjusted_cutoff = min(start_edge_length) * 1.5
                for i in range(self.n_graph_node):
                    if(start_edge_length[i] < adjusted_cutoff):
                        expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            #check whether new path has made or not
            if(expanded_graph.has_node('goal')) and (nx.has_path(expanded_graph, 'start', 'goal')):
                path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
                for (i, j) in zip(path[:-1], path[1:]):
                    self.edge_lengths.append(expanded_graph[i][j]['weight'])
            #if don't have path even though we made edges from start point
            else:
                while True:
                    nearestnode = np.argmin(goal_edge_length) #nearest point from the goal
                    if(expanded_graph.has_node(nearestnode)) and (nx.has_path(expanded_graph, 'start', nearestnode)):
                        path = nx.shortest_path(expanded_graph, 'start', nearestnode, weight='weight')
                        for (i, j) in zip(path[:-1], path[1:]):
                            self.edge_lengths.append(expanded_graph[i][j]['weight'])
                        path.append('goal')
                        self.edge_lengths.append(min(goal_edge_length))
                        break
                    else:
                        goal_edge_length[nearestnode] = inf_value #if that nearst point don't have path from start, remove it.
        elif(nx.has_path(expanded_graph, 'start', 'goal')):
            path = nx.shortest_path(expanded_graph, 'start', 'goal', weight='weight')
            for (i, j) in zip(path[:-1], path[1:]):
                self.edge_lengths.append(expanded_graph[i][j]['weight'])
        self.waypoint_vec = list(path)[1:-1]
        self.waypoint_idx = 0
        self.waypoint_chase_step = 0

    def check_easy_goal(self, ob, subgoal):
        expanded_graph = self.graph.copy()
        subgoal = subgoal[:self.dim]
        if self.args.use_oracle_G:
            start_edge_length = self._get_dist_from_start_oracle(ob, self.landmarks)
        else:
            start_edge_length = self.low_agent._get_dist_from_start(ob, self.landmarks)
        if self.args.use_oracle_G:
            goal_edge_length = self._get_dist_to_goal_oracle(self.states, subgoal)
        else:
            goal_edge_length = self.low_agent._get_dist_to_goal(self.states, subgoal)
        for i in range(self.n_graph_node):
            if(start_edge_length[i] < self.cutoff):
                expanded_graph.add_edge('start', i, weight = start_edge_length[i])
            if(goal_edge_length[i] < self.cutoff):
                expanded_graph.add_edge(i, 'goal', weight = goal_edge_length[i])

        if self.args.use_oracle_G:
            start_to_goal_length = np.squeeze(self._get_point_to_point_oracle(ob, subgoal))
        else:
            start_to_goal_length = np.squeeze(self.low_agent._get_point_to_point(ob, subgoal))
        if start_to_goal_length < self.cutoff:
            expanded_graph.add_edge('start', 'goal', weight = start_to_goal_length)
        
        if((not expanded_graph.has_node('start')) or (not expanded_graph.has_node('goal')) or (not nx.has_path(expanded_graph, 'start', 'goal'))):
            return None
        elif(nx.has_path(expanded_graph, 'start', 'goal')):
            start_edge_length = []
            for i in range (self.n_graph_node):
                if expanded_graph.has_node(i) and nx.has_path(expanded_graph, 'start', i):
                    start_edge_length.append(nx.shortest_path_length(expanded_graph, source='start', target=i, weight='weight'))
                else:
                    start_edge_length.append(5e3)
            start_edge_length = np.array(start_edge_length)
            farthest = random.choices(range(len(start_edge_length)), weights=start_edge_length)[0]
            #farthest = np.argmax(start_edge_length)

            return self.landmarks[farthest,:self.dim] + np.random.uniform(low=-3, high=3, size=self.args.subgoal_dim)

    def dist_from_graph_to_goal(self, subgoal):
        dist_list=[]
        for i in range(subgoal.shape[0]):  
            curr_subgoal = subgoal[i,:self.dim]
            if self.args.use_oracle_G:
                goal_edge_length = self._get_dist_to_goal_oracle(self.states, curr_subgoal)
            else:
                goal_edge_length = self.low_agent._get_dist_to_goal(self.states, curr_subgoal)
            dist_list.append(min(goal_edge_length))
        return np.array(dist_list)

    
    def get_waypoint(self, ob, subgoal):
        if self.graph is not None:
            self.waypoint_chase_step += 1 # how long does agent chased current waypoint
            if(self.waypoint_idx >= len(self.waypoint_vec)):
                waypoint_subgoal = subgoal
            else:
                # next waypoint or not
                if((self.waypoint_chase_step > self.edge_lengths[self.waypoint_idx]) or (np.linalg.norm(ob[:self.dim]-self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]) < 0.5)):
                    self.waypoint_idx += 1
                    self.waypoint_chase_step = 0

                if(self.waypoint_idx >= len(self.waypoint_vec)):
                    waypoint_subgoal = subgoal
                else:
                    waypoint_subgoal = self.landmarks[self.waypoint_vec[self.waypoint_idx]][:self.dim]
        else:
            waypoint_subgoal = subgoal
        return waypoint_subgoal


    #####################oracle graph#########################
    def _get_dist_to_goal_oracle(self, obs_tensor, goal):
        goal_repeat = np.ones_like(obs_tensor[:, :self.args.subgoal_dim]) \
            * np.expand_dims(goal[:self.args.subgoal_dim], axis=0)
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - goal_repeat, axis=1)
        return dist

    def _get_dist_from_start_oracle(self, start, obs_tensor):
        start_repeat = np.ones_like(obs_tensor) * np.expand_dims(start, axis=0)
        start_repeat = start_repeat[:, :self.args.subgoal_dim]
        obs_tensor = obs_tensor[:, :self.args.subgoal_dim]
        dist = np.linalg.norm(obs_tensor - start_repeat, axis=1)
        return dist

    def _get_point_to_point_oracle(self, point1, point2):
        point1 = point1[:self.args.subgoal_dim]
        point2 = point2[:self.args.subgoal_dim]
        dist = np.linalg.norm(point1-point2)
        return dist

    def _get_pairwise_dist_oracle(self, obs_tensor):
        goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = np.ones_like(goal_tensor) * np.expand_dims(obs, axis=0)
            dist = np.linalg.norm(obs_repeat_tensor[:, :self.args.subgoal_dim] - goal_tensor[:, :self.args.subgoal_dim], axis=1)
            dist_matrix.append(np.squeeze(dist))
        pairwise_dist = np.array(dist_matrix) #pairwise_dist[i][j] is dist from i to j
        return pairwise_dist