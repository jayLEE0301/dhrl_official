GPU=$1
SEED=$2



python DHRL/main.py \
--env_name 'AntMazeBottleneck-v0' \
--test_env_name 'AntMazeBottleneck-eval-v0' \
--action_max 30. \
--max_steps 600 \
--high_future_step 12 \
--subgoal_freq 50 \
--subgoal_scale 12. 12. \
--subgoal_offset 8. 8. \
--low_future_step 150 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 30 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 10 \
--gradual_pen 1.5 \
--subgoal_noise_eps 2 \
--cuda_num ${GPU} \
--seed ${SEED}