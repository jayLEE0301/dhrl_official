GPU=$1
SEED=$2



python DHRL/main.py \
--env_name 'AntMaze' \
--test_env_name 'AntMaze' \
--action_max 30. \
--max_steps 600 \
--high_future_step 15 \
--subgoal_freq 40 \
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
--gradual_pen 5.0 \
--subgoal_noise_eps 2 \
--cuda_num ${GPU} \
--seed ${SEED}