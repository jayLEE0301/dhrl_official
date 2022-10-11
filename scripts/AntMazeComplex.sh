GPU=$1
SEED=$2



python DHRL/main.py \
--env_name 'AntMazeComplex-v0' \
--test_env_name 'AntMazeComplex-v0' \
--action_max 30. \
--max_steps 2000 \
--high_future_step 25 \
--subgoal_freq 80 \
--subgoal_scale 28. 28. \
--subgoal_offset 24. 24. \
--low_future_step 150 \
--subgoaltest_threshold 1 \
--subgoal_dim 2 \
--l_action_dim 8 \
--h_action_dim 2 \
--cutoff 30 \
--n_initial_rollouts 700 \
--n_graph_node 500 \
--low_bound_epsilon 10 \
--gradual_pen 1.5 \
--subgoal_noise_eps 2 \
--cuda_num ${GPU} \
--seed ${SEED} \
--FGS