GPU=$1
SEED=$2



python DHRL/main.py \
--env_name 'Reacher3D-v0' \
--test_env_name 'Reacher3D-v0' \
--action_max 20. \
--max_steps 100 \
--high_future_step 10 \
--subgoal_freq 10 \
--subgoal_scale 1. 1. 1. \
--subgoal_offset 0. 0. 0. \
--low_future_step 100 \
--subgoaltest_threshold 0.1 \
--subgoal_dim 3 \
--l_action_dim 7 \
--h_action_dim 3 \
--cutoff 10 \
--n_initial_rollouts 200 \
--n_graph_node 300 \
--low_bound_epsilon 5 \
--gradual_pen 5.0 \
--subgoal_noise_eps 0.1 \
--cuda_num ${GPU} \
--seed ${SEED} \
--absolute_goal