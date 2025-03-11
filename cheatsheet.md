# AMP:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=locomotion_amp \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.01 \
rewards.reward_penalty_degree=0.0001

# loco_test:_1
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_1 \
robot.motion.motion_file="humanoidverse/data/motions/kit_loco/singles/0-KIT_3_walking_slow08_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0001

# loco_test_2:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp_test_2 \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
+checkpoints=logs/test_2/20250228_203833-TEST_Locomotion_AMP-locomotion-g1_29dof_anneal_23dof/model_2700.pt \
num_envs=4096 \
project_name=test_2 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0001

# loco_test_3:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_3 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.000065

# loco_test_4:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_4 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.000065
 <!-- coef:0.001 -->

# loco_test_5:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_5 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0001

# loco_test_6:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_6 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0001

# amp_test_0_task_lerp:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=3_10 \
robot.motion.motion_file="humanoidverse/data/motions/walk_short.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0003 \
algo.config.amp.amp_task_reward_lerp=0.0

# amp+task
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=debug_walk_reward_lerp_0.5 \
robot.motion.motion_file="humanoidverse/data/motions/kit_loco/singles/0-KIT_3_walking_slow08_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.0003 \
algo.config.amp.amp_task_reward_lerp=0.5 \
algo.config.amp.amp_reward_coef=0.01

# amp_validation_test_2:
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_validation_accad_walk_turn45 \
robot.motion.motion_file="humanoidverse/data/motions/accad_motions/g1_29dof_anneal_23dof/accad_loco/singles/0-ACCAD_Female1Walking_c3d_B10 - walk turn left_45_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.00001

<!-- coef = 0.1/ task_lerp = 0 -->

# replay:
HYDRA_FULL_ERROR=1 python humanoidverse/envs/replay/replay.py \
    +exp=locomotion \
    +simulator=isaacgym \
    +domain_rand=NO_domain_rand \
    +rewards=loco/reward_g1_locomotion \
    +robot=g1/g1_29dof_anneal_23dof \
    +terrain=terrain_locomotion_plane \
    +obs=loco/leggedloco_obs_singlestep_withlinvel \
    +opt=wandb \
    headless=False \
    num_envs=4 \
    project_name=test_1 \
    robot.motion.motion_file="humanoidverse/data/motions/walk_short.pkl" \
    rewards.reward_penalty_curriculum=True \
    rewards.reward_initial_penalty_scale=0.05 \
    rewards.reward_penalty_degree=0.0001 

# wave hand
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=locomotion_amp \
+simulator=isaacgym \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof_amp \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
+opt=wandb \
num_envs=4096 \
project_name=test_validation_wave_hand \
robot.motion.motion_file="humanoidverse/data/motions/kit_loco/singles/0-KIT_3_wave_left01_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.00001



# eval:
python humanoidverse/eval_agent.py +checkpoint=


# Reference:
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MotionTracking_CR7 \
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0