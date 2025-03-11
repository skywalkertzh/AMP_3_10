from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
from isaac_utils.rotations import get_euler_xyz_in_tensor
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
from humanoidverse.utils.motion_lib.motion_lib_amp_loader import MotionLibAMPLoader as AMPLoader
from scipy.stats import vonmises
import glob


class LeggedRobotLocomotionAMP(LeggedRobotLocomotion):
    def __init__(self, config, device):
        
        self.frames = None
        super().__init__(config, device)        
        # TODO: change AMPLoader back to motion_lib... 
        # #* __init__(self, motion_lib_cfg, num_envs, device)
        self.config.robot.motion.step_dt = self.dt
        self.amp_loader = AMPLoader(self.config.robot.motion, self.num_envs, self.device)
        self.amp_loader.load_motions()
        self.amp_loader.preload_transitions()
        # import pdb; pdb.set_trace()

    
    def _post_physics_step(self):
        super()._post_physics_step()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        terminal_amp_states = self.get_amp_observations()[env_ids]
        return env_ids, terminal_amp_states

    #########
    ## AMP ##
    #########
    def get_amp_observations(self):
        joint_pos = self.simulator.dof_pos
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.simulator.dof_vel
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
    
    def get_amp_observations_dim(self):
        return self.get_amp_observations().shape[-1]
    
    #######################
    ## legged_robot_base ##
    #######################

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)# if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf


    def _reset_robot_states_callback(self, env_ids, target_states=None):
        self.frames = self.amp_loader.get_full_frame_batch(len(env_ids))
        self._reset_dofs_amp(env_ids, self.frames)
        self._reset_root_states_amp(env_ids, self.frames)

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
    
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.simulator.dof_pos[env_ids] = self.amp_loader.get_joint_pose_batch(frames)
        self.simulator.dof_vel[env_ids] = self.amp_loader.get_joint_vel_batch(frames)
        # self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        #! 
    
    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = self.amp_loader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.simulator.robot_root_states[env_ids, :3] = root_pos
        root_orn = self.amp_loader.get_root_rot_batch(frames)
        self.simulator.robot_root_states[env_ids, 3:7] = root_orn
        self.simulator.robot_root_states[env_ids, 7:10] = quat_rotate(root_orn, self.amp_loader.get_linear_vel_batch(frames))
        self.simulator.robot_root_states[env_ids, 10:13] = quat_rotate(root_orn, self.amp_loader.get_angular_vel_batch(frames))

    #################
    ## env.step()  ##
    #################

    def step(self, actor_state):
        #amp version
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]

        self._pre_physics_step(actions)
        self._physics_step()
        reset_env_ids, terminal_amp_states = self._post_physics_step()
        # add AMP obs into obs_dict
        self.obs_buf_dict["amp_obs"] = self.get_amp_observations()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states
    
    def reset_all(self):
        """ Reset all robots"""
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        # self._refresh_env_idx_tensors(torch.arange(self.num_envs, device=self.device))
        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        obs_dict, _, _, _ , _, _= self.step(actor_state)
        return obs_dict
