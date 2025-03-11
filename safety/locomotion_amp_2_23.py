from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
# from humanoidverse.envs.env_utils.command_generator import CommandGenerator
# from humanoidverse.agents.amp_modules.motion_loader import AMPLoader
from scipy.stats import vonmises
import glob

COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET
#! usage unknown

class LeggedRobotLocomotionAMP(LeggedRobotLocomotion):
    def __init__(self, config, device):
        
        self.frames = None
        super().__init__(config, device)        
        # TODO:
        self.amp_motion_file_dir = self.config.amp.amp_motion_file_dir
        self.amp_motion_files = glob.glob(self.amp_motion_file_dir)
        
        self.amp_loader = AMPLoader(motion_files=self.amp_motion_files, device=self.device, time_between_frames=self.dt)
    
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
        foot_pos = self.foot_positions_in_base_frame(self.simulator.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.simulator.dof_vel
        z_pos = self.simulator.robot_root_states[:, 2:3]
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
    
    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        # foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions
    
    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)
    
    #######################
    ## legged_robot_base ##
    #######################


    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        super().reset_envs_idx(env_ids, target_states, target_buf)
        """        
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)
        """
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
        self.simulator.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(self.frames)
        self.simulator.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(self.frames)
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
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
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.simulator.robot_root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.simulator.robot_root_states[env_ids, 3:7] = root_orn
        self.simulator.robot_root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.simulator.robot_root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

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