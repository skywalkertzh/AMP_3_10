from isaacgym import gymapi, gymtorch, gymutil
import torch
import time
import os
import joblib
from humanoidverse.envs.locomotion.locomotion import LeggedRobotLocomotion
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

for frame_idx, blend_val in enumerate(blend_val_list):
    if frame_idx != 0:
        print(frame_idx)
        blend[0] = blend_val

        blr_rot = torch_utils.slerp(rest_rot, tar_rot, blend)
        dof_pos = self._motion_lib._local_rotation_to_dof(blr_rot)

        root_pos0 = rest_rp[0]
        root_pos1 = tar_rp[0]
        root_pos = (1.0 - blend_val) * root_pos0 + blend_val * root_pos1

        blr_root_rot = torch_utils.slerp(rest_rr, tar_rr, blend)

        self._root_states[0:1, 7:] = 0
        self._root_states[0:1, :3] = root_pos
        self._root_states[0:1, 3:7] = blr_root_rot[0:1]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._humanoid_actor_ids[:1]),
            len(self._humanoid_actor_ids[:1])
        )

        self._dof_pos[0:1, self.body_dof_ids] = dof_pos[0:1]
        self._dof_vel[0:1] = 0

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(self._humanoid_actor_ids[:1]),
            len(self._humanoid_actor_ids[:1])
        )

        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        
        index_list = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        tmp = self.rigid_body_states[:, index_list, :]
        rotations.append(blr_rot[0])
        root_translations.append(root_pos)
                
  