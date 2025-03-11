from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

import numpy as np
import torch


class MotionLibAMPLoader(MotionLibBase):
    #self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
    #TODO 0: 先不管多个file.先调通单个file
    #TODO Un:存为这样的traj的形式/如何从list的形式转为现在的形式
    
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device) # "self.m_cfg"
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        self.trajectory_idxs = np.arange(motion_lib_cfg.amp_config.MotionFileNum)
        self.trajectory_weights = self.m_cfg.amp_config.MotionWeight
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights) 
        self.num_preload_transitions = motion_lib_cfg.amp_config.num_preload_transitions 
        self.time_between_frames = self.m_cfg.step_dt #! time between frames: dt

    def preload_transitions(self): #TODO have to execute after load_motion
        """ calculate corresponding parameters
        # One "trajectory" means one amp motion trajectory colleted from one of the motion files
        """

        self.trajectory_frame_durations = self._motion_dt
        import ipdb; ipdb.set_trace()
        self.trajectory_num_frames = self._motion_num_frames
        self.trajectory_lens = self._motion_lengths
        #TODO 1
        print(f'Preloading {self.num_preload_transitions} transitions') #* done
        #TODO 2
        traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions) #* done
        #array([1, 5, 4, ..., 1, 0, 0]),长度20000
        #TODO 3
        times = (self.traj_time_sample_batch(traj_idxs)).to(self._device) #* done 
        #[0.42860936, 0.43710256, 8.52827445, ..., 0.47066521, 1.75933222, 1.20181006]，长度20000
        self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
        self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames) # [s,s']中的s'            print(f'Finished preloading')
        # import ipdb; ipdb.set_trace()
    def weighted_traj_idx_sample_batch(self, size): 
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        # import ipdb; ipdb.set_trace()
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * torch.rand(len(traj_idxs), device=self._device) - subst
        ret = torch.maximum(torch.zeros_like(time_samples), time_samples)
        return ret

    def get_full_frame_at_time_batch(self, traj_idxs, times): #* 需要在load_motion之后运行
        motion_res = self.get_motion_state(traj_idxs, times)
        root_pos, root_rot, dof_pos, rot_vel, rot_ang_vel, dof_vel = motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"]
        return torch.cat([root_pos, root_rot, dof_pos, rot_vel, rot_ang_vel, dof_vel], dim=-1) #* changable

    def get_full_frame_batch(self, num_frames):
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
    

    #* self.frames = self.amp_loader.get_full_frame_batch(len(env_ids))
    #*    get_full_frame_batch will return [2000000, 49]
    #* frames: torch.cat([dof_pos, rot_vel, rot_ang_vel, dof_vel], dim=-1)]
    #*                    3/ 4/ 23/ 3/ 3/ 23
    #* self.simulator.dof_pos[env_ids] = self.amp_loader.get_joint_pose_batch(frames)
    
    #! should change according to diffenrent robot
    def get_root_pos_batch(self, frames):
        return frames[:,0:3]
    def get_root_rot_batch(self, frames):
        return frames[:,3:7]
    def get_joint_pose_batch(self, frames): 
        return frames[:,7:30]
    def get_linear_vel_batch(self, frames):
        return frames[:,30:33]
    def get_angular_vel_batch(self, frames):
        return frames[:,33:36]
    def get_joint_vel_batch(self, frames):
        return frames[:,36:59]

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=mini_batch_size)
            
            s = self.preloaded_s[idxs, 7:] 
            s_next = self.preloaded_s_next[idxs, 7:]
            yield s, s_next
