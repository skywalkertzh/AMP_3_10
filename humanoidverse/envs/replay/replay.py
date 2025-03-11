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


class MotionReplay(LeggedRobotLocomotion):
    def __init__(self, config, device, motion_file):
        super().__init__(config, device)
        self.motion_data = joblib.load(motion_file)
        self.motion_data = self.motion_data['walk1_subject5']
        self.current_frame = 0

    def replay_motion(self):
        if self.current_frame >= self.motion_data['dof'].shape[0]:
            print("Replay finished.")
            return
        
        # Extract the current frame's data and Set the robot's state based on the frame data
        self.simulator.robot_root_states[:, :3] = torch.tensor(self.motion_data['root_trans_offset'][self.current_frame], device=self.device)
        self.simulator.robot_root_states[:, 3:7] = torch.tensor(self.motion_data['root_rot'][self.current_frame], device=self.device)
        self.simulator.dof_pos = torch.tensor(self.motion_data['dof'][self.current_frame], device=self.device).repeat(4)
        self.simulator.dof_vel = torch.zeros_like(self.simulator.dof_pos)
        self.simulator.dof_state = torch.stack((self.simulator.dof_pos, self.simulator.dof_vel), dim=1)

        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.simulator.robot_root_states)
        )
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.simulator.dof_state)
        )
        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.render()
        # 添加刷新查看器的代码
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.current_frame += 1
        

    def run(self):
        self.gym = self.simulator.gym
        self.sim = self.simulator.sim
        
        self.simulator.set_actor_root_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(torch.arange(self.num_envs, device=self.device), self.simulator.dof_state)
        self.gym.prepare_sim(self.sim)
        # rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        
        # rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)
        # dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # actor_root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        cam_props = gymapi.CameraProperties()
        cam_props.width = 1920
        cam_props.height = 1080
        
        self.viewer = self.simulator.viewer

        if self.viewer is None:
            print("Failed to create viewer")
            return

        cam_pos = gymapi.Vec3(5.0, 5.0, 5.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        print("Viewer created successfully")
        while not self.gym.query_viewer_has_closed(self.viewer):
            if self.current_frame < self.motion_data['dof'].shape[0]:
                self.replay_motion()
                print("Replaying motion...")
                print(self.current_frame)
            else: 
                print("Replay restarted.")
                self.current_frame = 0 
        self.gym.destroy_viewer(self.viewer)


@hydra.main(config_path="../../config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacGym':
        import isaacgym

    import torch
    from humanoidverse.utils.helpers import pre_process_config
    
    os.chdir(hydra.utils.get_original_cwd())
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    config = config.env.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_file = "humanoidverse/data/motions/walk_short.pkl"
    replay = MotionReplay(config, device, motion_file)

    replay.run()

if __name__ == "__main__":
    # Example configuration and device setup

    # config = ...  # Load your configuration here
    main()



