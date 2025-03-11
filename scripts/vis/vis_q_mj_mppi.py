
import time
from roboverse.utils.motion_lib.skeleton import SkeletonTree
import torch

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from termcolor import colored
from pathlib import Path
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print(colored("Reset", "red"))
        time_step = 0
    elif chr(keycode) == " ":
        print(colored("Paused", "green"))
        paused = not paused
    elif chr(keycode) == "N":
        print(colored("Next", "green"))
        if motion_id >= len(motion_data_keys) - 1:
            print(colored("End of Motion", "red"))
            motion_id = 0
        else:
            motion_id += 1
            curr_motion_key = motion_data_keys[motion_id]
            print(curr_motion_key)
    else:
        print(colored(f"Not mapped: {chr(keycode)}", "red"))


@hydra.main(version_base=None, config_path="../../roboverse/config", config_name="base")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")

    asset_path = Path(cfg.robot.motion.asset.assetRoot)
    asset_file = cfg.robot.motion.asset.assetFileName
    humanoid_xml = asset_path / asset_file

    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False

    if cfg.visualize_motion_file is None:
        logger.error(colored("No motion file provided", "red"))
    else:
        visualize_motion_file = cfg.visualize_motion_file
    logger.info(colored(f"Visualizing Motion: {visualize_motion_file}", "green"))
    motion_data = joblib.load(visualize_motion_file)
    # import ipdb; ipdb.set_trace()
    # motion_data_keys = list(motion_data.keys())

    

    mj_model = mujoco.MjModel.from_xml_path(str(humanoid_xml))
    mj_data = mujoco.MjData(mj_model)
    
    mj_model.opt.timestep = dt
    cnt = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        while viewer.is_running():
            # import ipdb; ipdb.set_trace()
            if cnt >= len(motion_data):
                cnt = 0
            mj_data.qpos[:3] = motion_data[cnt]['root_states'][:,:3].squeeze().cpu().numpy()
            mj_data.qpos[3:7] = motion_data[cnt]['root_states'][:, [6,3,4,5]].squeeze().cpu().numpy()  # isaac xyzw to mujoco wxyz
            mj_data.qpos[7:7+12] = motion_data[cnt]['dof_states'][:, :, 0].squeeze().cpu().numpy()
            mj_data.qvel[7+12:] = np.zeros(10)
                
            mujoco.mj_forward(mj_model, mj_data)
            
            cnt += 1
            time.sleep(0.02) # the dumped state data is 50hz
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()


if __name__ == "__main__":
    main()
