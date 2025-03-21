import glob
import os
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from isaac_utils.rotations import calc_heading_quat
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from roboverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf

from rich.progress import track
from rich.progress import Progress
from termcolor import colored
from loguru import logger


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    
def process_motion(key_names, key_name_to_pkls, cfg):
    logger.info(colored(f"Retargeting Motion...", "cyan"))
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot.motion) # load forward kinematics model
    num_augment_joint = len(cfg.robot.motion.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.motion.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.motion.joint_matches]
    robot_joint_pick_idx = [ robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    smpl_parser_n = SMPL_Parser(model_path="roboverse/data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"roboverse/data/shape/{cfg.robot.motion.humanoid_type}/shape_optimized_v1.pkl")

    all_data = {}
    
    # loss_weights = torch.ones(len(smpl_joint_pick))
    # for i in range(len(loss_weights)):
    #     if smpl_joint_pick[i] in ['L_Ankle', 'R_Ankle']:
    #         loss_weights[i] = 3
    # loss_weights = loss_weights[None,None,:,None]

    with Progress() as progress:
        task_retarget = progress.add_task(
            f"Retargeting Motion...", total=len(key_names)
        )
        task_fit = progress.add_task(
            f"Fitting Motion...", total=len(key_names)
        )
        for data_key in key_names:
            amass_data = load_amass_data(key_name_to_pkls[data_key])
            if amass_data is None: continue
            skip = int(amass_data['fps']//30)
            trans = torch.from_numpy(amass_data['trans'][::skip])
            N = trans.shape[0]
            pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()
            
            if N < 10:
                print("to short")
                continue

            with torch.no_grad():
                verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
                root_pos = joints[:, 0:1]
                joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
            joints[..., 2] -= verts[0, :, 2].min().item()
            
            offset = joints[:, 0] - trans
            root_trans_offset = (trans + offset).clone()

            gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this 
            gt_root_rot = torch.from_numpy(sRot.from_quat(calc_heading_quat(gt_root_rot_quat, w_last=True)).as_rotvec()).float() # so only use the heading. 
            
            # def dof_to_pose_aa(dof_pos):
            dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

            dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
            root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
            root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
            optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)
            optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset],lr=0.01)

            kernel_size = 5  # Size of the Gaussian kernel
            sigma = 0.75  # Standard deviation of the Gaussian kernel
            B, T, J, D = dof_pos_new.shape 

            for iteration in range(cfg.get("fitting_iterations", 1000)):
                pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)
                fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ] + root_pos_offset )
                
                if num_augment_joint > 0:
                    diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                else:
                    diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                
                # diff = diff * loss_weights
                loss_g = diff.norm(dim = -1).mean() 
                loss = loss_g
                
                optimizer_pose.zero_grad()
                optimizer_root.zero_grad()
                loss.backward()
                optimizer_pose.step()
                optimizer_root.step()
                
                dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

                progress.update(task_fit, advance=1, description=f"Fitting Motion:{data_key} \n -Iter: {iteration} \n -Loss: {loss.item() * 1000:.3f}")
                dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]
                
            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
            pose_aa_h1_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)

            height_diff = fk_return.global_translation[..., 2].min().item() 
            root_trans_offset_dump = (root_trans_offset + root_pos_offset ).clone()
            #js change
            joint_min_id = torch.argmin(torch.min(fk_return.global_translation[..., 2].detach(),dim=-1)[0]).item()
            combined_mesh = humanoid_fk.mesh_fk(pose_aa_h1_new[:, joint_min_id:joint_min_id+1].detach(), root_trans_offset_dump[None, joint_min_id:joint_min_id+1].detach())
            height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()

            root_trans_offset_dump[..., 2] -= height_diff
            joints_dump = joints.numpy().copy()
            joints_dump[..., 2] -= height_diff
            
            data_dump = {
                        "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
                        "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),   
                        "dof": dof_pos_new.squeeze().detach().numpy(), 
                        "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
                        "smpl_joints": joints_dump, 
                        "fps": 30
                        }
            all_data[data_key] = data_dump
            progress.update(task_retarget, advance=1, description=f"Retargeting Motion:{data_key}")
    return all_data
        

@hydra.main(version_base=None, config_path="../../roboverse/config", config_name="base")
def main(cfg : DictConfig) -> None:
    all_pkls = glob.glob("./roboverse/data/AMASS/AMASS_Complete/ACCAD/**/*.npz", recursive=True)
    hardcode_motion_name_idx = 5
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[hardcode_motion_name_idx:]).replace(".npz", ""): data_path for data_path in all_pkls}
    key_names = ["0-" + "_".join(data_path.split("/")[hardcode_motion_name_idx:]).replace(".npz", "") for data_path in all_pkls]
    # if not cfg.get("fit_all", False):
    #     key_names = ['0-ACCAD_Male2General_c3d_A6- Box lift_poses']
        # key_names = ['0-Robosiuuu_CR7_video_cr_2_filter_amass']
        # key_names = ['0-Robosiuuu_CR7_video_cr_11_filter_amass']
        # key_names = ['0-Robosiuuu_CR7_video_motion_yl1_filter_amass']
        # key_names = ['0-Robosiuuu_TairanG1_video_motion_tairan5_filter_amass']

    # Jiawei: Multi processing still have bugs until December 2024.
    # from multiprocessing import Pool
    # jobs = key_names
    # num_jobs = 30
    # chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    # jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    # job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
    # if len(job_args) == 1:
    #     all_data = process_motion(key_names, key_name_to_pkls, cfg)
    # else:
    #     try:
    #         pool = Pool(num_jobs)   # multi-processing
    #         all_data_list = pool.starmap(process_motion, job_args)
    #     except KeyboardInterrupt:
    #         pool.terminate()
    #         pool.join()
    #     all_data = {}
    #     for data_dict in all_data_list:
    #         all_data.update(data_dict)

    # single process
    # all_data = process_motion(key_names, key_name_to_pkls, cfg)

    # if len(all_data) == 1:
    #     data_key = list(all_data.keys())[0]
    #     os.makedirs(f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/v1/singles", exist_ok=True)
    #     dumped_file = f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/v1/singles/{data_key}.pkl"
    #     logger.info(colored(f"Dumping to {dumped_file}", "green"))
    #     joblib.dump(all_data, dumped_file)
    # else:
    #     os.makedirs(f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/v1/", exist_ok=True)
    #     joblib.dump(all_data, f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/v1/amass_all.pkl")
    #     logger.info(colored(f"Dumping to roboverse/data/motions/{cfg.robot.motion.humanoid_type}/v1/amass_all.pkl", "green"))

    select_motion_keys_list = ['0-ACCAD_Female1General_c3d_A1 - Stand_poses',
                           '0-ACCAD_Female1General_c3d_A5 - pick up box_poses',
                           '0-ACCAD_Female1General_c3d_A6 - lift box_poses',
                           '0-ACCAD_Female1General_c3d_A6- lift box t2_poses',
                           '0-ACCAD_Female1Walking_c3d_B1 - stand to walk_poses',
                           '0-ACCAD_Female1Walking_c3d_B2 - walk to stand t2_poses',
                           '0-ACCAD_Female1Walking_c3d_B2 - walk to stand_poses',
                           '0-ACCAD_Female1Walking_c3d_B3 - walk1_poses',
                           '0-ACCAD_Female1Walking_c3d_B9 - walk turn left (90)_poses',
                           '0-ACCAD_Female1Walking_c3d_B10 - walk turn left (45)_poses',
                           '0-ACCAD_Female1Walking_c3d_B11 - walk turn left (135)_poses',
                           '0-ACCAD_Female1Walking_c3d_B12 - walk turn right (90)_poses',
                           '0-ACCAD_Female1Walking_c3d_B19 - walk to pick up box_poses',
                           '0-ACCAD_Female1Walking_c3d_B20 - walk with box_poses',
                           '0-ACCAD_Female1Walking_c3d_B21 - put down box to walk_poses',
                           '0-ACCAD_Female1Walking_c3d_B21 s2 - put down box to walk_poses',
                           '0-ACCAD_Female1Walking_c3d_B21 s3 - put down box to walk_poses',
                           '0-ACCAD_Male1General_c3d_General A1 - Stand_poses',
                           '0-ACCAD_Male1General_c3d_General A5 - Pick Up Box_poses',
                           '0-ACCAD_Male1General_c3d_General A6 - Lift Box_poses',
                           '0-ACCAD_Male1Walking_c3d_Walk B4 - Stand to Walk Back_poses',
                           '0-ACCAD_Male1Walking_c3d_Walk B10 - Walk turn left 45_poses',
                           '0-ACCAD_Male1Walking_c3d_Walk B13 - Walk turn right 45_poses',
                           '0-ACCAD_Male1Walking_c3d_Walk B15 - Walk turn around_poses',
                           '0-ACCAD_Male1Walking_c3d_Walk B21 - Put Down Box to walk_poses',
                           '0-ACCAD_Male2General_c3d_A5- Pick up box_poses',
                           '0-ACCAD_Male2General_c3d_A6- Box lift_poses',
                           '0-ACCAD_Male2Walking_c3d_B4 - Stand to Walk backwards_poses',
                           '0-ACCAD_Male2Walking_c3d_B9 -  Walk turn left 90_poses',
                           '0-ACCAD_Male2Walking_c3d_B10 -  Walk turn left 45_poses',
                           '0-ACCAD_Male2Walking_c3d_B11 -  Walk turn left 135_poses',
                           '0-ACCAD_Male2Walking_c3d_B13 -  Walk turn right 90_poses',
                           '0-ACCAD_Male2Walking_c3d_B14 -  Walk turn right 45 t2_poses',
                           '0-ACCAD_Male2Walking_c3d_B19 -  Walk to pickup box_poses',
                           '0-ACCAD_Male2Walking_c3d_B20 -  Walk with box_poses',
                           '0-ACCAD_Male2Walking_c3d_B21 -  put down box to walk_poses',
                           '0-ACCAD_Male2Walking_c3d_B21 -  put down box to walk a_poses']

    # select_motion_keys_list = ['0-ACCAD_Male2General_c3d_A5- Pick up box_poses',
    #                        '0-ACCAD_Male2General_c3d_A6- Box lift_poses']

    # Process each motion individually
    for key_name in key_names:
        if key_name not in select_motion_keys_list:
            continue
        current_key_names = [key_name]
        current_data = process_motion(current_key_names, key_name_to_pkls, cfg)
        
        if len(current_data) > 0:  # Only save if processing was successful
            os.makedirs(f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/accad_loco/singles", exist_ok=True)
            dumped_file = f"roboverse/data/motions/{cfg.robot.motion.humanoid_type}/accad_loco/singles/{key_name}.pkl"
            logger.info(colored(f"Dumping to {dumped_file}", "green"))
            joblib.dump(current_data, dumped_file)

if __name__ == "__main__":
    main()
