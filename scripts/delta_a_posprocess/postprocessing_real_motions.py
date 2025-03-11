import joblib
import argparse 
import os
import numpy as np
from scipy.spatial.transform import Rotation as sRot

G1_29dof_DEFAULT_ANGLES = np.array([
    -0.1,  # left_hip_pitch_joint 
    0.0,  # left_hip_roll_joint
    0.0,  # left_hip_yaw_joint
    0.3,  # left_knee_joint
    -0.2, # left_ankle_pitch_joint
    0.0,  # left_ankle_roll_joint
    -0.1, # right_hip_pitch_joint
    0.0,  # right_hip_roll_joint
    0.0,  # right_hip_yaw_joint
    0.3,  # right_knee_joint
    -0.2, # right_ankle_pitch_joint
    0.0,  # right_ankle_roll_joint
    0.0,  # waist_yaw_joint
    0.0,  # waist_roll_joint
    0.0,  # waist_pitch_joint
    0.0,  # left_shoulder_pitch_joint
    0.0,  # left_shoulder_roll_joint
    0.0,  # left_shoulder_yaw_joint
    0.0,  # left_elbow_joint
    0.0,  # left_wrist_roll_joint
    0.0,  # left_wrist_pitch_joint
    0.0,  # left_wrist_yaw_joint
    0.0,  # right_shoulder_pitch_joint
    0.0,  # right_shoulder_roll_joint
    0.0,  # right_shoulder_yaw_joint
    0.0,  # right_elbow_joint
    0.0,  # right_wrist_roll_joint
    0.0,  # right_wrist_pitch_joint
    0.0   # right_wrist_yaw_joint
])

G1_29DOF_DOF_AXIS = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])


def process_single_motion(one_motion_data):
    """
    Desired motion format
    root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
    root_rot = self.simulator.robot_root_states[:, 3:7].cpu()
    root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
    dof = self.simulator.dof_pos.cpu()
    # T, num_env, J, 3
    pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
    self.motions_for_saving['root_trans_offset'].append(root_trans)
    self.motions_for_saving['root_rot'].append(root_rot)
    self.motions_for_saving['dof'].append(dof)
    self.motions_for_saving['pose_aa'].append(pose_aa)
    self.motions_for_saving['action'].append(self.actions.cpu())
    self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
    self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
    self.motion_times_buf.append(motion_times.cpu())
    """
    single_motion_data_motionlib_format = {"root_trans_offset": [], "root_rot": [], "dof": [], "pose_aa": [], "action": [], "actor_obs": [], "terminate": [], "motion_times": []}
    # for key in data.files:
    #     print(key, data[key][:10])
    motion_length = len(one_motion_data["joint_pos"])

    for t in range(motion_length):
        single_motion_data_motionlib_format["root_trans_offset"].append(one_motion_data['pos'][t]) # [n, 3]
        Z_OFFSET = 0.07
        single_motion_data_motionlib_format["root_trans_offset"][t][2] += Z_OFFSET
        # convert wxyz to xyzw
        single_motion_data_motionlib_format["root_rot"].append(one_motion_data['IMU_quaternion'][t][[1,2,3,0]])
        single_motion_data_motionlib_format["dof"].append(one_motion_data['joint_pos'][t]) # [n, num_dof]

        root_rot_vec = sRot.from_quat(np.array([one_motion_data['IMU_quaternion'][t][[1,2,3,0]]])).as_rotvec()
        NUM_AUGMENT_JOINT = 3
        single_motion_data_motionlib_format["pose_aa"].append(np.concatenate([root_rot_vec, G1_29DOF_DOF_AXIS * one_motion_data['joint_pos'][t][:, None], np.zeros((NUM_AUGMENT_JOINT, 3))], axis=0))

        # need to minus default joint angles here, and rescale the action by 4.0 to align actual RL action

        single_motion_data_motionlib_format["action"].append((one_motion_data['joint_pos'][t] - G1_29dof_DEFAULT_ANGLES) * 4.0 )
        single_motion_data_motionlib_format["actor_obs"].append(None) 
        single_motion_data_motionlib_format["terminate"].append(0)
        single_motion_data_motionlib_format["motion_times"].append(one_motion_data['time'][t] - one_motion_data['time'][0])
        single_motion_data_motionlib_format["fps"] = 50.0
    
    # turn all the lists into np arrays
    for key in single_motion_data_motionlib_format.keys():
        if key != "fps":
            single_motion_data_motionlib_format[key] = np.array(single_motion_data_motionlib_format[key])
    
    # make sure all the datatype is float
    for key in single_motion_data_motionlib_format.keys():
        if key != "terminate" and key != "fps":
            single_motion_data_motionlib_format[key] = single_motion_data_motionlib_format[key].astype(np.float32)

    return single_motion_data_motionlib_format
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", type=str, default=None)
    args = parser.parse_args()


    motion_dir = args.motion_dir
    motion_list = os.listdir(motion_dir)
    motion_list = sorted(motion_list)
    print("motion_list: ", motion_list)
    truncated_data = {}

    for motion in motion_list: # npz file lists
        if "npz" not in motion:
            continue
        episode_path = os.path.join(motion_dir, motion)
        print("processing ", episode_path)
        data=np.load(episode_path, allow_pickle=True)
        proceseed_data = process_single_motion(data)
        truncated_data[motion.replace(".npz", "")] = proceseed_data
        # import ipdb; ipdb.set_trace()


    # import ipdb; ipdb.set_trace() 
    new_motion_path = os.path.join(motion_dir, "truncated_motion_data.pkl")
    joblib.dump(truncated_data, new_motion_path)
