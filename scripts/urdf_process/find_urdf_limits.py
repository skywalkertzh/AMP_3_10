import xml.etree.ElementTree as ET

urdf_file_path = 'roboverse/data/robots/g1/g1_29dof.urdf'

print("urdf_file_path:", urdf_file_path)


# h1
# dof_names = [
#     'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint',
#     'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint',
#     'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
#     'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
# ]

# g1_12dof
# dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
#               'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
  
# g1_23dof
# dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
#               'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
#               'waist_yaw_joint', 
#               'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 
#               'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']

# g1_unitree_deprecated
# dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
#               'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']
  
# g1_29dof
dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
              'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
              'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
              'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
              'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
              'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']

dof_pos_lower_limit_list = []
dof_pos_upper_limit_list = []
dof_vel_limit_list = []
dof_effort_limit_list = []

tree = ET.parse(urdf_file_path)
root = tree.getroot()

for joint_name in dof_names:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is not None:
        limit = joint.find('limit')
        if limit is not None:
            lower = float(limit.get('lower', '0'))
            upper = float(limit.get('upper', '0'))
            effort = float(limit.get('effort', '0'))
            velocity = float(limit.get('velocity', '0'))
            
            dof_pos_lower_limit_list.append(lower)
            dof_pos_upper_limit_list.append(upper)
            dof_vel_limit_list.append(velocity)
            dof_effort_limit_list.append(effort)
        else:
            print(f"joint '{joint_name}' lack 'limit' element.")
    else:
        print(f"didn't find '{joint_name}'.")

print("dof_pos_lower_limit_list:", dof_pos_lower_limit_list)
print("dof_pos_upper_limit_list:", dof_pos_upper_limit_list)
print("dof_vel_limit_list:", dof_vel_limit_list)
print("dof_effort_limit_list:", dof_effort_limit_list)
