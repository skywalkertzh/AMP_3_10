import xml.etree.ElementTree as ET
import numpy as np
def extract_actuatorfrcrange_from_file(file_path):
    """
    Extracts all joint names and their actuatorfrcrange from an XML file.
    
    Parameters:
        file_path (str): The path to the XML file.

    Returns:
        list: A list of tuples, where each tuple contains the joint name and its actuatorfrcrange.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # List to store joint names and actuatorfrcrange
    joint_ranges = []
    
    # Iterate over all 'joint' elements in the XML
    for joint in root.findall(".//joint"):
        name = joint.get('name')
        actuatorfrcrange = joint.get('actuatorfrcrange')
        
        # Append the joint name and actuatorfrcrange to the list
        if name and actuatorfrcrange:
            joint_ranges.append((name, actuatorfrcrange))
    
    return joint_ranges


def extract_posfrcrange_from_file(file_path):
    """
    Extracts all joint names and their actuatorfrcrange from an XML file.
    
    Parameters:
        file_path (str): The path to the XML file.

    Returns:
        list: A list of tuples, where each tuple contains the joint name and its actuatorfrcrange.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # List to store joint names and actuatorfrcrange
    joint_ranges = []
    
    # Iterate over all 'joint' elements in the XML
    for joint in root.findall(".//joint"):
        name = joint.get('name')
        actuatorfrcrange = joint.get('range')
        
        # Append the joint name and actuatorfrcrange to the list
        if name and actuatorfrcrange:
            joint_ranges.append((name, actuatorfrcrange))
        else:
            joint_ranges.append((name, "-1000 1000"))
    
    return joint_ranges

# Example usage
xml_file_path = "/media/tairanh/lecar4tb/tairanh/Workspace/RoboVerse/roboverse/data/robots/ea/ea.xml"  # Replace with your XML file path

# Extract the actuatorfrcrange
joint_actuator_ranges = extract_actuatorfrcrange_from_file(xml_file_path)

joint_pos_ranges = extract_posfrcrange_from_file(xml_file_path)

correct_joint_order = ["back.bkl", "back.bkr",
              "head.nkz",
              "larm.wbd","larm.ft","lleg.aki","lleg.kn_act_out","lleg.ako","larm.el","lleg.hpx","larm.shx","larm.wbp","larm.shz","lleg.hpy","lleg.hpz","larm.shy",
              "back.bkz",
              "rarm.wbd","rarm.ft","rleg.aki","rleg.kn_act_out","rleg.ako","rarm.el","rleg.hpx","rarm.shx","rarm.wbp","rarm.shz","rleg.hpy","rleg.hpz","rarm.shy"]

default_joint_angles = [ 0.000159516,
    0,
    0,
    0,
    0,
    0.00016108,
    0,
    0,
    0,
    0,
    0,
    0.42,
    0.21,
    0.089,
    0.88,
    -0.012,
    0.108,
    0.04,
    0,
    0,
    0,
    -0.104984,
    -0.000324603,
    -0.0279778,
    -0.0845888,
    -0.114996,
    0.158608,
    -0.00178125,
    -0.000702461,
    0.00268862,
    0.0080953,
    0.0295835,
    -0.00181072,
    0.0007798,
    0.00272783,
    -0.00732946,
    -0.0316949,
    -0.0926288,
    -0.0011431,
    -0.105049,
    -0.0012566,
    -0.0278992,
    -0.0847225,
    -0.115157,
    0.158808,
    -0.00179576,
    0.000735175,
    0.00270815,
    -0.00778315,
    -0.030501,
    -0.00180064,
    -0.00074798,
    0.00271465,
    0.00765631,
    0.0308506,
    -0.0927389,
    -0.000189268,
    0.42,
    0.21,
    0.089,
    0.88,
    -0.012,
    0.108,
    0.04]


# Print the results
for joint_name, range_value in joint_actuator_ranges:
    print(f"Joint: {joint_name}, Actuator Force Range: {range_value}")

# Print the results based on the correct joint order
correct_joint_order_effort_list = []
correct_joint_order_pos_list_lower = []
correct_joint_order_pos_list_upper = []
for joint_name in correct_joint_order:
    for joint, range_value in joint_actuator_ranges:
        if joint == joint_name:
            print(f"Joint: {joint}, Actuator Force Range: {range_value}")

            # append
            correct_joint_order_effort_list.append(float(range_value.split(' ')[-1]))

            break

    else:
        print(f"Joint: {joint_name}, Actuator Force Range: Not Found")
    
    for joint, range_value in joint_pos_ranges:
        if joint == joint_name:
            # print(f"Joint: {joint}, Pos Range: {range_value}")
            pos_range = range_value.split(' ')
            correct_joint_order_pos_list_lower.append(float(pos_range[0]))
            correct_joint_order_pos_list_upper.append(float(pos_range[1]))
            break
    else:
        pass
        # print(f"Joint: {joint_name}, Pos Range: Not Found")

print("correct_joint_order_effort_list:", correct_joint_order_effort_list)
print("correct_joint_order_pos_list_lower:", correct_joint_order_pos_list_lower)
print("correct_joint_order_pos_list_upper:", correct_joint_order_pos_list_upper)


# ipdb> self._mj_model.jnt_qposadr[joint_ids]
# array([ 9, 14, 17, 24, 22, 36, 31, 41, 21, 29, 19, 23, 20, 28, 30, 18, 27,
#        70, 68, 54, 49, 59, 67, 47, 65, 69, 66, 46, 48, 64], dtype=int32)
actuated_qpos_adr = np.array([ 9, 14, 17, 24, 22, 36, 31, 41, 21, 29, 19, 23, 20, 28, 30, 18, 27,
       70, 68, 54, 49, 59, 67, 47, 65, 69, 66, 46, 48, 64])
actuated_joint_idx = actuated_qpos_adr - 7
# print joint names and default joint angles with the following forat: joint_name: default_joint_angle
# not that the default_joint angle is qpos[7:7+num_dof] in the mujoco model
for idx, joint_idx in enumerate(actuated_joint_idx):
    print(f"{correct_joint_order[idx]} : {default_joint_angles[joint_idx]}")

# import ipdb; ipdb.set_trace()


kp_list = [
    254.64790891378243, 254.64790891378243,
    -0.031290482061166636,
    -0.31290482061166636, -0.31290482061166636,
    157.07963264164164, -59.09090998622591, 157.07963264164164,
    -8.062828787040132, -40.31414393520066,
    20.15707196760033, 0.31290482061166636,
    -16.125657574080265, -98.48484997704318,
    78.78787998163455, -16.125657574080265,
    9.848484997704318,
    -0.31290482061166636, 0.31290482061166636,
    157.07963264164164, 59.09090998622591, 157.07963264164164,
    -8.062828787040132, -40.31414393520066,
    20.15707196760033, -0.31290482061166636,
    16.125657574080265, 98.48484997704318,
    -78.78787998163455, 16.125657574080265
]

kd_list = [
    2.5464790891378244, 2.5464790891378244,
    -0.0,
    -0.18774289236699981, -0.18774289236699981,
    1.2732395445689122, -0.9848484997704319, 1.2732395445689122,
    -0.2015707196760033, -0.2015707196760033,
    0.2015707196760033, 0.18774289236699981,
    -0.2015707196760033, -0.9848484997704319,
    0.9848484997704319, -0.2015707196760033,
    0.49242424988521594,
    -0.18774289236699981, 0.18774289236699981,
    1.2732395445689122, 0.9848484997704319, 1.2732395445689122,
    -0.2015707196760033, -0.2015707196760033,
    0.2015707196760033, -0.18774289236699981,
    0.2015707196760033, 0.9848484997704319,
    -0.9848484997704319, 0.2015707196760033
]

print("---------------------------------    ")
# print the kp and kd values for each joint following the format of joint_name: kp_value with all the digits
for idx, joint_name in enumerate(correct_joint_order):
    # print(idx)
    print(f"{joint_name}: {abs(kp_list[idx])}")
# import ipdb; ipdb.set_trace()


print("---------------------------------    ")
# print the kp and kd values for each joint following the format of joint_name: kd_value with all the digits
for idx, joint_name in enumerate(correct_joint_order):
    # print(idx)
    print(f"{joint_name}: {abs(kd_list[idx])}")