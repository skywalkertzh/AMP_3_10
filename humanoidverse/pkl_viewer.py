import joblib
import json
import numpy as np

# 读取pkl文件
with open("humanoidverse/data/motions/kit_loco/singles/0-KIT_3_walking_slow08_poses.pkl", "rb") as f:
    data = joblib.load(f)
    import pdb; pdb.set_trace()
    a = data['0-KIT_3_walking_slow08_poses']
    print(a)

