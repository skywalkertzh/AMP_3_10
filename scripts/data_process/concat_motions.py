import argparse
from pathlib import Path
import joblib

def concat_motions(motions_data):
    concat_motion_data = {}
    for motion_data in motions_data:
        for key, value in motion_data.items():
            concat_motion_data[key] = value
    return concat_motion_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, default="/home/jiawei/Research/humanoid/RoboVerse/roboverse/data/motions/h1_2_19dof/accad_loco")
    args = parser.parse_args()

    motion_dir = Path(args.motion_path)

    singles_motion_dir = motion_dir / "singles"
    concat_motion_dir = motion_dir / "concat"
    concat_motion_dir.mkdir(exist_ok=True)
    motions_data = []
    for motion_file in singles_motion_dir.glob("*.pkl"):
        motion_data = joblib.load(motion_file)
        motions_data.append(motion_data)

    concat_motion_data = concat_motions(motions_data)

    concat_motion_name = motion_dir.name + "_concat"
    concat_motion_path = concat_motion_dir / (concat_motion_name + ".pkl")
    joblib.dump(concat_motion_data, concat_motion_path)

