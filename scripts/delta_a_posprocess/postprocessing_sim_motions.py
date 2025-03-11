import joblib
import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, default=None)
    args = parser.parse_args()


    motion_path = args.motion_path
    data=joblib.load(motion_path)
    import ipdb; ipdb.set_trace() 
    # print(data.keys())
    for motion_key, motion_value in data.items():
        terminate = motion_value['terminate']    #[n, ]
        # truncate the other motion_values when terminate is True at some timestep 
        terminate_step = -1
        for i in range(len(terminate)):
            if terminate[i]:
                terminate_step = i
                break
        
        if terminate_step != -1:
            for key, value in motion_value.items():
                if key != 'fps':
                    motion_value[key] = value[:terminate_step]

    # import ipdb; ipdb.set_trace()

    new_motion_path = motion_path.replace(".pkl", "_truncated.pkl")
    joblib.dump(data, new_motion_path)
