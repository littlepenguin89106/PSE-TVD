import fire
import pyvista
from pathlib import Path
import scipy.ndimage
import numpy as np
import pandas as pd
import random

random.seed(0)

# for already split ensemble data
def split_dataset(select_file, ratio):

    def write_list(arr,filename):
        df = pd.DataFrame(arr)
        df.to_pickle(filename)

    with open(select_file,'r') as f:
        slices = f.readlines()
    random.shuffle(slices)

    total_num = len(slices)

    test_num = total_num * ratio
    sample_range = total_num // test_num

    count = 0
    test_list = []
    train_list = []

    for s in slices:
        s_parts = Path(s.strip()).parts
        m_params = s_parts[-2]
        s_params = s_parts[-1].split('_')

        param_dict = {
            'burst': m_params[1],
            'asteriod': m_params[2],
            'angle': m_params[3],
            'axis_z': s_params[0],
            'time': s_params[1].rstrip('.bin')
        }

        if count % sample_range == 0:
            test_list.append(param_dict)
        else:
            train_list.append(param_dict)

        count += 1
            
    write_list(test_list,"test.pkl")
    write_list(train_list,"train.pkl")


# for single member
def split_x(src_path, dst_path, scalar, log_file):
    files = sorted(Path(src_path).iterdir())

    for f in files:
        print(f)
        try:
            dataset = pyvista.read(f)
            raw_data = dataset.point_data[scalar]
            raw_data = raw_data.reshape(300, 300, 300)
            org_name = f.stem
            str_time = org_name.split('_')[3]

            for x in range(300):
                # Resize the raw data from (500, 500) to (256, 256)
                resize_data = scipy.ndimage.interpolation.zoom(
                    raw_data[x, :, :], 256 / 300)

                # Write a file
                resize_path = Path(dst_path, f'{x}_{str_time}.bin')
                resize_data.tofile(resize_path)
        except:
            with open(log_file, 'a') as rf:
                rf.write(f'{f}\n')

def cal_min_max(ens_path):
    data_max = -float('inf')
    data_min = float('inf')
    ensemble = Path(ens_path).iterdir()

    for m in ensemble:
        for f in m.iterdir():
            data = np.fromfile(f)
            if(data_max < np.max(data)):
                data_max = np.max(data)
            if(data_min > np.min(data)):
                data_min = np.min(data)

    with open('deep_water_min_max.txt','w') as f:
        f.write(f'min:{data_min} max:{data_max}')

if __name__ == "__main__":
    fire.Fire()