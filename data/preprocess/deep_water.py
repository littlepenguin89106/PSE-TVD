import fire
import pyvista
from pathlib import Path
import scipy.ndimage
import numpy as np
import pandas as pd

# for already split ensemble data
def split_dataset(ens_path, ratio):

    def write_list(arr,filename):
        df = pd.DataFrame(arr)
        df.to_pickle(filename)

    ensemble = [Path(ens_path,m) for m in ['yA31','yB31','yC31']]

    total_num = 0
    for m in ensemble:
        total_num += len(m.iterdir())

    test_num = total_num * ratio
    sample_range = total_num // test_num

    count = 0
    test_list = []
    train_list = []

    for m in ensemble:
        m_params = m.stem

        for f in m:
            f_params = f.stem.split('_')
            param_dict = {
                'burst': m_params[0],
                'asteriod': m_params[1],
                'angle': m_params[2],
                'axis_z': f_params[0],
                'time': f_params[1]
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


if __name__ == "__main__":
    fire.Fire()