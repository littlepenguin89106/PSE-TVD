import fire
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
        images = f.readlines()
    random.shuffle(images)

    total_num = len(images)

    test_num = total_num * ratio
    sample_range = total_num // test_num

    count = 0
    test_list = []
    train_list = []

    for i in images:
        i_parts = Path(i.strip()).parts
        m_params = i_parts[-2]
        i_params = i_parts[-1].split('_')

        param_dict = {
            'burst': m_params[1],
            'asteriod': m_params[2],
            'angle': m_params[3],
            'time': i_params[0].rstrip('.png')
        }

        if count % sample_range == 0:
            test_list.append(param_dict)
        else:
            train_list.append(param_dict)

        count += 1
            
    write_list(test_list,"test.pkl")
    write_list(train_list,"train.pkl")

if __name__ == "__main__":
    fire.Fire()