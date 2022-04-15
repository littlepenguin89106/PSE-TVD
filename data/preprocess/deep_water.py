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
        fname_parts = Path(i.strip()).stem.split('_')
        mparams = fname_parts[0]

        param_dict = {
            'burst': mparams[1],
            'diameter': mparams[2],
            'angle': mparams[3],
            'time': fname_parts[1],
            'theta': fname_parts[2],
            'phi': fname_parts[3]
        }

        if count % sample_range == 0:
            test_list.append(param_dict)
        else:
            train_list.append(param_dict)

        count += 1
            
    write_list(test_list,"test.pkl")
    write_list(train_list,"train.pkl")

def test_dataframe(path):
    df = pd.read_pickle(path)
    print(df)

if __name__ == "__main__":
    fire.Fire()