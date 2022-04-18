import fire
from pathlib import Path
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
        img_path = Path(i.strip())
        fname_parts = img_path.stem.split('_')
        mparams = img_path.parts[-2].split('_')

        param_dict = {
            'OmM': mparams[0],
            'OmB': mparams[1],
            'h': mparams[2],
            'time': fname_parts[0],
            'theta': fname_parts[1],
            'phi': fname_parts[2]
        }

        if count % sample_range == 0:
            test_list.append(param_dict)
        else:
            train_list.append(param_dict)

        count += 1
            
    write_list(test_list,"test.pkl")
    write_list(train_list,"train.pkl")

def write_render_list(dir_path,file_name):
    with open(file_name,'w') as f:
        for m in sorted(Path(dir_path).iterdir()):
            for i in sorted(m.iterdir()):
                f.write(f'{i}\n')

def collect_dataset_info(dir_path,select_list_filename,min_max_list_filename):
    d_max = -float("inf")
    d_min = float("inf")

    with open(select_list_filename,'w') as f:
        for m in Path(dir_path).iterdir():
            if m.is_dir():

                for t in (m/'density').iterdir():
                    f.write(f'{t}\n')
                    data = np.fromfile(t,dtype="float32")
                    d_max = max(d_max,np.max(data))
                    d_min = min(d_min,np.min(data))
    
    with open(min_max_list_filename,'w') as f:
        f.write(f'min:{d_min}\n')
        f.write(f'max:{d_max}')


def test_dataframe(path):
    df = pd.read_pickle(path)
    print(df)

if __name__ == "__main__":
    fire.Fire()