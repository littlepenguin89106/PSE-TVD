import random
from pathlib import Path
import zipfile
import numpy as np

random.seed(0)

def random_slice(src_path):
    ensemble = sorted(Path(src_path).iterdir())
    with open("select.txt",'w') as f:
        for m in ensemble:
            slices = list(m.iterdir()).sort()
            select_indices = random.sample(slices,len(slices)//3)
            for s in select_indices:
                f.write(s)

def zipFiles(select_file):
    with zipfile.ZipFile("2018_split.zip",'w') as zf:
        
        with open(select_file,'r') as f:
            for line in f:
                zf.write(line)

def cal_min_max(select_file):
    data_max = -float('inf')
    data_min = float('inf')
    with open(select_file,'r') as f:
        for line in f:
            line = line.strip()
            data = np.fromfile(line)
            if(data_max < np.max(data)):
                data_max = np.max(data)
            if(data_min > np.min(data)):
                data_min = np.min(data)

    with open('deep_water_min_max.txt','w') as f:
        f.write(f'min:{data_min} max:{data_max}')