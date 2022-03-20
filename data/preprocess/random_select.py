import random
from pathlib import Path
import zipfile
import fire
import numpy as np
import pyvista

random.seed(0)

def random_slice(src_path):
    ensemble = sorted(Path(src_path).iterdir())
    with open("select.txt",'w') as f:
        for m in ensemble:
            slices = sorted(list(m.iterdir()))

            select_indices = random.sample(slices,15000)
            for s in select_indices:
                f.write(f'{s}\n')
            print(m)

def zipFiles(select_file):
    with zipfile.ZipFile("2018_split.zip",'w') as zf:
        
        with open(select_file,'r') as f:
            for line in f:
                line = line.strip()
                zf.write(line,Path(*line.split('/')[-3:]))


if __name__ == "__main__":
    fire.Fire()