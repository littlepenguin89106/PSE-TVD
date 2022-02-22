from torch.utils.data import Dataset
import pathlib
import torch
import numpy as np

def normalize(data,d_min,d_max):
    n_data = (data - d_min) / (d_max - d_min)
    return n_data * 2 - 1
    # [-1,1]

def denormalize(data,d_min,d_max):
    n_data = (data + 1) / 2
    return n_data * (d_max - d_min) + d_min


class IsabelDataset(Dataset):

    def __init__(self,data_path, filename):
        self.size = 0
        self.data_path = data_path
        self.field_label_list = self.read_file(filename)

    def __getitem__(self, index):
        index = index % self.size
        field_name = self.field_label_list[index][0]
        sp = self.field_label_list[index][1]
        field_path = pathlib.Path(self.data_path,field_name)
        field = self.data_preproccess(field_path)
        sample = {'f': torch.tensor(field), 'sp': torch.tensor(sp)}

        return sample

    def __len__(self):
        return self.size

    def read_file(self, filename):
        file_path = pathlib.Path(self.data_path,filename)
        field_label_list = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                field_name = content[0]
                sp = float(content[1])
                t = float(content[2])
                # Normalization
                sp = normalize(sp,0,99)
                t = normalize(t,1,48)

                field_label_list.append((field_name, [sp, t]))
                self.size += 1
        
        return field_label_list 

    def data_preproccess(self, data_path):
        raw_data = np.fromfile(data_path, dtype='float32')
        raw_data = normalize(raw_data,-5471.85791,3225.42578)
        raw_data = raw_data.reshape(1, 256, 256)

        return raw_data