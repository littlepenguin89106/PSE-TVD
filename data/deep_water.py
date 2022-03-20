from torch.utils.data import Dataset
import pathlib
import numpy as np
from .normalize import normalize
import pandas as pd
import torch

def transfer_airburst(s):
    if s == "A":
        return 0
    elif s == "B":
        return 5
    elif s == "C":
        return 10
    elif s == "D":
        return 15

def transfer_asteriod(s):
    if s=="1":
        return 100
    elif s=="3":
        return 250
    elif s=="5":
        return 500

def transfer_angle(s):
    if s=="0":
        return 27.4
    elif s=="1":
        return 45
    elif s=="2":
        return 60

def convert_tensor(arr):
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr,dtype="float32")
    return torch.from_numpy(arr)

class DeepWaterDataset(Dataset):

    def __init__(self,data_path, filename):
        self.size = 0
        self.data_path = data_path
        self.read_file(filename)

    def __getitem__(self, index):
        index = index % self.size
        
        s_burst = self.label_list.iloc[index]['burst']
        burst = transfer_airburst(s_burst)
        burst = normalize(burst,0,15)
        burst = convert_tensor([burst])

        s_asteriod = self.label_list.iloc[index]['asteriod']
        asteriod = transfer_asteriod(s_asteriod)
        asteriod = normalize(asteriod,100,500)
        asteriod = convert_tensor([asteriod])
        
        s_angle = self.label_list.iloc[index]['angle']
        angle = transfer_angle(s_angle)
        angle = normalize(angle,27.4,60)
        angle = convert_tensor([angle])
        
        s_z = self.label_list.iloc[index]['axis_z']
        z = normalize(int(s_z),0,299)
        z = convert_tensor([z])
        
        s_time = self.label_list.iloc[index]['time']
        time = normalize(int(s_time),0,220062)
        time = convert_tensor([time])
        
        field_path = pathlib.Path(self.data_path,f'y{s_burst}{s_asteriod}{s_angle}',f'{s_z}_{s_time}.bin')
        field = self.data_preproccess(field_path)
        field = convert_tensor(field)

        params = {'burst':burst,'angle':angle,'z':z,'time':time}

        sample = {'in': params,'out':field}

        return sample

    def __len__(self):
        return self.size

    def read_file(self, filename):
        file_path = pathlib.Path(self.data_path,filename)
        self.label_list = pd.read_pickle(file_path)
        self.size = len(self.label_list)

    def data_preproccess(self, data_path):
        raw_data = np.fromfile(data_path, dtype='float32')
        raw_data = normalize(raw_data,-2.239696925053734e-07,7.96058750165821)
        raw_data = raw_data.reshape(1, 256, 256)

        return raw_data