from torch.utils.data import Dataset
import pathlib
import numpy as np
from skimage import io
from .normalize import normalize
import pandas as pd
import torch

def convert_airburst(s):
    if s == "A":
        return 0
    elif s == "B":
        return 5
    elif s == "C":
        return 10
    elif s == "D":
        return 15

def convert_diameter(s):
    if s=="1":
        return 100
    elif s=="3":
        return 250
    elif s=="5":
        return 500

def convert_angle(s):
    if s=="0":
        return 27.4
    elif s=="1":
        return 45
    elif s=="2":
        return 60

def convert_theta(s):
    theta = float(s)
    return np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))

def convert_phi(s):
    phi = float(s)
    return phi / 90

def to_tensor(arr):
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
        burst = convert_airburst(s_burst)
        burst = normalize(burst,0,15)

        s_diameter = self.label_list.iloc[index]['diameter']
        diameter = convert_diameter(s_diameter)
        diameter = normalize(diameter,100,500)
        
        s_angle = self.label_list.iloc[index]['angle']
        angle = convert_angle(s_angle)
        angle = normalize(angle,27.4,60)
        
        s_time = self.label_list.iloc[index]['time']
        time = normalize(int(s_time),0,49978)

        sparams = to_tensor([burst,diameter,angle,time])

        s_theta = self.label_list.iloc[index]['theta']
        theta = convert_theta(s_theta)
        s_phi = self.label_list.iloc[index]['phi']
        phi = convert_phi(s_phi)

        vparams = to_tensor([*theta,phi])
        
        image_path = pathlib.Path(self.data_path,f'y{s_burst}{s_diameter}{s_angle}',f'{s_time}_{s_theta}_{s_phi}.png')
        image = self.data_preproccess(image_path)
        image = to_tensor(image)

        sample = {'sparams':sparams,'vparams':vparams,'img':image}

        return sample

    def __len__(self):
        return self.size

    def read_file(self, filename):
        file_path = pathlib.Path(self.data_path,filename)
        self.label_list = pd.read_pickle(file_path)
        self.size = len(self.label_list)

    def data_preproccess(self, data_path):
        raw_data = io.imread(data_path).astype('float32')
        raw_data = normalize(raw_data,0,255)
        raw_data = raw_data[:,:,0:3].transpose((2,0,1))

        return raw_data