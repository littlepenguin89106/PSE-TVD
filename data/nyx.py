from torch.utils.data import Dataset
import pathlib
import numpy as np
from skimage import io
from .normalize import normalize
import pandas as pd
import torch

def convert_theta(s):
    theta = float(s)
    return np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))

def convert_phi(s):
    phi = float(s)
    return phi / 90

def convert_sparam(s,d_min,d_max):
    val = float(s)
    return normalize(val,d_min,d_max)

def to_tensor(arr):
    if not isinstance(arr,np.ndarray):
        arr = np.array(arr,dtype="float32")
    return torch.from_numpy(arr)

class NyxDataset(Dataset):

    def __init__(self,data_path, filename):
        self.size = 0
        self.data_path = data_path
        self.read_file(filename)

    def convert_sparam(self,s,d_min,d_max):
        val = float(s)
        return normalize(val,d_min,d_max)

    def __getitem__(self, index):
        index = index % self.size
        
        s_OmM = self.label_list.iloc[index]['OmM']
        OmM = convert_sparam(s_OmM,0.21554,0.27766)

        s_OmB = self.label_list.iloc[index]['OmB']
        OmB = convert_sparam(s_OmB,0.04523,0.04571)

        s_h = self.label_list.iloc[index]['h']
        h = convert_sparam(s_h,0.61384,0.82706)

        s_time = self.label_list.iloc[index]['time']
        time = convert_sparam(s_time,0,200)

        s_theta = self.label_list.iloc[index]['theta']
        theta = convert_theta(s_theta)
        
        s_phi = self.label_list.iloc[index]['phi']
        phi = convert_phi(s_phi)

        sparams = to_tensor([OmM,OmB,h,time])

        vparams = to_tensor([*theta,phi])
        
        image_path = pathlib.Path(self.data_path,f'{s_OmM}_{s_OmB}_{s_h}',f'{s_time}_{s_theta}_{s_phi}.png')
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