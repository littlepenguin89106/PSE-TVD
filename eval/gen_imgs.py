import torch
from torch.utils.data.dataloader import DataLoader
from model.insitu import Generator
from skimage import io

from torch.utils.data import Dataset
import pathlib
import numpy as np
from data.normalize import normalize
import pandas as pd
from torchvision.utils import save_image

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
        
        image = pathlib.Path(self.data_path,f'y{s_burst}{s_diameter}{s_angle}',f'y{s_burst}{s_diameter}{s_angle}_{s_time}_{s_theta}_{s_phi}.png')
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

def collate_fn(data):
    return pd.DataFrame(data).to_dict('list')

def gen_imgs(check_filename,dataset_dirname,out_dir):

    device = 'cuda:0'
    checkpoint = torch.load(check_filename)

    dataset = DeepWaterDataset(dataset_dirname,'test.pkl')
    test_dataloader = DataLoader(dataset,batch_size=64,num_workers=6,pin_memory=True,collate_fn=collate_fn)

    g_model = Generator(4,512,3,512,32,128)
    g_model.load_state_dict(checkpoint['g_model_state_dict'])
    g_model.to(device)
    g_model.eval()

    with torch.no_grad():

        for sample in test_dataloader:
            sparams = torch.stack(sample['sparams']).to(device)
            vparams = torch.stack(sample['vparams']).to(device)
            truth_images = sample['img']
            gen_images = g_model(sparams,vparams)
            gen_images = (gen_images + 1) * 0.5

            for i in range(gen_images.size()[0]):
                img = gen_images[i]
                img_name = truth_images[i].name
                save_image(img,pathlib.Path(out_dir,img_name))
