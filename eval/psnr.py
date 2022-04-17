import torch
from model.insitu import Generator
from data.deep_water import DeepWaterDataset
from data.normalize import denormalize

from torch.utils.data import DataLoader
import math
import numpy as np

device = "cuda:0"

checkpoint = torch.load('/home/vida/tvsr/storage/insitu_deep_water_raw/model_60_0.002350_0.000127.pth.tar')
g_model = Generator(5,512,32)
g_model.load_state_dict(checkpoint['g_model_state_dict'])
g_model.to(device)

test_dataset = DeepWaterDataset('/home/vida/tvsr/2018_split','test.pkl')
kwargs = {"num_workers": 2, "pin_memory": True}
test_loader =  DataLoader(test_dataset, batch_size=20, **kwargs)

g_model.eval()
test_psnr = 0.

def mse_loss(truth,pred):
    return np.mean((truth-pred)**2)

with torch.no_grad():
    for i, sample in enumerate(test_loader):
        fields = sample["out"].to(device)
        sparams = torch.cat([sample['in']['burst'],
                                sample['in']['diameter'],
                                sample['in']['angle'],
                                sample['in']['z'],
                                sample['in']['time']],1).to(device)

        fake_fields = g_model(sparams)

        fields = fields.cpu().numpy()
        fields = denormalize(fields,0.00387,2.32885)
        fake_fields = fake_fields.cpu().numpy()
        fake_fields = denormalize(fake_fields,0.00387,2.32885)

        for i in range(20):
            single_field = fields[i][0]
            single_fake_field = fake_fields[i][0]
            test_psnr += 20*np.log10(2.32885 - 0.00387) - 10 *np.log10(mse_loss(single_field,single_fake_field))

    test_psnr = test_psnr / len(test_dataset)

print(test_psnr)