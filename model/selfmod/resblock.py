import torch.nn as nn
import torch.nn.functional as F
from .self_mod_layer import SelfModBatchNorm

class GenResBlock(nn.Module):
    def __init__(self,in_chs,out_chs,ze,hidden_chs):
        super(GenResBlock,self).__init__()

        self.res_conv = nn.Conv2d(in_chs,out_chs,1,1,0,bias=False)
        self.smbn0 = SelfModBatchNorm(ze,in_chs,hidden_chs)
        self.conv0 = nn.Conv2d(in_chs,out_chs,3,1,1,bias=False)
        self.smbn1 = SelfModBatchNorm(ze,out_chs,hidden_chs)
        self.conv1 = nn.Conv2d(out_chs,out_chs,3,1,1,bias=False)

    def forward(self, x,z):
        r = x
        r = F.interpolate(r,scale_factor=2)
        r = self.res_conv(r)

        out = self.smbn0(x,z)
        out = F.relu(out)
        out = F.interpolate(out,scale_factor=2)
        out = self.conv0(out)

        out = self.smbn1(out,z)
        out = F.relu(out)
        out = self.conv1(out)

        return out + r

class OptimDisResBlock(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(OptimDisResBlock,self).__init__()

        self.res_conv = nn.Conv2d(in_chs,out_chs,1,1,0)
        self.main = nn.Sequential(
            nn.Conv2d(in_chs,out_chs,3,1,1),
            nn.ReLU(),
            nn.Conv2d(out_chs,out_chs,3,1,1)
        )

    def forward(self,x):
        r = self.res_conv(x)
        r = F.avg_pool2d(r,kernel_size = 2)
        
        out = self.main(x)
        out = F.avg_pool2d(out,kernel_size=2)

        return out + r

class DisResBlock(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(DisResBlock,self).__init__()

        self.res_conv = nn.Conv2d(in_chs,out_chs,1,1,0)
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_chs,out_chs,3,1,1),
            nn.ReLU(),
            nn.Conv2d(out_chs,out_chs,3,1,1),
        )

    def forward(self,x):
        r = x
        r = self.res_conv(r)
        r = F.avg_pool2d(r,kernel_size = 2)

        out = self.main(x)

        out = F.avg_pool2d(out, kernel_size =2)

        return out + r
