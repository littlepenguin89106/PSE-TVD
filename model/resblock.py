import torch.nn as nn
import torch.nn.functional as F 

class GenResBlock(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(GenResBlock,self).__init__()

        self.res_conv = nn.Conv2d(in_chs,out_chs,1,1,0,bias=False)
        self.main0 = nn.Sequential(
            nn.BatchNorm2d(in_chs),
            nn.ReLU(),
        )

        self.main1 = nn.Sequential(
            nn.Conv2d(in_chs,out_chs,3,1,1,bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs,out_chs,3,1,1,bias=False)
        )

    def forward(self, x):
        r = x
        r = F.interpolate(r,scale_factor=2)
        r = self.res_conv(r)

        out = self.main0(x)
        out = F.interpolate(out,scale_factor=2)
        out = self.main1(out)

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
