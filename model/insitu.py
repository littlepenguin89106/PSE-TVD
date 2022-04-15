import torch
import torch.nn as nn
from .resblock import GenResBlock,DisResBlock,OptimDisResBlock

class Generator(nn.Module):
    def __init__(self,dsp,dspe,dvp,dvpe,ch):
        super(Generator,self).__init__()
        self.dsp,self.dspe = dsp,dspe
        self.ch = ch

        self.sp_sub = nn.Sequential(
            nn.Linear(dsp,dspe),nn.ReLU(),
            nn.Linear(dspe,dspe),nn.ReLU(),
        )

        self.vp_sub = nn.Sequential(
            nn.Linear(dvp,dvpe),nn.ReLU(),
            nn.Linear(dvpe,dvpe),nn.ReLU(),
        )

        self.mp_sub = nn.Linear(dspe+dvpe, ch * 16 * 4 * 4, bias=False)

        self.blocks = nn.Sequential(
            GenResBlock(ch * 16,ch * 16),
            GenResBlock(ch * 16,ch * 8),
            GenResBlock(ch * 8,ch * 8),
            GenResBlock(ch * 8,ch * 4),
            GenResBlock(ch * 4,ch * 2),
            GenResBlock(ch * 2,ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch,3,3,1,1),
            nn.Tanh()
        )

    def forward(self,sp,vp):
        sp = self.sp_sub(sp)
        vp = self.vp_sub(vp)

        mp = torch.cat((sp,vp),1)
        mp = self.mp_sub(mp)
        mp = mp.view(mp.size(0),self.ch * 16,4,4)

        out = self.blocks(mp)

        return out

class Discriminator(nn.Module):
    def __init__(self,dsp,dspe,dvp,dvpe,ch):
        super(Discriminator,self).__init__()
        self.dsp,self.dspe = dsp,dspe
        self.ch = ch

        self.sp_sub = nn.Sequential(
            nn.Linear(dsp,dspe),nn.ReLU(),
            nn.Linear(dspe,dspe),nn.ReLU(),
        )

        self.vp_sub = nn.Sequential(
            nn.Linear(dvp,dvpe),nn.ReLU(),
            nn.Linear(dvpe,dvpe),nn.ReLU(),
        )

        self.mp_sub = nn.Linear(dspe + dvpe,16 * ch)

        self.blocks = nn.Sequential(
            OptimDisResBlock(3,ch),
            DisResBlock(ch,ch*2),
            DisResBlock(ch*2,ch*4),
            DisResBlock(ch*4,ch*8),
            DisResBlock(ch*8,ch*8),
            DisResBlock(ch*8,ch*16),
            DisResBlock(ch*16,ch*16),
            nn.ReLU()
        )

        self.out_sub = nn.Linear(ch*16,1)
    
    def forward(self,sp,vp,x):
        sp = self.sp_sub(sp)

        vp = self.vp_sub(vp)

        mp = torch.cat((sp,vp),1)
        mp = self.mp_sub(mp)

        x = self.blocks(x)
        x = torch.sum(x,(2,3))

        out = self.out_sub(x)
        out += torch.sum(mp * x,1,keepdim=True)

        return out        
