import torch
import torch.nn as nn
from .resblock import GenResBlock,DisResBlock,OptimDisResBlock

class Generator(nn.Module):
    def __init__(self,dsp,dspe,ch):
        super(Generator,self).__init__()
        self.dsp,self.dspe = dsp,dspe
        self.ch = ch

        self.sp_sub = nn.Sequential(
            nn.Linear(dsp,dspe),nn.ReLU(),
            nn.Linear(dspe,dspe),nn.ReLU(),
            nn.Linear(dspe,ch * 16 * 4 * 4)
        )

        self.blocks = nn.Sequential(
            GenResBlock(ch * 16,ch * 16),
            GenResBlock(ch * 16,ch * 8),
            GenResBlock(ch * 8,ch * 8),
            GenResBlock(ch * 8,ch * 4),
            GenResBlock(ch * 4,ch * 2),
            GenResBlock(ch * 2,ch),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch,1,3,1,1),
            nn.Tanh()
        )

    def forward(self,sp):
        sp = self.sp_sub(sp)
        sp = sp.view(sp.size(0),self.ch * 16,4,4)
        out = self.blocks(sp)

        return out

class Discriminator(nn.Module):
    def __init__(self,dsp,dspe,ch):
        super(Discriminator,self).__init__()
        self.dsp,self.dspe = dsp,dspe
        self.ch = ch

        self.sp_sub = nn.Sequential(
            nn.Linear(dsp,dspe),nn.ReLU(),
            nn.Linear(dspe,dspe),nn.ReLU(),
            nn.Linear(dspe,ch * 16)
        )

        self.blocks = nn.Sequential(
            OptimDisResBlock(1,ch),
            DisResBlock(ch,ch*2),
            DisResBlock(ch*2,ch*4),
            DisResBlock(ch*4,ch*8),
            DisResBlock(ch*8,ch*8),
            DisResBlock(ch*8,ch*16),
            DisResBlock(ch*16,ch*16),
            nn.ReLU()
        )

        self.out_sub = nn.Linear(ch*16,1)
    
    def forward(self,sp,sf):
        sp = self.sp_sub(sp)

        sf = self.blocks(sf)
        sf = torch.sum(sf,(2,3))

        out = self.out_sub(sf)
        out += torch.sum(sp * sf,1,keepdim=True)

        return out        
