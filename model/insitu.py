import torch
import torch.nn as nn
from .resblock import GenResBlock,DisResBlock,OptimDisResBlock
from .self_mod import SelfModBatchNorm

class Generator(nn.Module):
    def __init__(self,dsp,dspe,ch,hidden_nums):
        super(Generator,self).__init__()
        self.dsp,self.dspe = dsp,dspe
        self.ch = ch

        self.sp_sub = nn.Sequential(
            nn.Linear(dsp,dspe),nn.ReLU(),
            nn.Linear(dspe,dspe),nn.ReLU(),
        )

        self.linear0 = nn.Linear(dspe,ch * 16 * 4 * 4)

        self.blocks = nn.ModuleList([
            GenResBlock(ch * 16,ch * 16,hidden_nums),
            GenResBlock(ch * 16,ch * 8,hidden_nums),
            GenResBlock(ch * 8,ch * 8,hidden_nums),
            GenResBlock(ch * 8,ch * 4,hidden_nums),
            GenResBlock(ch * 4,ch * 2,hidden_nums),
            GenResBlock(ch * 2,ch,hidden_nums),
        ])

        self.smbn0 = SelfModBatchNorm(512,ch,hidden_nums)

        self.out_sub = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch,1,3,1,1),
            nn.Tanh()
        )

    def forward(self,sp):
        sp = self.sp_sub(sp)
        z = sp
        out = self.linear0(sp)
        out = out.view(out.size(0),self.ch * 16,4,4)

        for m in self.blocks:
            out = m(out,z)

        out = self.smbn0(out,z)
        out = self.out_sub(out)

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
