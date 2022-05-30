import torch
import torch.nn as nn
import torch.nn.functional as F

from .resblock import BasicBlockGenerator
from .resblock import FirstBlockDiscriminator, BasicBlockDiscriminator

class InsituG(nn.Module):
  def __init__(self, dsp=1, dvp=3, dspe=512, dvpe=512, ch=64):
    super(InsituG, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.dvp, self.dvpe = dvp, dvpe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe + dvpe, ch * 16 * 4 * 4, bias=False)
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1),
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(ch),
      nn.ReLU(),
      nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
      nn.Tanh()
    )

  def forward(self, sp, vp):
    sp = self.sparams_subnet(sp)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp, vp), 1)
    mp = self.mparams_subnet(mp)

    x = mp.view(mp.size(0), self.ch * 16, 4, 4)
    x = self.img_subnet(x)

    return x


class InsituD(nn.Module):
  def __init__(self, dsp=1, dvp=3, dspe=512, dvpe=512, ch=64):
    super(InsituD, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.dvp, self.dvpe = dvp, dvpe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe + dvpe, ch * 16),
      nn.ReLU()
    )

    # image classification subnet
    self.img_subnet = nn.Sequential(
      FirstBlockDiscriminator(3, ch, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3,
                              stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1, downsample=False),
      nn.ReLU()
    )

    # output subnets
    self.out_subnet = nn.Sequential(
      nn.Linear(ch * 16, 1)
    )

  def forward(self, sp, vp, x):
    sp = self.sparams_subnet(sp)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp, vp), 1)
    mp = self.mparams_subnet(mp)

    x = self.img_subnet(x)
    x = torch.sum(x, (2, 3))

    out = self.out_subnet(x)
    out += torch.sum(mp * x, 1, keepdim=True)

    return out
