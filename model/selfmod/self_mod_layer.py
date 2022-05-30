import torch.nn as nn

class SelfModBatchNorm(nn.Module):
    def __init__(self,in_nums,out_nums,hidden_nums):
        super(SelfModBatchNorm,self).__init__()

        self.out_nums = out_nums
        self.bn = nn.BatchNorm2d(out_nums,affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_nums,hidden_nums),
            nn.ReLU()
        )
        self.gamma = nn.Linear(hidden_nums,out_nums)
        self.beta = nn.Linear(hidden_nums,out_nums)

    def forward(self,x,z):
        x = self.bn(x)
        
        z = self.mlp(z)
        g = self.gamma(z)
        g = g.view(-1,self.out_nums,1,1)
        b = self.beta(z)
        b = b.view(-1,self.out_nums,1,1)

        return g*x + b

