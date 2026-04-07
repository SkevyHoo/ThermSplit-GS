
import torch
import torch.nn as nn

from utils.general_utils import inverse_sigmoid

# inverse_sigmoid : (0, 1) --> (-inf, +inf)

class LowerBoundSigmoid(nn.Module):  #  (-inf, +inf) --> (lb , 1)
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.sigmoid(x) * (1.0 - lb) + lb
    
class InverseLowerBoundSigmoid(nn.Module): # (lb , 1) --> (-inf, +inf)
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return inverse_sigmoid((x - lb) / (1.0 - lb) )
    
class Clamp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # alias.
        return x.clamp(0.0,1.0)
    
class InverseClamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(0.0,1.0)

class LowerBoundExponent(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.exp(x)+lb

class LowerBoundLog(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.eps = 0.001

    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.log( (x-lb).clamp_min(self.eps) )

class BoundSigmoid(nn.Module):  # Sigmoid曲线被缩放和平移，值域从 [0,1] 变为 [lb,ub]。
    def __init__(self, lb, ub):
        super().__init__()
        self.lb, self.ub = lb, ub
    
    def forward(self, x):
        # alias.
        lb, ub = self.lb, self.ub

        return torch.sigmoid(x) / (ub-lb) + lb
    
class InverseBoundSigmoid(nn.Module): # 是 BoundSigmoid 的反函数，将 [lb,ub] 映射回实数域。
    def __init__(self,lb, ub):
        super().__init__()
        self.lb, self.ub = lb, ub
        self.eps = (ub-lb)*0.001
    
    def forward(self, x):
        # alias.
        lb, ub = self.lb, self.ub

        return inverse_sigmoid( ((x-lb) * (ub-lb)).clamp(self.eps, 1.0-self.eps))

class InverseSoftplus(nn.Module): #对 x≥20 是线性函数 y=x，对 x<20 是 log(exp(x)-1)。
# 在 x=0 附近平滑过渡到负值。
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ret = torch.zeros_like(x)
        ret[x>=20] = x[x>=20]
        ret[x<20] = torch.log(torch.expm1(x[x<20]))
        return ret
