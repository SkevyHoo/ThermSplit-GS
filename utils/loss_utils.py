#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass


C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


# def subframe_l1_loss(subframes, gt):
#      subframes_num = subframes.shape[0]
#      loss = 0
#      for i in range(subframes_num):
#          loss += torch.abs((subframes[i].squeeze(0) - gt)).mean()
#      return loss/subframes_num

def subframe_l1_loss(subframes: torch.Tensor,
                     gt: torch.Tensor,
                     reduction: str = 'mean') -> torch.Tensor:

    # Compute L1 loss between multiple subframes and a ground truth frame.
    # Args:
    #     subframes (torch.Tensor): Tensor of shape (N, C, H, W) where N is number of subframes.
    #     gt (torch.Tensor): Ground truth tensor of shape (C, H, W) or (1, C, H, W).
    #     reduction (str): Reduction method - 'mean' (全局平均), 'sum' (全局求和) or 'none' (保留各元素差异).

    # Returns:
    #     torch.Tensor: Computed L1 loss.
    # """
    # 确保gt形状为 (C, H, W)
    gt = gt.squeeze(0) if gt.dim() == 4 and gt.shape[0] == 1 else gt
    abs_diff = torch.abs(subframes - gt.unsqueeze(0))  # shape (N, C, H, W)
    # 根据reduction参数处理
    if reduction == 'mean':
        return abs_diff.mean()  # 直接全局平均（等价于先按子帧平均再对子帧结果平均）
    elif reduction == 'sum':
        return abs_diff.sum()
    elif reduction == 'none':
        return abs_diff
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum' or 'none'.")


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def tv_loss(x:torch.Tensor):  # 鼓励图像的相邻像素值平滑变化
    """
    ARGUMENTS
    ---------
    x: torch.Tensor [b,c,h,w]

    RETURNS
    -------
    smoothness_loss torch.tensor (float) of size(1)
    """
    horizontal_loss = l2_loss(x[: , : , :-1 , :] , x[ : , : , 1 : , :])
    vertical_loss   = l2_loss(x[: , :,  : , :-1] , x[ : , : , : , 1: ])
    return horizontal_loss + vertical_loss

def batchwise_smoothness_loss(x:torch.Tensor):  #鼓励 batch 内相邻样本之间的平滑过渡
    """
    ARGUMENTS
    ---------
    x: torch.Tensor [b,3,h,w]

    RETURNS
    -------
    smoothness_loss (batch-side tv loss) torch.tensor (float) of size(1)
    """
    device= x.device
    if x.shape[0]==1:
        return torch.zeros(1, device=device)
    return l1_loss( x[1:], x[:-1] )


def hinge_l2(x:torch.Tensor):  # 结合了 hinge loss 和 L2 loss 的特性；鼓励 x 的值在 0 和 1 之间
    """
    hinge loss
    """
    loss = torch.zeros_like(x)
    
    loss[x<=0.0] = x[x<=0.0] ** 2
    loss[x>=1.0] = (x[x>=1.0] - 1.0) ** 2
    return loss.mean()

def Skewed_binaryzation_loss(x:torch.Tensor, k:int=1, eps:float=1e-15):  
    """
    binaryzation loss for opacity.
    motivated by the paper "D^2-NeRF"
    """
    # 对输入进行裁剪，确保在 [eps, 1-eps] 范围内
    x_clamped = torch.clamp(x, min=eps, max=1-eps)
    x = x_clamped ** k
    loss = -(x_clamped * torch.log(x_clamped) + (1 - x_clamped) * torch.log(1 - x_clamped))

    return loss.mean()


def hard_loss(x:torch.Tensor):  
    """
    hard loss for opacity.
    motivated by the paper "LOL-NeRF"
    """
    ip_w = torch.exp(-torch.abs(x)) + torch.exp(-torch.abs(1 - x))
    loss = -torch.log(ip_w + 1e-15)  # Add a small epsilon to avoid log(0)
    
    return loss.mean()


# 建模动态场景利用了 场景表征的低秩性,比如d-nerf

class GridGradientCentralDiff:
    # 'from BAD-GS'
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)

        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()

        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_

        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


class EdgeAwareVariationLoss(nn.Module):
    # 'from BAD-GS'
    def __init__(self, in1_nc=3, grad_fn=GridGradientCentralDiff):
        super(EdgeAwareVariationLoss, self).__init__()
        self.in1_grad_fn = grad_fn(in1_nc)
        # self.in2_grad_fn = grad_fn(in2_nc)

    def forward(self, in1, mean=False):
        in1_dx, in1_dy = self.in1_grad_fn(in1)
        # in2_dx, in2_dy = self.in2_grad_fn(in2)

        abs_in1_dx, abs_in1_dy = in1_dx.abs().sum(dim=1, keepdim=True), in1_dy.abs().sum(dim=1, keepdim=True)
        # abs_in2_dx, abs_in2_dy = in2_dx.abs().sum(dim=1,keepdim=True), in2_dy.abs().sum(dim=1,keepdim=True)

        weight_dx, weight_dy = torch.exp(-abs_in1_dx), torch.exp(-abs_in1_dy)

        variation = weight_dx * abs_in1_dx + weight_dy * abs_in1_dy

        if mean != False:
            return variation.mean()
        return variation.sum()


class GrayEdgeAwareVariationLoss(nn.Module):
    # 'from BAD-GS'
    def __init__(self, in1_nc=3, in2_nc=3, grad_fn=GridGradientCentralDiff):
        super(GrayEdgeAwareVariationLoss, self).__init__()
        self.in1_grad_fn = grad_fn(in1_nc)  # Gray
        self.in2_grad_fn = grad_fn(in2_nc)  # Sharp

    def forward(self, in1, in2, mean=False):
        in1_dx, in1_dy = self.in1_grad_fn(in1)
        in2_dx, in2_dy = self.in2_grad_fn(in2)

        abs_in1_dx, abs_in1_dy = in1_dx.abs().sum(dim=1, keepdim=True), in1_dy.abs().sum(dim=1, keepdim=True)
        abs_in2_dx, abs_in2_dy = in2_dx.abs().sum(dim=1, keepdim=True), in2_dy.abs().sum(dim=1, keepdim=True)

        weight_dx, weight_dy = torch.exp(-abs_in2_dx), torch.exp(-abs_in2_dy)

        variation = weight_dx * abs_in1_dx + weight_dy * abs_in1_dy

        if mean != False:
            return variation.mean()
        return variation.sum()