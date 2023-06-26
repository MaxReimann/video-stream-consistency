# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

# Slightly modified by Moritz Hilscher 
# Modified by Max Reimann for pytorch/onnx-based flow visualization

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import torch


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    RY, YG, GC, CB, BM, MR = [42]*6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

colorwheel = make_colorwheel()
colorwheel_torch = torch.from_numpy(colorwheel).float()

def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:.transpose(2,1,0)

    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    #colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


@torch.jit.script
def make_flow_image(u):
    return torch.zeros((u.shape[0], u.shape[1], 3), dtype=torch.uint8).to(u.device)



# onnx does not support torch.atan2
def atan2_alternative(A, B):
    pi = torch.Tensor([np.pi]).float().to(A.device)
    pi_half = pi / 2

    # Calculate atan(A, B)
    atan_A_B = torch.atan(A / B)
    # B > 0
    result = torch.where(B > 0, atan_A_B, torch.zeros_like(A))
    # B < 0, A >= 0
    result = torch.where((B < 0) & (A >= 0), atan_A_B + pi, result)
    # B < 0, A < 0
    result = torch.where((B < 0) & (A < 0), atan_A_B - pi, result)
    # B = 0, A > 0
    result = torch.where((B == 0) & (A > 0), pi_half, result)
    # B = 0, A < 0
    result = torch.where((B == 0) & (A < 0), -pi_half, result)
    # B = 0, A = 0, NaN values
    zero = torch.tensor(0).float().to(A.device)
    result = torch.where((B == 0) & (A == 0), zero, result)

    return result


# @torch.jit.script
def torch_flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param u: torch.Tensor, input horizontal flow
    :param v: torch.Tensor, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return: torch.Tensor, flow color image of shape [H,W,3]
    '''

    # flow_image = make_flow_image(u)

    # colorwheel = make_colorwheel()  
    ncols = 252#colorwheel_torch.shape[0]
    # print(ncols, "ncols")

    rad = torch.sqrt(torch.square(u) + torch.square(v))
    # atan2 = lambda A,B: torch.atan(-v / (-u+1e-6))
    # a = torch.atan(-u / (-v+1e-6)) / np.pi #torch.tensor([np.pi], dtype=torch.float32).to(u.device)
    # a = torch.atan2(-v, -u) / np.pi #torch.tensor([np.pi], dtype=torch.float32).to(u.device)
    a = atan2_alternative(-v,-u) / np.pi
    # print("diff ", torch.abs(a - torch.atan2(-v, -u) / np.pi).sum())
    # a=a.contiguous()

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    # for i in range(3):
    tmp = torch.Tensor(colorwheel_torch).to(u.device) #[:, i]
    col0 = tmp[k0] / 255.0
    col1 = tmp[k1] / 255.0
    f = f.unsqueeze(2).repeat(1, 1, 3)
    col = (1 - f) * col0 + f * col1

    # idx = idx.unsqueeze(2).repeat(1, 1, 3)
    rad = rad.unsqueeze(2).repeat(1, 1, 3)
    idx = (rad <= 1)
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    col[~idx] = col[~idx]* 0.75   # out of range?

    return  col #torch.floor(255 * col)#.to(dtype=torch.uint8) # flow_image


# @torch.jit.script
def torch_flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,3]

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    :param flow_uv: torch.Tensor of shape [H,W,3]
    :param clip_flow: float, maximum clipping value for flow
    :return: torch.Tensor, flow color image of shape [H,W,3]
    '''

    # assert flow_uv.ndim == 4, 'input flow must have three dimensions'
    # assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = torch.clamp(flow_uv, 0, clip_flow)

    u = flow_uv[..., 0].contiguous() #.double()
    v = flow_uv[..., 1].contiguous()#.double()

    rad = torch.sqrt(torch.square(u) + torch.square(v))
    rad_max = torch.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return torch_flow_compute_color(u, v, convert_to_bgr)

if __name__ == "__main__":
    img = Image.new("RGB", (colorwheel.shape[0], 1))
    pixels = img.load()
    for i in range(img.size[0]):
        pixels[i, 0] = tuple(list(colorwheel[i].astype(np.uint8)))
    img.save("colorwheel.png")