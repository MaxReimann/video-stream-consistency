## test onnx export and run custom layers and complete network in various configurations
## Author: Max Reimann

import os
import sys
import numpy as np
from numpy.core.fromnumeric import shape
import torch
from torch import nn
import onnxruntime as ort

import pwc_pytorch.pwcnet as pwcnetmodule
from pwc_pytorch.pwcnet import PWCNet
from pwc_pytorch.pwcnet import DummyCorrelationFunction, DummyBackwardFunction
#from pwc_pytorch.correlation import correlation 

from modelrunner import ModelRunner
import flowvis

from convert_onnx import export_to_onnx, PyTorchWrapper
from PIL import Image

from spatial_correlation_sampler import  spatial_correlation_sample


class Corr(nn.Module):
    def __init__(self, legacy, md = 4):
        self.md = md
        self.legacy_corr = legacy
        super().__init__()
    def forward(self, t1, t2):
        return DummyCorrelationFunction.apply(t1.contiguous(), t2.contiguous(), self.md, self.legacy_corr)

class Warp(nn.Module):
    def forward(self, t1, f):
        return DummyBackwardFunction.apply(t1, f)

def test_correlation_custom_layer():
    # test combination of (legacy, GPU on)
    test_comb = [ (False, True), (False, False)]
    # test_comb = [(True, True), (False, True), (False, False)]
    for legacy, use_cuda in test_comb:
        print("=========== Testing Correlation ===========")
        print("Legacy_corr: ", legacy, " use CUDA: ", use_cuda)
        print("===========================================")
        net = Corr(legacy=legacy)
        first = torch.rand((4,64, 128, 128)) * 10
        second = torch.rand((4,64, 128, 128)) * 10

        modelname = "corr_only.onnx"
        torch.onnx.export(net,
                        (first.cpu(),second.cpu()),
                        modelname,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        #do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['frame1', "frame2"],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes = {'frame1':{0:'batch',
                                    2:'width',
                                    3:'height'},
                                    'frame2':{0:'batch',
                                    2:'width',
                                    3:'height'},
                                    'output':{0:'batch',
                                    3:'width',
                                    4:'height'}})
        runner = ModelRunner(modelname, providers=['CUDAExecutionProvider' if use_cuda else "CPUExecutionProvider"])
        out_onnx = runner.run({"frame1": first.clone().numpy(), "frame2": second.clone().numpy()})
        out_onnx = runner.run({"frame1": first.clone().numpy(), "frame2": second.clone().numpy()})

        print("out_onnx shape", out_onnx[0].shape)

        if legacy:
            out = correlation.FunctionCorrelation(tenFirst=first.cuda(), tenSecond=second.cuda())
            out = out.cpu()
        else:
            if use_cuda:
                first = first.cuda()
                second = second.cuda()
            out = spatial_correlation_sample(first, second, patch_size=9).cpu()

        print("torch out shape", out.shape)
        print("diff sum: ", np.sum(out_onnx - out.numpy()))
        print("diff avg: ", np.average(out_onnx - out.numpy()))

        if use_cuda:
            print("GPU binding test")
            ortvalue1 = ort.OrtValue.ortvalue_from_numpy(first.clone().cpu().numpy(), 'cuda', 0)
            ortvalue2 = ort.OrtValue.ortvalue_from_numpy(second.clone().cpu().numpy(), 'cuda', 0)
            out_onnx = runner.run({"frame1": ortvalue1, "frame2": ortvalue2})
            out_onnx = runner.run({"frame1": ortvalue1, "frame2": ortvalue2})

            print("torch out shape", out.shape)
            print("diff sum: ", np.sum(out_onnx - out.numpy()))
            print("diff avg: ", np.average(out_onnx - out.numpy()))


def warp_alt(tensorInput, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow
    """
    x = tensorInput

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).type(x.dtype)

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    #return output

    # I found that masking is not really necessary
    # checked other implementations, arflow doen't do it as well for example
    mask = torch.ones(x.size(), device=x.device)
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # TODO this could be optimized, is a bit slow in layerwise timing
    mask[mask<0.999] = 0
    mask[mask>0] = 1

    return output*mask


def test_warp_custom_layer():
    for use_cuda in [True, False]:
        print("=========== Testing Warp ===========")
        print("use CUDA: ", use_cuda)
        print("====================================")
        net = Warp()
        first = torch.rand((2,64, 64, 32)) * 10
        flow = torch.rand((2,2, 64, 32)) #* 3

        modelname = "warp_only.onnx"
        torch.onnx.export(net,
                        (first.cpu(),flow.cpu()),
                        modelname,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        #do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['frame1', "flow"],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes = {'frame1':{0:'batch',
                                    1: "channels",
                                    2:'width',
                                    3:'height'},
                                    'flow':{0:'batch',
                                    2:'width',
                                    3:'height'},
                                    'output':{0:'batch',
                                    1:"channels",
                                    2:'width',
                                    3:'height'}})

        runner = ModelRunner(modelname, providers=['CUDAExecutionProvider' if use_cuda else "CPUExecutionProvider"])
        out_onnx = runner.run({"frame1": first.clone().numpy(), "flow": flow.clone().numpy()})
        out_onnx = runner.run({"frame1": first.clone().numpy(), "flow": flow.clone().numpy()})
        print("out shape", out_onnx[0].shape)


        out = warp_alt(first.to("cuda" if use_cuda else "cpu"), flow.to("cuda" if use_cuda else "cpu"))
        print("torch out shape", out.shape)

        print("diff sum: ", np.sum(out_onnx[0] - out.cpu().numpy()))
        print("diff avg: ", np.average(out_onnx[0] - out.cpu().numpy()))

        if use_cuda:
            print("GPU binding test")
            ortvalue1 = ort.OrtValue.ortvalue_from_numpy(first.clone().cpu().numpy(), 'cuda', 0)
            ortvalue2 = ort.OrtValue.ortvalue_from_numpy(flow.clone().cpu().numpy(), 'cuda', 0)
            out_onnx = runner.run({"frame1": ortvalue1, "flow": ortvalue2})
            out_onnx = runner.run({"frame1": ortvalue1, "flow": ortvalue2})

            print("torch out shape", out.shape)
            print("diff sum: ", np.sum(out_onnx - out.cpu().numpy()))
            print("diff avg: ", np.average(out_onnx - out.cpu().numpy()))
    

state = {
    "inputWidth": 1024,
    "inputHeight": 436,
    "count": 2,
    "batchnum": 1
}



import math
import scipy
import scipy.ndimage

def resize(array, shape):
    factor = tuple(new / old for old, new in zip(array.shape, shape))

    if type(array) == np.ndarray:
        return scipy.ndimage.zoom(array, factor)
    elif type(array) == cp.ndarray:
        return cupyx.scipy.ndimage.zoom(array, factor)

def preprocess(inputs, arguments):
    print("start preprocess")
    frame1 = inputs["frame1"]
    frame2 = inputs["frame2"]

    output_dict = {}

    for input, output_name in [(frame1, "frame1"),(frame2, "frame2")]:
        height, width, _ = input.shape 
        state["inputHeight"], state["inputWidth"] = height, width
        print("height, width", height, width)
        intPreprocessedWidth = int(math.floor(math.ceil(width / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(height / 64.0) * 64.0))


        processedInput = resize(input, (intPreprocessedHeight, intPreprocessedWidth, 4))
        # if arguments["flip"]:
        #     processedInput = np.flip(processedInput, 0)
        
        processedInput = processedInput[:,:,:3]
        processedInput = processedInput.transpose(2, 0, 1).astype(np.float32)
        # processedInput = processedInput.transpose(2, 1, 0).astype(np.float32)

        processedInput = processedInput / 255.0
        processedInput = processedInput[np.newaxis,...]

        output_dict[output_name] = processedInput
    return output_dict

def test_complete_model():

    for use_spatial_corr in [True]: # [True, False]:
        print("=========== Testing Full Model ===========")
        print("use spatial corr pkg model: ", use_spatial_corr)
        print("==========================================")

        # modelname = export_to_onnx(use_spatial_corr_pkg=use_spatial_corr)
        modelname = "pwcnet-dense.onnx"
        print("----- predicting using: ", modelname, " ---------------")

        inp1 = np.array(Image.open("first.png"))
        inp2 = np.array(Image.open("second.png"))

        preprocessed = preprocess({"frame1": inp1, "frame2": inp2}, {})

        runner = ModelRunner(modelname, providers=['CUDAExecutionProvider'])
        out_onnx = runner.run(preprocessed)

        # ================ pytorch ===========
        pwcnetmodule.FOR_EXPORT = False
        if use_spatial_corr:
            model = PWCNet.from_spec("pwcnet-my-dense-chairs-things-sintel")#.clone()
        else:
            model = PWCNet.from_spec("pwcnet-ft-sintel")#.clone()
        model.cuda()
        
        net = PyTorchWrapper(model)

        with torch.no_grad():
            first = net.frameToTensor(inp1)
            second = net.frameToTensor(inp2)
            out = net.model.forward(first, second)
            flowarr_pytorch = out.cpu().numpy()

        diff = (out_onnx[0] - flowarr_pytorch)
        print("diff sum: ", np.sum(diff))
        print("diff avg: ", np.average(diff))
        
        out_onnx = runner.run(preprocessed)
        diff = (out_onnx[0] - flowarr_pytorch)
        print("diff sum: ", np.sum(diff))
        print("diff avg: ", np.average(diff))

        print("GPU binding test")
        runner = ModelRunner(modelname, providers=['CUDAExecutionProvider'])
        ortvalue1 = ort.OrtValue.ortvalue_from_numpy(first.clone().cpu().numpy(), 'cuda', 0)
        ortvalue2 = ort.OrtValue.ortvalue_from_numpy(second.clone().cpu().numpy(), 'cuda', 0)
        out_onnx = runner.run({"frame1": ortvalue1, "frame2": ortvalue2})
        # out_onnx = runner.run({"frame1": ortvalue1, "flow": ortvalue2})

        diff = (out_onnx[0] - flowarr_pytorch)
        print("diff sum: ", np.sum(diff))
        print("diff avg: ", np.average(diff))

        out_onnx = runner.run({"frame1": ortvalue1, "frame2": ortvalue2})
        # out_onnx = runner.run({"frame1": ortvalue1, "flow": ortvalue2})

        diff = (out_onnx[0] - flowarr_pytorch)
        print("diff sum: ", np.sum(diff))
        print("diff avg: ", np.average(diff))

    
        #writes pwcnet_out_onnx.png
        #postprocess(out_onnx, {}, show=True)
def test_with_pre_processing():
    print("=========== Testing Full Model with  preprocessing===========")
    print("use spatial corr pkg model: ", True)
    print("==========================================")

    modelname = "PWCNet-light-wpreproc.onnx"
    # resize Image.open("first.png") to half size
    inp1 =  Image.open("first.png")
    inp1 = inp1.resize((inp1.width//2, inp1.height//2), Image.ANTIALIAS)
    inp1 = np.array(inp1)[np.newaxis,...]
    inp1 = np.concatenate((inp1, np.zeros_like(inp1[...,:1])), axis=3) 
    inp2 = Image.open("second.png")
    inp2 = inp2.resize((inp2.width//2, inp2.height//2), Image.ANTIALIAS)
    inp2 = np.array(inp2)[np.newaxis,...]
    inp2 = np.concatenate((inp2, np.zeros_like(inp2[...,:1])), axis=3)
    print("input shape: ", inp1.shape)

    intHeight = inp1.shape[1]
    intWidth = inp1.shape[2]


    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))



    runner = ModelRunner(modelname, providers=['CUDAExecutionProvider'])
    # out_onnx = runner.run({"frame1": inp1, "frame2": inp2,'scale_to_width': np.array(intPreprocessedWidth), 'scale_to_height':np.array(intPreprocessedHeight)})
    out_onnx = runner.run({"frame1": inp1, "frame2": inp2}) #,'scale_to_width': np.array(intPreprocessedWidth), 'scale_to_height':np.array(intPreprocessedHeight)})


    # ================ pytorch ===========
    pwcnetmodule.FOR_EXPORT = False
    model = PWCNet.from_spec("pwcnet-my-dense-chairs-things-sintel")#.clone()
    model.cuda()
    
    net = PyTorchWrapper(model)
    
    firstUnprocessed = torch.tensor(inp1)
    secondUnprocessed = torch.tensor(inp2)
    with torch.no_grad():
        first = firstUnprocessed.cuda() #net.frameToTensor(inp1)
        second = secondUnprocessed.cuda() #net.frameToTensor(inp2)
        out = net.forward(first, second)#, intPreprocessedWidth, intPreprocessedHeight)
        flowarr_pytorch = out.cpu().numpy()
        print("output pyt shape: ", flowarr_pytorch.shape)

    diff = (out_onnx[0] - flowarr_pytorch)
    # diff2 = (out_onnx[0] - flowarr_pytorch2)
    print("output shape: ", out_onnx[0].shape)
    print("diff sum: ", np.sum(diff))#, "diff2", np.sum(diff2))
    print("diff avg: ", np.average(diff))#,  "diff2", np.average(diff2))


def test_flowvis_model():
    for use_cuda in [False, True]:#, False]:
        print("=========== Testing FLowvis model ===========")
        print("use CUDA: ", use_cuda)
        print("====================================")
        
        pwcnetmodule.FOR_EXPORT = False
        model = PWCNet.from_spec("pwcnet-my-dense-chairs-things-sintel")#.clone()
        model.cuda()
        
        net = PyTorchWrapper(model)
        inp1 = np.array(Image.open("first.png"))
        inp2 = np.array(Image.open("second.png"))

        firstUnprocessed = torch.tensor(inp1).unsqueeze(0).cuda()
        secondUnprocessed = torch.tensor(inp2).unsqueeze(0).cuda()
        with torch.no_grad():
            # first = net.frameToTensor(inp1)
            # second = net.frameToTensor(inp2)
            # out = net.model.forward(first, second)
            out = net.forward(firstUnprocessed, secondUnprocessed)#, intPreprocessedWidth, intPreprocessedHeight)
            # flowarr_pytorch = out.cpu().numpy()
        
        flow_np = out.cpu().numpy()
        print("flow_np_shape", flow_np.shape)
        modelname = "flowvisonly.onnx"
        runner = ModelRunner(modelname, providers=['CUDAExecutionProvider' if use_cuda else "CPUExecutionProvider"],register_custom_ops=False)
        flowvis_out_onnx = runner.run({"flow": flow_np})
        print("out shape", flowvis_out_onnx[0].shape)
        flowimg = Image.fromarray(flowvis_out_onnx[0])
        flowimg.save("out_onnx.png")
        # print("output pyt shape: ", flowarr_pytorch.shape)
        flowvis_np = flowvis.flow_to_color(flow_np[0, :, :, :2])
        flowvis_out_onnx = torch.floor(255 * flowvis_out_onnx[0]).to(dtype=torch.uint8) 

        diff = (flowvis_out_onnx - flowvis_np)
        # diff2 = (out_onnx[0] - flowarr_pytorch2)
        print("output shape: ", flowvis_out_onnx[0].shape)
        print("diff sum: ", np.sum(diff))#, "diff2", np.sum(diff2))
        print("diff avg: ", np.average(diff))#,  "diff2", np.average(diff2))


def preprocess_batched(frame1, frame2):
    output_dict = {}
    for input, output_name in [(frame1, "frame1"),(frame2, "frame2")]:
        batchnum, height, width, _ = input.shape 
        state["batchnum"], state["inputHeight"], state["inputWidth"] = batchnum, height, width
        intPreprocessedWidth = int(math.floor(math.ceil(width / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(height / 64.0) * 64.0))

        rgbinput = input[:,:,:,:3]
        with torch.no_grad():
            processedInput = torch.from_numpy(rgbinput.transpose(0,3,1,2)).float().cuda()
            processedInput = torch.nn.functional.interpolate(processedInput,size=(intPreprocessedHeight, intPreprocessedWidth), mode="bilinear")
        processedInput = processedInput / 255.0
        output_dict[output_name] = np.ascontiguousarray(processedInput.cpu().numpy())
    return output_dict


def test_batched():
    print("=========== Testing Batched Full Model ===========")
    print("use spatial corr pkg model: ", True)
    print("==========================================")

    modelname = "pwcnet-dense.onnx"
    inp1 = np.repeat(np.array(Image.open("first.png"))[np.newaxis,...], 4, axis=0)
    inp2 = np.repeat(np.array(Image.open("second.png"))[np.newaxis,...], 4, axis=0)
    print("input shape: ", inp1.shape)

    preprocessed = preprocess_batched(inp1,inp2)

    runner = ModelRunner(modelname, providers=['CUDAExecutionProvider'])
    out_onnx = runner.run(preprocessed)

    # ================ pytorch ===========
    pwcnetmodule.FOR_EXPORT = False
    model = PWCNet.from_spec("pwcnet-my-dense-chairs-things-sintel")#.clone()
    model.cuda()
    
    net = PyTorchWrapper(model)

    with torch.no_grad():
        first = torch.from_numpy(preprocessed["frame1"]).cuda() #net.frameToTensor(inp1)
        second = torch.from_numpy(preprocessed["frame2"]).cuda() #net.frameToTensor(inp2)
        out = net.model.forward(first, second)
        flowarr_pytorch = out.cpu().numpy()
    
    diff = (out_onnx[0] - flowarr_pytorch)
    print("output shape: ", out_onnx[0].shape)
    print("diff sum: ", np.sum(diff))
    print("diff avg: ", np.average(diff))

if __name__ == "__main__":
    # test_warp_custom_layer()
    # test_correlation_custom_layer()
    # test_complete_model()
    # test_with_pre_processing()
    test_flowvis_model()
    # test_batched()s