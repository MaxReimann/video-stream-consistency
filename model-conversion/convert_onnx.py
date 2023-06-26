## convert pytorch pwcnet to onnx
## Author: Max Reimann

import numpy as np
import math
import torch
from PIL import Image
import flowvis

@torch.jit.script
def get_size(x: torch.IntTensor, y: torch.IntTensor):# -> torch.Size:
    item = torch.Size(x.item(), y.item())
    return item

@torch.jit.script
def interpolate(X: torch.Tensor):# -> torch.Size:
    intPreprocessedHeight = int(math.floor(math.ceil(X.size(2) / 64.0) * 64.0))
    intPreprocessedWidth = int(math.floor(math.ceil(X.size(3) / 64.0) * 64.0))

    return torch.nn.functional.interpolate(input=X, size=(intPreprocessedHeight,intPreprocessedWidth), mode='bilinear', align_corners=False)

@torch.jit.script
def rescaleFlow(X: torch.Tensor, Y: torch.Tensor):# -> torch.Size:
    return torch.nn.functional.interpolate(input=X, size=(Y.size(2), Y.size(3)), mode='bilinear', align_corners=False)

@torch.jit.script
def interpolateFlowToSize(X: torch.Tensor, Y: torch.Tensor):# -> torch.Size:
    u = Y.size(2) / X.size(2)
    v = Y.size(3) / X.size(3)
    flowresized = torch.nn.functional.interpolate(input=X, size=(Y.size(2), Y.size(3)), mode='bilinear', align_corners=False)
    # flowresized[:, 0] *= u 
    # flowresized[:, 1] *= v 
    # instead of above 2 lines we create a tensor with correct dimensions.. otherwise scatter_nd nodes are created in onnx which are somehow executed on cpu
    flowresized *= torch.tensor([u,v]).to(X.device).unsqueeze(0).repeat(X.size(0),1).unsqueeze(2).unsqueeze(3)
    return flowresized



class PyTorchWrapper(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super(PyTorchWrapper, self).__init__()
        self.model = model
        self.model.eval()
        self.model.to("cuda")
        self.normalize = normalize

    def frameToTensor(self, frame):
        data = frame.transpose(2, 0, 1).astype(np.float32)
        data = np.ascontiguousarray(data[np.newaxis,:,:,:])

        if self.normalize:
            data = data / 255.0

        tensor = torch.from_numpy(data)
        tensor = tensor.cuda()

        intWidth = tensor.size(3)
        intHeight = tensor.size(2)
        tensorPreprocessed = tensor.view(1, 3, intHeight, intWidth)
        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
        tensorPreprocessed = torch.nn.functional.interpolate(input=tensorPreprocessed, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        return tensorPreprocessed
    
    # traces resizing to factor of 32 and othe preprocessing steps as well for onnx
    def forward(self, frame1, frame2):
        print('shape frame1', frame1.shape)
        frame1 = frame1.permute(0, 3, 1, 2).float()
        frame2 = frame2.permute(0, 3, 1, 2).float()
        frame1 = frame1[:,:3,...] / 255.0
        frame2 = frame2[:,:3,...] / 255.0
        tensor1Preprocessed = interpolate(frame1)
        tensor2Preprocessed = interpolate(frame2)
        tensor1Preprocessed = tensor1Preprocessed.contiguous()
        tensor2Preprocessed = tensor2Preprocessed.contiguous()
        print('new shape frame 1', tensor1Preprocessed.shape)
        pred = self.model(tensor1Preprocessed, tensor2Preprocessed)
        # u = frame1.size(2) / pred.size(2)
        # v = frame1.size(3) / pred.size(3)
        # tensorOutput[:, 0, :, :] *= frame1.size(2) / pred.size(2)
        # tensorOutput[:, 1, :, :] *= frame1.size(3) / pred.size(3) #/ float(intHeight)
        tensorOutput = interpolateFlowToSize(pred, frame1)

        tensorOutput = tensorOutput.permute(0, 2, 3, 1)
        # add channel to create RGB32 form
        tensorOutput = torch.cat((tensorOutput, torch.zeros_like(tensorOutput[...,:1]).float()), dim=3) 

        return tensorOutput
    
class FlowVisWrapper(PyTorchWrapper):
    def __init__(self, model, normalize=True):
        super(FlowVisWrapper, self).__init__(model, normalize)
    
    def forward(self, frame1, frame2):
        tensorOutput = super(FlowVisWrapper, self).forward(frame1, frame2)
        tensorOutput = flowvis.torch_flow_to_color(tensorOutput[0])
        return tensorOutput
    
class TestFlowVisOnly(torch.nn.Module):
    def __init__(self):
        super(TestFlowVisOnly, self).__init__()

    def forward(self, flow):
        tensorOutput = flowvis.torch_flow_to_color(flow[0])
        return tensorOutput

def export_to_onnx(use_spatial_corr_pkg, custom_model=None,custom_export_name=None):
    import pwc_pytorch.pwcnet as pwcnetmodule
    from pwc_pytorch.pwcnet import PWCNet
    pwcnetmodule.FOR_EXPORT = False

    if custom_model:
        model = custom_model
    else:
        if use_spatial_corr_pkg:
            #model = PWCNet.from_spec("pwcnet-my-light4dec")
            #model = PWCNet.from_spec("pwcnet-my-dense-chairs-things-sintel") ## works
            model = PWCNet.from_spec("pwcnet-my-light4dec-sepref-chairs-things-sintel")
        else:
            model = PWCNet.from_spec("pwcnet-ft-sintel")
    model.eval()
    netflowvis = FlowVisWrapper(model)
    flowvisonly = TestFlowVisOnly()
    net = PyTorchWrapper(model)
    # for whole model
    h, w = 448, 1024 # mpi sintel resolution
    # SINTEL RESOLUTION IS ACTUALLY 1024x436 (1024x448 with padding for pwcnet) 
    #inputs = (torch.zeros(1, 3, h, w), torch.zeros(1, 3, h, w))

    with torch.no_grad():
        inp1 = np.array(Image.open("first.png"))#[:, :, ::-1]
        first = net.frameToTensor(inp1)
        inp2 = np.array(Image.open("second.png"))#[:, :, ::-1]
        second = net.frameToTensor(inp2)
        # print(first.shape)

        img = Image.open("first.png")
        intWidth = img.width
        intHeight = img.height
        firstUnprocessed = torch.tensor(inp1).unsqueeze(0)
        secondUnprocessed = torch.tensor(inp2).unsqueeze(0)
        print("input shape", firstUnprocessed.shape)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        firstUnprocessed = torch.cat((firstUnprocessed, torch.zeros_like(firstUnprocessed[...,:1])), dim=3) 
        secondUnprocessed = torch.cat((secondUnprocessed, torch.zeros_like(secondUnprocessed[...,:1])), dim=3)

        outflow = net.forward(firstUnprocessed.clone().cuda(), secondUnprocessed.clone().cuda())#,torch.IntTensor([intPreprocessedWidth]),torch.IntTensor([intPreprocessedHeight]))
        print(outflow.shape)
        
        flowarr =  outflow[0, :, :, :2].cpu().numpy()#.transpose(1,2,0)# net.tensorToFlow(out, (t1.shape[3], t1.shape[2]))
        flows = flowvis.flow_to_color(flowarr)
        flowimg = Image.fromarray(flows)
        flowimg.save("out_pytorch.png")

        flowvisout = (flowvisonly.forward(outflow) * 255).cpu().numpy().astype(np.uint8)
        flowimg = Image.fromarray(flowvisout)#.astype(np.uint8))
        flowimg.save("out_model_flowvisonly.png")

        out = netflowvis.forward(firstUnprocessed.cuda(), secondUnprocessed.cuda())#,torch.IntTensor([intPreprocessedWidth]),torch.IntTensor([intPreprocessedHeight]))
        print(out.shape)
        flowimg = Image.fromarray(out.cpu().numpy().astype(np.uint8))
        flowimg.save("out_model_flowvis.png")

        if custom_export_name:
            out_name = custom_export_name 
        else:
            out_name = "pwcnet-dense.onnx" if use_spatial_corr_pkg else "pwcnet-orig.onnx"

        pwcnetmodule.FOR_EXPORT = True
        # net.model = net.model.cpu()
        net = net.cuda()
        torch.onnx.export(net,
                        (firstUnprocessed.cuda(),secondUnprocessed.cuda()),#torch.IntTensor([intPreprocessedWidth]),torch.IntTensor([intPreprocessedHeight])),
                        out_name,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=17,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        verbose=False,
                        input_names = ['frame1', 'frame2'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes = {'frame1':{0:'batch',
                                    1:'height',
                                    2:'width'},
                                    'frame2':{0:'batch',
                                    1:'height',
                                    2:'width'},
                                   'output':{0:'batch',
                                    1:'height',
                                    2:'width'}
                                    }
        )
        print('Successfully exported PWCNet')

    return out_name

if __name__ == "__main__":
    export_to_onnx(use_spatial_corr_pkg=True,custom_export_name="PWCNet-light-wpreproc.onnx")
