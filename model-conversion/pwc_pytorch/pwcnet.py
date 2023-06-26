## Original PWCNet for optical flow using Pyramid, Warping, and Cost Volume, Sun et al., CVPR 2018
## Code based on remimplementation by sniklaus https://github.com/sniklaus/pytorch-pwc
## Light-PWC architecture for optimization on mobile by Moritz Hilscher, 2021
## Adapted for ONNX export by Max Reimann
## Licensed under GPL v2


try:
    from . import model_spec
except ImportError:
    import model_spec

import torch
import torch.autograd
import torch.nn.functional as F

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

def import_legacy_correlation():
    # a bit whacky... just don't cause errors on the training server
    # might be removed at some point anyways
    try:
        try:
            from correlation import correlation # the custom cost volume layer
        except:
            from .correlation import correlation
    except:
        sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
    return correlation

from spatial_correlation_sampler import SpatialCorrelationSampler

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

Backward_tensorGrid = {}
Backward_tensorPartial = {}

def BackwardOrig(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
    # end

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones([ tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])
    # end

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    tensorInput = torch.cat([ tensorInput, Backward_tensorPartial[str(tensorFlow.size())] ], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    tensorMask = tensorOutput[:, -1:, :, :]
    tensorMask[tensorMask > 0.999] = 1.0
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask
# end


def BackwardAlt(tensorInput, tensorFlow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    x = tensorInput
    flo = tensorFlow

    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).type(x.dtype)

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    #return output

    # I found that masking is not really necessary
    # checked other implementations, arflow doen't do it as well for example
    mask = torch.ones(x.size(), device=x.device)
    mask = torch.autograd.Variable(mask)
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)

    # TODO this could be optimized, is a bit slow in layerwise timing
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask

from torch.onnx import symbolic_helper
class DummyBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensorInput, tensorFlow):
        return tensorInput

    @staticmethod
    @symbolic_helper.parse_args("v", "v")
    def symbolic(g, input1, input2):
        return g.op("custom::Warp", input1, input2).setType(input1.type())

# from torch.onnx.symbolic_helper import parse_args
class DummyCorrelationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor1, tensor2, md, legacy):
        b, c, h, w = tensor1.size()
        if legacy:
            return torch.zeros((b, (2*md+1)**2, h, w))
        else:
            return torch.zeros((b, (2*md+1), (2*md+1), h, w)).to(tensor1.device)

    # @parse_args('v', 'v', 'i', 'i')
    @staticmethod
    @symbolic_helper.parse_args("v", "v", "i", "i")
    def symbolic(g, input1, input2, md, legacy):
        # print all methods of input1
        # print(dir(input1))
        # print(input1.type().symbolic_sizes())
        output_shape = [input1.type().symbolic_sizes()[0], (2*md+1), (2*md+1), input1.type().symbolic_sizes()[2], input1.type().symbolic_sizes()[3]]
        return g.op("custom::Correlation", input1, 
                    input2, max_displacement_i=md, 
                    legacy_i=1 if legacy else 0).setType(
            input1.type().
            with_dtype(torch.float32).
            with_sizes(output_shape)
            )

FOR_EXPORT = False

def Backward(tensorInput, tensorFlow):
    if FOR_EXPORT:
        return DummyBackwardFunction.apply(tensorInput, tensorFlow)
    else:
        return BackwardAlt(tensorInput, tensorFlow)

def dw_separated_conv(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bn=False):
    conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation, bias=False)
    conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

    seq = []
    seq.append(conv1)
    if bn:
        seq.append(torch.nn.BatchNorm2d(in_channels))
    seq.append(conv2)
    if bn:
        seq.append(torch.nn.BatchNorm2d(out_channels))

    return torch.nn.Sequential(*seq)

##########################################################

# number of output features per pyramid level, may be decreased with a factor
BASE_FEATURE_CHANNEL_COUNTS = [16, 32, 64, 96, 128, 196]
def get_feature_channel_counts(channel_alpha):
    return [ int(c * channel_alpha) for c in BASE_FEATURE_CHANNEL_COUNTS ]

class Extractor(torch.nn.Module):

    def __init__(self, channel_alpha=1.0):
        super(Extractor, self).__init__()

        c = get_feature_channel_counts(channel_alpha)
        #ci = [12, 23, 45, 68, 90, 196]

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=c[0], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[0], out_channels=c[0], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[0], out_channels=c[0], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[0], out_channels=c[1], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[1], out_channels=c[2], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[2], out_channels=c[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[2], out_channels=c[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[2], out_channels=c[3], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[3], out_channels=c[3], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[3], out_channels=c[3], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[3], out_channels=c[4], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[4], out_channels=c[4], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[4], out_channels=c[4], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[4], out_channels=c[5], kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[5], out_channels=c[5], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=c[5], out_channels=c[5], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    # end

    def forward(self, tensorInput):
        #print("Extractor input:", tensorInput.size())
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
    # end
# end

class ExtractorSeparated(torch.nn.Module):
    def __init__(self):
        super(ExtractorSeparated, self).__init__()

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            dw_separated_conv(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    # end

    def forward(self, tensorInput):
        #print("Extractor input:", tensorInput.size())
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
    # end
# end



class BaseDecoder(torch.nn.Module):
    def __init__(self, use_legacy_correlation, corr_md=4):
        super(BaseDecoder, self).__init__()

        self.use_legacy_correlation = use_legacy_correlation
        self.corr_md = corr_md
        self.correlation = SpatialCorrelationSampler(kernel_size=1, patch_size=self.corr_md*2+1, stride=1, padding=0, dilation_patch=1)

    def corr(self, t1, t2):
        # the correlation function that came with this implementation somehow doesn't train so well (e.g. needs more epochs),
        # so use spatial-correlation-sampler for own trained variants instead
        if FOR_EXPORT:
            c = t1.size(1)
            corr = DummyCorrelationFunction.apply(t1, t2, self.corr_md, self.use_legacy_correlation)
            if not self.use_legacy_correlation:
                # corr *= c
                return  torch.reshape(corr, (t1.size(0), (self.corr_md*2+1) ** 2, t1.size(2), t1.size(3)))
            return corr
        elif self.use_legacy_correlation:
            correlation = import_legacy_correlation()
            out = correlation.FunctionCorrelation(tenFirst=t1, tenSecond=t2)
            return out
        else:
            c = t1.size(1)
            out = self.correlation(t1, t2).type(t1.dtype)
            b, ph, pw, h, w = out.size()
            return torch.reshape(out, (b, ph*pw, h, w)) #/ c
 
class Decoder(BaseDecoder):
    def __init__(self, intLevel, use_legacy_correlation, channel_alpha=1.0, kd_features="", corr_md=4, use_regularization=False, use_batch_norm=False):
        super(Decoder, self).__init__(use_legacy_correlation, corr_md=corr_md)

        assert not use_batch_norm, "No batch norm implemented in dense decoder"
        assert channel_alpha == 1.0, "No channel alpha support for dense decoder"

        self.level = intLevel

        intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

        if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

        if use_regularization:
            self.regularization = Regularization(intLevel)
        else:
            self.regularization = None
    # end

    def forward(self, tensorFirst, tensorSecond, objectPrevious, imageFirst, imageSecond):
        tensorFlow = None
        tensorFeat = None

        #print("Decoder %d: %d input features" % (self.level, tensorFirst.size(1)))

        if objectPrevious is None:
            #print("first tensorFirst:", tensorFirst.size())
            tensorFlow = None
            tensorFeat = None
            tensorVolume = torch.nn.functional.leaky_relu(input=self.corr(tensorFirst, tensorSecond), negative_slope=0.1, inplace=False)

            tensorFeat = torch.cat([ tensorVolume ], 1)

        elif objectPrevious is not None:
            previousFlow = objectPrevious["tensorFlow"]
            previousFeat = objectPrevious["tensorFeat"]
            tensorFlow = self.moduleUpflow(previousFlow)
            tensorFeat = self.moduleUpfeat(previousFeat)


            backwardTensorSecond = Backward(tensorSecond, tensorFlow * self.dblBackward)#.contiguous()
            tensorVolume = torch.nn.functional.leaky_relu(input=self.corr(tensorFirst, backwardTensorSecond), negative_slope=0.1, inplace=False)
            #print("Corr out: %s" % str(list(tensorVolume.size())))


            tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow, tensorFeat ], 1)

        # end

        #print("  for moduleOne: %d input features" % tensorFeat.size(1))
        tensorFeat = torch.cat([ self.moduleOne(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleTwo(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleThr(tensorFeat), tensorFeat ], 1)
        tensorFeat = torch.cat([ self.moduleFou(tensorFeat), tensorFeat ], 1)
        tensorIntermediate = self.moduleFiv(tensorFeat)
        tensorFeat = torch.cat([ torch.nn.functional.leaky_relu(tensorIntermediate, negative_slope=0.1), tensorFeat ], 1)

        tensorFlow = self.moduleSix(tensorFeat)
        if self.regularization is not None:
            tensorFlow = self.regularization(imageFirst, imageSecond, tensorFirst, tensorSecond, tensorFlow)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat,
            "tensorIntermediate" : tensorIntermediate,
        }
    # end
# end

class DecoderLight(BaseDecoder):
    def __init__(self, intLevel, use_legacy_correlation, channel_alpha=1.0, corr_md=4, separated=False, use_regularization=False, use_batch_norm=False, kd_features="decend"):
        super(DecoderLight, self).__init__(use_legacy_correlation, corr_md=corr_md)

        print("Decoder %d, alpha=%.2f, corr_md=%d, separated=%d" % (intLevel, channel_alpha, corr_md, separated))
        self.intLevel = intLevel

        self.use_batch_norm = use_batch_norm

        corr_chans = (2*self.corr_md+1)**2
        chans = [None, None] + [ f+corr_chans+2 for f in get_feature_channel_counts(channel_alpha)[1:] ] + [None]
        intPrevious = chans[intLevel + 1]
        intCurrent = chans[intLevel + 0]

        if intLevel < 6: self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        #if intLevel < 6: self.moduleUpfeat = torch.nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.dblBackward = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

        alpha = channel_alpha
        #if intLevel in (3, 2):
        #    alpha = 0.5

        self.lighter = False
        #if intLevel in (3, 2):
        #    self.lighter = True

        if separated:
            self.moduleOne = torch.nn.Sequential(
                dw_separated_conv(intCurrent, int(128 * alpha), kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleTwo = torch.nn.Sequential(
                dw_separated_conv(int(128 * alpha), int(128 * alpha), kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleThr = torch.nn.Sequential(
                dw_separated_conv(int((128 + 128) * alpha), int(96 * alpha), kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleFou = torch.nn.Sequential(
                dw_separated_conv(int((128 + 96) * alpha), int(64 * alpha), kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleFiv = torch.nn.Sequential(
                dw_separated_conv(int((96 + 64) * alpha), 32, kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
                # remove leaky relu to get intermediate tensor for kd
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            # keep last convolution non-separated helps quality! (according to MobileNet, and own flow results)
            self.moduleSix = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32 + int(64 * alpha), out_channels=2, kernel_size=3, stride=1, padding=1)
                #dw_separated_conv(32 + 64, 2, kernel_size=3, stride=1, padding=1, bn=use_batch_norm),
            )
        else:

            self.moduleOne = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intCurrent, out_channels=int(128 * alpha), kernel_size=3, stride=1, padding=1),
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            if not self.lighter:
                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=int(128 * alpha), out_channels=int(128 * alpha), kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            thrInChans = 128 + 128
            if self.lighter:
                thrInChans = 128
            self.moduleThr = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int(thrInChans * alpha), out_channels=int(96 * alpha), kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleFou = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int((128 + 96) * alpha), out_channels=int(64 * alpha), kernel_size=3, stride=1, padding=1),
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleFiv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=int((96 + 64) * alpha), out_channels=32, kernel_size=3, stride=1, padding=1),
                # remove leaky relu to get intermediate tensor for kd
                #torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

            self.moduleSix = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32 + int(64 * alpha), out_channels=2, kernel_size=3, stride=1, padding=1)
            )

        self.kd_features = kd_features
        if alpha < 1.0:
            self.moduleStudentTransform = torch.nn.Conv2d(in_channels=32 + int(64 * alpha), out_channels=32 + 64, kernel_size=1)
        else:
            self.moduleStudentTransform = None

    # end

    def forward(self, tensorFirst, tensorSecond, objectPrevious, imageFirst, imageSecond):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = torch.zeros(tensorFirst.size(0), 2, tensorFirst.size(2), tensorFirst.size(3), dtype=tensorFirst.dtype, device=tensorFirst.device).type(tensorFirst.dtype)
            tensorFeat = None

            tensorVolume = torch.nn.functional.leaky_relu(input=self.corr(tensorFirst, tensorSecond), negative_slope=0.1, inplace=False)
            #tensorFeat = torch.cat([ tensorVolume ], 1)
            tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow ], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
            #tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

            backwardTensorSecond = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)#.contiguous()
            tensorVolume = torch.nn.functional.leaky_relu(input=self.corr(tensorFirst, backwardTensorSecond), negative_slope=0.1, inplace=False)
            tensorFeat = torch.cat([ tensorVolume, tensorFirst, tensorFlow ], 1)

        # end

        leaky_relu = lambda x: F.leaky_relu(input=x, negative_slope=0.1, inplace=False)
        if not self.lighter:
            x1_linear = self.moduleOne(tensorFeat)
            x1 = leaky_relu(x1_linear)
            x2 = self.moduleTwo(x1)
            x3 = self.moduleThr(torch.cat([x1, x2], dim=1))
            x4_linear = self.moduleFou(torch.cat([x2, x3], dim=1))
            x4 = leaky_relu(x4_linear)
            x5_linear = self.moduleFiv(torch.cat([x3, x4], dim=1))
            x5 = leaky_relu(x5_linear)

            tensorFeat = x5
            tensorFlow = self.moduleSix(torch.cat([x4, x5], dim=1))

            features = [x4_linear, x5_linear]
            if "decbeginend" in self.kd_features:
                features = [x1_linear, x4_linear, x5_linear]
            tensorIntermediate = torch.cat(features, dim=1)
            if self.moduleStudentTransform is not None:
                tensorIntermediate = self.moduleStudentTransform(tensorIntermediate)
        else:
            # variant with one of the heavy 128 channel out conv less - 18% performance improvement on ipad
            x1 = self.moduleOne(tensorFeat)
            x2 = self.moduleThr(x1)
            x3_linear = self.moduleFou(torch.cat([x1, x2], dim=1))
            x3 = leaky_relu(x3_linear)
            x4_linear = self.moduleFiv(torch.cat([x2, x3], dim=1))
            x4 = leaky_relu(x4_linear)

            tensorFeat = x4
            tensorFlow = self.moduleSix(torch.cat([x3, x4], dim=1))

            tensorIntermediate = torch.cat([x3_linear, x4_linear], dim=1)
            if self.moduleStudentTransform is not None:
                tensorIntermediate = self.moduleStudentTransform(tensorIntermediate)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat,
            "tensorIntermediate" : tensorIntermediate,
        }
    # end
# end

class Refiner(torch.nn.Module):
    def __init__(self, in_channels, channel_alpha=1.0):
        super(Refiner, self).__init__()

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=int(128 * channel_alpha), kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(128 * channel_alpha), out_channels=int(128 * channel_alpha), kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(128 * channel_alpha), out_channels=int(128 * channel_alpha), kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(128 * channel_alpha), out_channels=int(96 * channel_alpha), kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(96 * channel_alpha), out_channels=int(64 * channel_alpha), kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(64 * channel_alpha), out_channels=int(32 * channel_alpha), kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=int(32 * channel_alpha), out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    # end

    def forward(self, tensorInput):
        # for compatibility reasons, split sequential module here
        # to get kd features and refinement output separately
        modules = list(self.moduleMain)
        moduleFeatures = torch.nn.Sequential(*modules[:-2])
        moduleRefinement = torch.nn.Sequential(*modules[-2:])

        features = moduleFeatures(tensorInput)
        refinement = moduleRefinement(features)
        return features, refinement
    # end
# end

class RefinerSeparated(torch.nn.Module):
    def __init__(self, in_channels, channel_alpha=1.0):
        super(RefinerSeparated, self).__init__()

        assert channel_alpha == 1.0

        print("RefinerSeparated")

        in_channels_padded = int(math.ceil(in_channels / 4) * 4)
        self.pad_channels = in_channels_padded - in_channels
        assert in_channels_padded % 4 == 0
        self.moduleMain = torch.nn.Sequential(
            self.dw_separated_conv(in_channels=in_channels_padded, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            self.dw_separated_conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            self.dw_separated_conv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            self.dw_separated_conv(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            self.dw_separated_conv(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            self.dw_separated_conv(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            # keep last convolution non-separated helps quality! (according to MobileNet, and own flow results)
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def dw_separated_conv(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation, bias=False),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
            #torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, tensorInput):
        #return self.moduleMain(tensorInput)

        B, _, H, W = tensorInput.size()
        if self.pad_channels != 0:
            padding = torch.zeros(B, self.pad_channels, H, W, device=tensorInput.device)
            tensorInput = torch.cat((tensorInput, padding), 1)
        # unfortunately not implemented in coreml, have to go for the one with fixed frame size
        # (but not so bad as models run only with fixed size at the moment anyways)
        # padding order here: h, w, c, b
        #padded = torch.nn.functional.pad(tensorInput, (0, 0, 0, 0, 0, padding_chans, 0, 0), "constant", 0.0)
        return None, self.moduleMain(tensorInput)

class Regularization(torch.nn.Module):
    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

        if intLevel >= 5:
            self.moduleFeat = torch.nn.Sequential()

        elif intLevel < 5:
            self.moduleFeat = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        # end
        
        # [ 0, 0, 131, 131, 131, 131, 195 ][
        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 199 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel >= 5:
            self.moduleDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
            )

        elif intLevel < 5:
            self.moduleDist = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 3, 2, 2, 1, 1 ][intLevel], 0)),
                torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][intLevel]))
            )

        # end

        self.moduleScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
    # eny

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
        # mean causes error because copy_ not foundi
        # additionall im2col not found, it's probably unfold
        tensorFirst = torch.nn.functional.interpolate(input=tensorFirst, size=(tensorFeaturesFirst.size(2), tensorFeaturesFirst.size(3)), mode='bilinear', align_corners=False)
        tensorFirst[:, 0, :, :] -= 0.41
        tensorFirst[:, 1, :, :] -= 0.43
        tensorFirst[:, 2, :, :] -= 0.45
        tensorSecond = torch.nn.functional.interpolate(input=tensorSecond, size=(tensorFeaturesFirst.size(2), tensorFeaturesFirst.size(3)), mode='bilinear', align_corners=False)
        tensorSecond[:, 0, :, :] -= 0.41
        tensorSecond[:, 1, :, :] -= 0.43
        tensorSecond[:, 2, :, :] -= 0.45
        tensorDifference = (tensorFirst - Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(1, True).sqrt().detach()

        tensorDist = self.moduleDist(self.moduleMain(torch.cat([ tensorDifference, tensorFlow - tensorFlow.view(tensorFlow.size(0), 2, -1).mean(2, True).view(tensorFlow.size(0), 2, 1, 1), self.moduleFeat(tensorFeaturesFirst) ], 1)))
        tensorDist = tensorDist.pow(2.0).neg()
        tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

        tensorDivisor = tensorDist.sum(1, True).reciprocal()

        tensorScaleX = self.moduleScaleX(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor
        tensorScaleY = self.moduleScaleY(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor

        return torch.cat([ tensorScaleX, tensorScaleY ], 1)
    # end
# end

class PWCNet(torch.nn.Module):

    @classmethod
    def from_spec(cls, model_id, **extra_kwargs):
        path = model_spec.get_path(model_id)
        kwargs = model_spec.get_args(model_id)
        print(path)
        kwargs.update(extra_kwargs)

        if "custom_pth" in kwargs:
            model = torch.load(kwargs["custom_pth"])
            print("Loading custom pth", kwargs["custom_pth"])
            if path is not None:
                checkpoint = torch.load(path)
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            # set custom attributes according to extra_kwargs now...
            model.full_output = kwargs.get("full_output", False)
            return model

        return cls(path, **kwargs)

    def __init__(self,
            # general options
            path,
            legacy=True,
            load_strict=True,
            full_output=False,
            channel_alpha=1.0, # for thinning network
            # extractor related
            no_extractor=False,
            extractor_separated=False,
            # decoder related
            separated_decoder=False,
            separated_refinement=False,
            decoder_light=False,
            decoder_count=5,
            decoder_corr_md=4,
            decoder_bn=False,
            # refinement related
            use_refinement=True,
            # kd training stuff
            kd_features="decend",
            # misc experimental stuff
            use_regularization=False,
            halfres=False):
        super(PWCNet, self).__init__()

        self.no_extractor = no_extractor
        self.use_refinement = use_refinement
        self.separated_refinement = separated_refinement
        self.decoder_light = decoder_light
        self.decoder_count = decoder_count
        self.decoder_bn = decoder_bn
        self.halfres = halfres
        self.legacy = legacy

        self.kd_features = kd_features
        self.full_output = full_output

        extractor_cls = ExtractorSeparated if extractor_separated else Extractor
        extractor_kwargs = {"channel_alpha" : channel_alpha}
        self.moduleExtractor = extractor_cls(**extractor_kwargs)

        decoder_cls = DecoderLight if decoder_light else Decoder
        decoder_kwargs = {
            "channel_alpha" : channel_alpha,
            "use_legacy_correlation" : legacy,
            "corr_md" : decoder_corr_md,
            "use_regularization" : use_regularization,
            "use_batch_norm" : decoder_bn,
            "kd_features" : kd_features
        }

        if separated_decoder is True:
            decoder_kwargs["separated"] = True
        self.moduleSix = decoder_cls(6, **decoder_kwargs)
        self.moduleFiv = decoder_cls(5, **decoder_kwargs)
        self.moduleFou = decoder_cls(4, **decoder_kwargs)
        if separated_decoder == "partially":
            decoder_kwargs["separated"] = True
        self.moduleThr = decoder_cls(3, **decoder_kwargs)
        self.moduleTwo = decoder_cls(2, **decoder_kwargs)

        refiner_cls = RefinerSeparated if self.separated_refinement else Refiner
        refiner_kwargs = {
            "channel_alpha" : channel_alpha
        }
        refiner_channels = 81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32
        if decoder_light:
            refiner_channels = 32 + 2
        self.moduleRefiner = refiner_cls(refiner_channels, **refiner_kwargs)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity="leaky_relu", a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

        if path is not None:
            checkpoint = torch.load(path)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # variants with channel alpha don't know student transform... somehow causes error
            if channel_alpha < 1.0:
                load_strict = False
            self.load_state_dict(state_dict, strict=load_strict)

        # to get counting parameters right, and to keep loading models with excess parameters saved working
        if self.decoder_count < 5:
            del self.moduleTwo
        if not self.use_refinement:
            del self.moduleRefiner
    # end

    def forward(self, tensorFirst, tensorSecond):
        if self.halfres:
            tensorFirst = torch.nn.functional.interpolate(tensorFirst, scale_factor=0.5)
            tensorSecond = torch.nn.functional.interpolate(tensorSecond, scale_factor=0.5)
        #print(tensorFirst.size())
        imageFirst = tensorFirst.detach().clone()
        imageSecond = tensorSecond.detach().clone()

        if not self.no_extractor:
            tensorFirst = self.moduleExtractor(tensorFirst)
            tensorSecond = self.moduleExtractor(tensorSecond)
        else:
            tensorFirst = []
            tensorFirst.append(torch.zeros(1, 16, 224, 512))
            tensorFirst.append(torch.zeros(1, 32, 112, 256))
            tensorFirst.append(torch.zeros(1, 64, 56, 128))
            tensorFirst.append(torch.zeros(1, 96, 28, 64))
            tensorFirst.append(torch.zeros(1, 128, 14, 32))
            tensorFirst.append(torch.zeros(1, 196, 7, 16))
            # if I generate the second tensors like this... don't get the loadConstantND ops on coreml generated...
            tensorSecond = []
            tensorSecond.append(torch.ones(1, 16, 224, 512))
            tensorSecond.append(torch.ones(1, 32, 112, 256))
            tensorSecond.append(torch.ones(1, 64, 56, 128))
            tensorSecond.append(torch.ones(1, 96, 28, 64))
            tensorSecond.append(torch.ones(1, 128, 14, 32))
            tensorSecond.append(torch.ones(1, 196, 7, 16))
            tensorSecond = tensorFirst
            #tensorSecond = [ tensor.clone().detach() for tensor in tensorFirst ]

        #print("Extractor shapes:")
        #for i, tensor in enumerate(tensorFirst):
        #    print(i, tensor.size())

        extra = {"imageFirst" : imageFirst, "imageSecond" : imageSecond}
        estimates = []
        hiddenFeatures = []
        if "features" in self.kd_features:
            for i in range(1, 5+1):
                hiddenFeatures.append(tensorFirst[-i])
                hiddenFeatures.append(tensorSecond[-i])

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None, **extra)
        if self.decoder_count >= 2:
            estimates.append(objectEstimate["tensorFlow"])
            hiddenFeatures.append(objectEstimate["tensorIntermediate"])
            objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate, **extra)
            # debug.append(objectEstimate["tensorFeat"])
            if self.decoder_count >= 3:
                estimates.append(objectEstimate["tensorFlow"])
                hiddenFeatures.append(objectEstimate["tensorIntermediate"])
                objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate, **extra)
                if self.decoder_count >= 4:
                    estimates.append(objectEstimate["tensorFlow"])
                    hiddenFeatures.append(objectEstimate["tensorIntermediate"])
                    objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate, **extra)
                    if self.decoder_count == 5:
                        estimates.append(objectEstimate["tensorFlow"])
                        hiddenFeatures.append(objectEstimate["tensorIntermediate"])
                        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate, **extra)


        ## apparently that factor makes it impossible to train with pytorch,
        ## but was necessary for weights converted from caffe
        #factor = 5 if self.legacy else 1
        # --> issue solved: multiplying by 20 is okay when gt data is downscaled by 20

        flow = objectEstimate['tensorFlow']
        if self.decoder_light:
            # for light decoder
            refine_in = torch.cat([objectEstimate["tensorFeat"], objectEstimate["tensorFlow"]], dim=1)
        else:
            # for normal decoder
            refine_in = objectEstimate["tensorFeat"]

        if self.use_refinement:
            refineFeatures, refine = self.moduleRefiner(refine_in)
            flow = flow + refine
            hiddenFeatures.append(refineFeatures)
        estimates.append(flow)

        # number of decoders -> required flow upscale factor
        scale_factors = {1 : 16*2*2, 3 : 16, 4 : 8, 5 : 4}
        scale_factor = scale_factors[self.decoder_count]
        flow_factor = 20
        if self.halfres:
            scale_factor *= 2
            flow_factor *= 2

        # when exporting to coreml, add upsampling manually
        # if FOR_EXPORT:
        #     # divide by scale_factor because, without scaling now, it is added later
        #     if self.use_refinement:
        #         f = 20 / scale_factor
        #         return flow * f
        #     else:
        #         return flow, refine_in

        final = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)(flow * flow_factor)
        # THIS for exporting flow in consistency evaluation
        #final = flow * flow_factor
        if self.full_output:
            return final, estimates, hiddenFeatures
        else:
            return final

    # end
# end