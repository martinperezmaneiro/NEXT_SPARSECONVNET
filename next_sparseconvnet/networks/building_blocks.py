import torch
import sparseconvnet as scn
from torch.autograd import Function

# Gradient Reversal Layer definition
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=0.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

class ResidualBlock_downsample(torch.nn.Module):
    def __init__(self, inplanes, kernel, stride, bias = False, dim = 3, momentum = 0.99):
        torch.nn.Module.__init__(self)

        outplanes = 2 * inplanes

        #f1
        self.bnr1    = scn.BatchNormReLU(inplanes, momentum = momentum)
        self.conv1   = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr2    = scn.BatchNormReLU(outplanes, momentum = momentum)
        self.subconv = scn.SubmanifoldConvolution(dim, outplanes, outplanes, kernel, bias)

        #f2
        self.conv2   = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)

        self.add     = scn.AddTable()

    def forward(self, x):
        x = self.bnr1(x)

        #f1
        y1 = self.conv1(x)
        y1 = self.bnr2(y1)
        y1 = self.subconv(y1)

        #f2
        y2 = self.conv2(x)

        #sum
        out = self.add([y1, y2])

        return out

class ResidualBlock_basic(torch.nn.Module):
    def __init__(self, inplanes,  kernel, dim=3, momentum = 0.99):
        torch.nn.Module.__init__(self)
        self.bnr1 = scn.BatchNormReLU(inplanes, momentum = momentum)
        self.subconv1 = scn.SubmanifoldConvolution(dim, inplanes, inplanes, kernel, 0)
        self.bnr2 = scn.BatchNormReLU(inplanes, momentum = momentum)
        self.subconv2 = scn.SubmanifoldConvolution(dim, inplanes, inplanes, kernel, 0)
        self.add = scn.AddTable()

    def forward(self, x):
        y = self.bnr1(x)
        y = self.subconv1(y)
        y = self.bnr2(y)
        y = self.subconv2(y)
        x = self.add([x,y])

        return x

class ResidualBlock_upsample(torch.nn.Module):
    def __init__(self, inplanes, kernel, stride, bias = False, dim = 3, momentum = 0.99):
        torch.nn.Module.__init__(self)

        outplanes = int(inplanes / 2)

        #f1
        self.bnr1      = scn.BatchNormReLU(inplanes, momentum = momentum)
        self.deconv1   = scn.Deconvolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr2      = scn.BatchNormReLU(outplanes, momentum = momentum)
        self.subconv   = scn.SubmanifoldConvolution(dim, outplanes, outplanes, kernel, bias)

        #f2
        self.deconv2   = scn.Deconvolution(dim, inplanes, outplanes, kernel, stride, bias)

        self.add       = scn.AddTable()

    def forward(self, x):
        x = self.bnr1(x)

        #f1
        y1 = self.deconv1(x)
        y1 = self.bnr2(y1)
        y1 = self.subconv(y1)

        #f2
        y2 = self.deconv2(x)

        #sum
        out = self.add([y1, y2])

        return out


class ConvBNBlock(torch.nn.Module):
    def __init__(self, inplanes, outplanes, kernel, stride = 1, bias = False, dim = 3, momentum = 0.99):
        torch.nn.Module.__init__(self)
        if stride==1:
            self.conv = scn.SubmanifoldConvolution(dim, inplanes, outplanes, kernel, bias)
        else:
            self.conv = scn.Convolution(dim, inplanes, outplanes, kernel, stride, bias)
        self.bnr = scn.BatchNormReLU(outplanes, momentum = momentum)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnr(x)
        return x


def calculate_output_dimension(spatial_size, kernel_sizes, stride_sizes):
    """Assures that kernel and stride sizes are suitable for our input size and returns the output size for the bottom layer after downsampling.
       Note that kernel_sizes' last element does not correspond to the kernel of a downsample, so it does not count for the calculation
    """
    out_dim = []
    for o in spatial_size:
        for i, k in enumerate(kernel_sizes[:-1]):
            o = (o - k)/stride_sizes[i] + 1
            assert o == int(o), 'Shape mismatch: kernel size {} in level {} does not return a suitable size for the output'.format(k, i)
        out_dim.append(int(o))
    return out_dim
