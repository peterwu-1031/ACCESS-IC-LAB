import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########
##  PACT
##########


class PactClip(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, upper_bound):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

            upper_bound   if input > upper_bound
        y = input         if 0 <= input <= upper_bound
            0             if input < 0
        """
        ctx.save_for_backward(input, upper_bound)
        return torch.clamp(input, 0, upper_bound.data)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, upper_bound, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_upper_bound = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>upper_bound] = 0
        grad_upper_bound[input<=upper_bound] = 0
        return grad_input, torch.sum(grad_upper_bound)

class PactReLU(nn.Module):
    def __init__(self, upper_bound=6.0):
        super(PactReLU, self).__init__()
        self.upper_bound = nn.Parameter(torch.tensor(upper_bound))

    def forward(self, input):
        return PactClip.apply(input, self.upper_bound)


##########
##  Mask
##########


class SparseGreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<threshold] = 0
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None


##########
##  Quant
##########


class Floor(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the round function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Clamp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the clamp function is defined as the identity function.
        """
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None

class TorchBinarize(nn.Module):
    """ Binarizes a value in the range [-1,+1] to {-1,+1} """
    def __init__(self):
        super(TorchBinarize, self).__init__()

    def forward(self, input):
        """  clip to [-1,1] """
        input = Clamp.apply(input, -1.0, 1.0)
        """ rescale to [0,1] """
        input = (input+1.0) / 2.0
        """ round to {0,1} """
        input = Round.apply(input)
        """ rescale back to {-1,1} """
        input = input*2.0 - 1.0
        return input

class TorchRoundToBits(nn.Module):
    """ Quantize a tensor to a bitwidth larger than 1 """
    def __init__(self, bits=2):
        super(TorchRoundToBits, self).__init__()
        assert bits > 1, "RoundToBits is only used with bitwidth larger than 1."
        self.bits = bits
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Round.apply(input * (2.0**self.bits-1.0)) / (2.0**self.bits-1.0)
        return input * sign, scaling

class TorchTruncate(nn.Module):
    """ 
    Quantize an input tensor to a b-bit fixed-point representation, and
    remain the bh most-significant bits.
        Args:
        input: Input tensor
        b:  Number of bits in the fixed-point
        bh: Number of most-significant bits remained
    """
    def __init__(self, b=8, bh=4):
        super(TorchTruncate, self).__init__()
        assert b > 0, "Cannot truncate floating-point numbers (b=0)."
        assert bh > 0, "Cannot output floating-point numbers (bh=0)."
        assert b > bh, "The number of MSBs are larger than the total bitwidth."
        self.b = b
        self.bh = bh
        self.epsilon = 1e-7

    def forward(self, input):
        """ extract the sign of each element """
        sign = torch.sign(input).detach()
        """ get the mantessa bits """
        input = torch.abs(input)
        scaling = torch.max(input).detach() + self.epsilon
        input = Clamp.apply( input/scaling ,0.0, 1.0 )
        """ round the mantessa bits to the required precision """
        input = Round.apply( input * (2.0**self.b-1.0) )
        """ truncate the mantessa bits """
        input = Floor.apply( input / (2**(self.b-self.bh) * 1.0) )
        """ rescale """
        input *= (2**(self.b-self.bh) * 1.0)
        input /= (2.0**self.b-1.0)
        return input * scaling * sign

class TorchQuantize(nn.Module):
    """ 
    Quantize an input tensor to the fixed-point representation. 
        Args:
        input: Input tensor
        bits:  Number of bits in the fixed-point
    """
    def __init__(self, bits=0):
        super(TorchQuantize, self).__init__()
        if bits == 0:
            self.quantize = nn.Identity()
        elif bits == 1:
            self.quantize = TorchBinarize()
        else:
            self.quantize = TorchRoundToBits(bits)

    def forward(self, input):
        return self.quantize(input)


##########
##  Layer
##########


class QuantizedConv2d(nn.Conv2d):
    """ 
    A convolutional layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', wbits=8, abits=3,
                 ADCprecision=10, vari=0):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, 
                                              kernel_size, stride, 
                                              padding, dilation, groups, 
                                              bias, padding_mode)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.weight_rescale = \
            np.sqrt(1.0/(3**2 * in_channels)) if (wbits == 1) else 1.0
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.subArray=128
        self.bitWeight=8
        self.cellBit=4
        self.t=0
        self.v=0
        self.detect=0
        self.target=0
        self.onoffratio=1000
        self.bitActivation=8

    #這兩個function是handle device variation的，但目前還沒成功執行
    def Retention(self, x, t, v, detect, target):
        lower = torch.min(x).item()
        upper = torch.max(x).item()
        target = (torch.max(x).item() - torch.min(x).item())*target
        if detect == 1: # need to define the sign of v 
            sign = torch.zeros_like(x)
            truncateX = (x+1)/2
            truncateTarget = (target+1)/2
            sign = torch.sign(torch.add(torch.zeros_like(x),truncateTarget)-truncateX)
            ratio = t**(v*sign)
        else :  # random generate target for each cell
            sign = torch.randint_like(x, -1, 2)
            truncateX = (x+1)/2
            ratio = t**(v*sign)

        return torch.clamp((2*truncateX*ratio-1), lower, upper)

    #這個是處理ADC quantized effect
    def LinearQuantizeOut(self, x, bit):
        minQ = torch.min(x)
        delta = torch.max(x) - torch.min(x)
        y = x.clone()

        stepSizeRatio = 2.**(-bit)
        stepSize = stepSizeRatio*delta.item()
        index = torch.clamp(torch.floor((x-minQ.item())/stepSize), 0, (2.**(bit)-1))
        y = index*stepSize + minQ.item()

        return y
    def DeviceVariation(self, input):
        input, input_scaling = self.quantize_a(input)
        weight, weight_scaling = self.quantize_w(self.weight)
        outputOriginal = F.conv2d(input * input_scaling,
                        weight * weight_scaling * self.weight_rescale,
                        self.bias, self.stride, self.padding, 
                        self.dilation, self.groups)
        output=torch.zeros_like(outputOriginal)
        del outputOriginal

        upper=1
        lower=1/self.onoffratio
        cellRange=2**self.cellBit
        
        #weight=self.quantize_w(self.weight) * self.weight_rescale
        dummyP = torch.zeros_like(weight)
        dummyP[:, :, :, :] = (cellRange-1)*(upper+lower)/2
        #print(weight)
        #self.weight.shape[1]=16;self.weight.shape[2]=3;self.weight.shape[3]=3
        for i in range(self.weight.shape[2]):
            #print(i)
            for j in range(self.weight.shape[3]):
                numSubArray = int(weight.shape[1]/self.subArray)
                if numSubArray == 0:
                    #print("A")
                    mask = torch.zeros_like(weight)
                    mask[:, :, i, j] = 1
                    if weight.shape[1] == 3:
                        X_decimal = torch.round((2**self.bitWeight - 1)/2 * (weight+1) + 0)*mask
                        outputP = torch.zeros_like(output)
                        outputD = torch.zeros_like(output)    
                        for k in range(int(self.bitWeight/self.cellBit)):
                            remainder = torch.fmod(X_decimal, cellRange)*mask
                            remainder = self.Retention(remainder, self.t, self.v, self.detect, self.target)
                            X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                            remainderQ = (upper-lower) * (remainder-0)+(cellRange-1)*lower
                            remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari, device='cuda', dtype=torch.float))
                            outputPartial = F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            outputDummyPartial = F.conv2d(input, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            scaler = cellRange**k
                            outputP = outputP + outputPartial * scaler*2/(1-1/self.onoffratio)
                            outputD = outputD + outputDummyPartial * scaler*2/(1-1/self.onoffratio)
                        outputP = outputP - outputD
                        output = output + outputP
                    else:
                        #print("B")
                        #print("AA")
                        inputQ = torch.round((2**self.bitActivation - 1)/1 * (input-0) + 0)
                        outputIN = torch.zeros_like(output)
                        for z in range(self.bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)
                            outputP = torch.zeros_like(output)
                            X_decimal = torch.round((2**self.bitWeight - 1)/2 * (weight+1) + 0)*mask
                            outputD = torch.zeros_like(output)
                            #print(int(self.bitWeight/self.cellBit))
                            for k in range(int(self.bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask
                                remainder = self.Retention(remainder, self.t, self.v, self.detect, self.target)
                                X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                remainderQ = (upper-lower) * (remainder-0)+(cellRange-1)*lower
                                remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari, device='cuda', dtype=torch.float))
                                outputPartial = F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial = F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputPartialQ = self.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                outputDummyPartialQ = self.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                scaler = cellRange**k
                                outputP = outputP + outputPartialQ * scaler*2/(1-1/self.onoffratio)
                                outputD = outputD + outputDummyPartialQ * scaler*2/(1-1/self.onoffratio)
                            #print("tTTTTT")
                            scalerIN = 2**z
                            outputIN = outputIN + (outputP - outputD)*scalerIN
                        #print("LLLL")
                        output = output + outputIN/(2**self.bitActivation)
                else:
                    #print("C")
                    inputQ = torch.round((2**self.bitActivation - 1)/1 * (input-0) + 0)
                    outputIN = torch.zeros_like(output)
                    for z in range(self.bitActivation):
                        inputB = torch.fmod(inputQ, 2)
                        inputQ = torch.round((inputQ-inputB)/2)
                        outputP = torch.zeros_like(output)
                        for s in range(self.numSubArray):
                            mask = torch.zeros_like(weight)
                            mask[:, (s*self.subArray):(s+1) * self.subArray, i, j] = 1
                            X_decimal = torch.round((2**self.bitWeight - 1)/2 * (weight+1) + 0)*mask
                            outputSP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range(int(self.bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask
                                remainder=self.Retention(remainder,self.t,self.v,self.detect,self.target)
                                X_decimal=torch.round((X_decimal-remainder)/cellRange)*mask
                                remainderQ = (upper-lower) * (remainder-0)+(cellRange-1)*lower
                                remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari, device='cuda', dtype=torch.float))
                                outputPartial = F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial = F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputPartialQ = self.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                outputDummyPartialQ = self.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                scaler =cellRange**k
                                outputSP = outputSP + outputPartialQ * scaler*2/(1-1/self.onoffratio)
                                outputD = outputD + outputDummyPartialQ * scaler*2/(1-1/self.onoffratio)
                            outputSP-=outputD
                            outputP+=outputSP
                        scalerIN =2**z
                        outputIN+=outputP*scalerIN
                    output+=outputIN/(2**self.bitActivation)
        #print("===============================================")
        output/=2**self.bitWeight
        output *= input_scaling
        output *= weight_scaling
        return output

    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform convolution
        """
        #out = F.conv2d(self.quantize_a(input)[0] * self.quantize_a(input)[1], self.quantize_w(self.weight)[0] * self.quantize_w(self.weight)[1] * self.weight_rescale, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #在跑inference之前要先檢查這邊
        #print(self.ADCprecision)
        #print("PPPPPPP")
        out = self.DeviceVariation(input)
        #out=LinearQuantizeOut(out,self.ADCprecision)
        return out

class QuantizedLinear(nn.Linear):
    """ 
    A fully connected layer with its weight tensor and input tensor quantized. 
    """
    def __init__(self, in_features, out_features, bias=True, wbits=0, abits=0):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.weight_rescale = np.sqrt(1.0/in_features) if (wbits == 1) else 1.0
        
    def forward(self, input):
        """ 
        1. Quantize the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform matrix multiplication 
        """
        return F.linear(self.quantize_a(input), 
                        self.quantize_w(self.weight) * self.weight_rescale, 
                        self.bias)
        
class PGConv2d(nn.Conv2d):
    """ 
    A convolutional layer computed as out = out_msb + mask . out_lsb
        - out_msb = I_msb * W
        - mask = (I_msb * W)  > Delta
        - out_lsb = I_lsb * W
    out_msb calculates the prediction results.
    out_lsb is only calculated where a prediction result exceeds the threshold.

    **Note**: 
        1. PG predicts with <activations>.
        2. bias must set to be False!
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', wbits=8, abits=8, pred_bits=4, 
                 sparse_bp=False, alpha=5):
        super(PGConv2d, self).__init__(in_channels, out_channels, 
                                       kernel_size, stride, 
                                       padding, dilation, groups, 
                                       bias, padding_mode)
        self.quantize_w = TorchQuantize(wbits)
        self.quantize_a = TorchQuantize(abits)
        self.trunc_a = TorchTruncate(b=abits, bh=pred_bits)
        self.gt = SparseGreaterThan.apply if sparse_bp else GreaterThan.apply
        self.weight_rescale = \
            np.sqrt(1.0/(kernel_size**2 * in_channels)) if (wbits == 1) else 1.0
        self.alpha = alpha

        """ 
        zero initialization
        nan loss while using torch.Tensor to initialize the thresholds 
        """
        self.threshold = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        """ number of output features """
        self.num_out = 0
        """ number of output features computed at high precision """
        self.num_high = 0

    def forward(self, input):
        """ 
        1. Truncate the input tensor
        2. Quantize the weight tensor
        3. Rescale via McDonnell 2018 (https://arxiv.org/abs/1802.08530)
        4. perform MSB convolution
        """
        out_msb = F.conv2d(self.trunc_a(input),
                           self.quantize_w(self.weight) * self.weight_rescale,
                           self.bias, self.stride, self.padding, 
                           self.dilation, self.groups)
        """ Calculate the mask """
        mask = self.gt(torch.sigmoid(self.alpha*(out_msb-self.threshold)), 0.5)
        """ update report """
        self.num_out = mask.cpu().numel()
        self.num_high = mask[mask>0].cpu().numel()
        """ perform LSB convolution """
        out_lsb = F.conv2d(self.quantize_a(input)-self.trunc_a(input),
                           self.quantize_w(self.weight) * self.weight_rescale, 
                           self.bias, self.stride, self.padding, 
                           self.dilation, self.groups)
        """ combine outputs """
        return out_msb + mask * out_lsb
