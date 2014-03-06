require 'complex_conv'

local SpatialCircularConvolution, parent = torch.class('nn.SpatialCircularConvolution', 'nn.Module')

function SpatialCircularConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   dH = dH or 1
   dW = dW or 1
   parent.__init(self)
   self.conv = nn.SpatialComplexConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   assert(dW == 1) -- for now
   self.k = kW-1
   self.input = torch.Tensor()
end

function SpatialCircularConvolution:reset(stdv)
   self.conv:reset(stdv)
end

function SpatialCircularConvolution:updateOutput(input)
   assert(input:nDimension() == 4)
   self.input:resize(input:size(1), input:size(2),
		     input:size(3)+self.k, 2)
   self.input[{{},{},{1,input:size(3)}}]:copy(input)
   self.input[{{},{},{input:size(3)+1, input:size(3)+self.k}}]:copy(input[{{},{},{1, self.k}}])
   self.output = self.conv:updateOutput(self.input)
   return self.output
end

function SpatialCircularConvolution:updateGradInput(input, gradOutput)
   self.gradInput = self.conv:updateGradInput(self.input, gradOutput)
   return self.gradInput
end

function SpatialCircularConvolution:accGradParameters(input, gradOutput, scale)
   self.conv:accGradParameters(self.input, gradOutput, scale)
end
