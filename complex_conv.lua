require 'nn'

local SpatialComplexConvolution, parent = torch.class('nn.SpatialComplexConvolution', 'nn.Module')

local function Re(a)
   return a:select(a:nDimension(), 1)
end
local function Im(a)
   return a:select(a:nDimension(), 2)
end

function SpatialComplexConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)
   self.convReRe = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   self.convReIm = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   self.convImRe = self.convReRe:clone('weight', 'bias', 'gradWeight', 'gradBias')
   self.convImIm = self.convReIm:clone('weight', 'bias', 'gradWeight', 'gradBias')
end

function SpatialComplexConvolution:reset(stdv)
   self.convReRe:reset(stdv)
   self.convReIm:reset(stdv)
end

function SpatialComplexConvolution:updateOutput(input)
   assert(input:size(input:nDimension()) == 2) -- must be complex
   local inputRe = Re(input)
   local inputIm = Im(input)
   local outReRe = self.convReRe:updateOutput(inputRe)
   local outReIm = self.convReIm:updateOutput(inputRe)
   local outImRe = self.convImRe:updateOutput(inputIm)
   local outImIm = self.convImIm:updateOutput(inputIm)
   self.output = self.output or torch.Tensor()
   if input:nDimension() == 4 then
      self.output:resize(outReRe:size(1), outReRe:size(2),
			 outReRe:size(3), 2)
   elseif input:nDimension() == 5 then
      self.output:resize(outReRe:size(1), outReRe:size(2),
			 outReRe:size(3), outReRe:size(4), 2)
   else
      error("SpatialComplexConvolution: input must be complex")
   end
   local outRe = Re(self.output)
   local outIm = Im(self.output)
   outRe:copy(outReRe):add(-1,outImIm)
   outIm:copy(outReIm):add(outImRe)
   return self.output
end

function SpatialComplexConvolution:updateGradInput(input, gradOutput)
   local inputRe = Re(input)
   local inputIm = Im(input)
   local gradOutputRe = Re(gradOutput)
   local gradOutputIm = Im(gradOutput)
   local giReRe =self.convReRe:updateGradInput(inputRe, gradOutputRe)
   local giImIm =self.convImIm:updateGradInput(inputIm,-gradOutputRe)
   local giReIm =self.convReIm:updateGradInput(inputRe, gradOutputIm)
   local giImRe =self.convImRe:updateGradInput(inputIm, gradOutputIm)
   self.gradInput = self.self.gradInput or torch.Tensor()
   self.gradInput:resizeAs(input)
   local giRe = Re(self.gradInput)
   local giIm = Im(self.gradInput)
   giRe:copy(giReRe):add(giReIm)
   giIm:copy(giImRe):add(giImIm)
   return self.gradInput
end

function SpatialComplexConvolution:accGradParameters(input, gradOutput, scale)
   local inputRe = Re(input)
   local inputIm = Im(input)
   local gradOutputRe = Re(gradOutput)
   local gradOutputIm = Im(gradOutput)
   self.convReRe:accGradParameters(inputRe, gradOutputRe)
   self.convImIm:accGradParameters(inputIm,-gradOutputRe)
   self.convReIm:accGradParameters(inputRe, gradOutputIm)
   self.convImRe:accGradParameters(inputIm, gradOutputIm)
end
