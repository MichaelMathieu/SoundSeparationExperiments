require 'torch'
require 'audio'
require 'nn'
require 'circular_conv'
require 'complex_conv'

local net = nn.Sequential()
net:add(nn.SpatialCircularConvolution(1, 8, 9, 9))
net:add(nn.Threshold())
net:add(nn.SpatialCircularConvolution(8, 16, 9, 9))
net:add(nn.Threshold())
net:add(nn.SpatialCircularConvolution(16, 1, 9, 9))


voice = audio.samplevoice()
input = audio.stft(voice, 128, 'hann', 1)
print{input}
input = input[{{1,2000}}]
input = input:reshape(1, input:size(1), input:size(2), 2)
output = net:forward(input)
print{output}
