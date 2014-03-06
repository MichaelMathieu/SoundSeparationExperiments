require 'torch'
require 'audio'

--sound = audio.samplevoice()
--sound = audio.load("obama.mp3")

function soundStatistics(t)
   return {min = t:min(),max = t:max(),
	   mean = t:mean(), std = t:std()}
end

function normalizeSound(t)
   local mean = t:mean()
   local std = t:std()
   return (t - mean)/std
end

function addNoise(sound, sigma)
   local noise = torch.randn(sound.size):mul(sigma)
   noise:add(sound)
   return noise
end

