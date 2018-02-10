require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstn'
if withCuda then
   require 'libcustn'
end

require('stn.BilinearSamplerBDHW')

require('stn.test')

return nn
