# stnbdhw

This is a warping module (BDHW layout) needed for optical flow warping.

``` lua
require 'stn'

nn.BilinearSamplerBDHW()
-- takes a table {inputImages, grids} as inputs
-- outputs the interpolated images according to the grids
-- inputImages is a batch of samples in BDHW layout
-- grids is a batch of grids (relative coordinates)
-- output is also BDHW
```

