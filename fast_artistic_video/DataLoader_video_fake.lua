require 'torch'
require 'hdf5'
require 'stn'

local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'

local DataLoader_video_fake = torch.class('DataLoader_video_fake')

local vr = require 'fast_artistic_video.vr_helper'

local function addBatchDim(input, num)
  return input:view(1, input:size(1), input:size(2), input:size(3)):expand(num, input:size(1), input:size(2), input:size(3))
end

function DataLoader_video_fake:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.batch_size, 'Must provide batch size')
  self.preprocess_fn = preprocess[opt.preprocessing].preprocess

  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  
  -- For vr, make sure to compute this only once
  self.map_first_left, self.map_first_right, self.map_first_top, self.map_first_bottom = nil, nil, nil, nil
  self.map_second_left, self.map_second_right, self.map_second_top, self.map_second_bottom = nil, nil, nil, nil
  self.warpNet = nil
  
  self.idx = 1
  
  self.image_paths = {
    train = '/train2014/images',
    val = '/val2014/images',
  }
  
  self.img_size_h = opt.train_img_size:split(':')[1]
  self.img_size_w = opt.train_img_size:split(':')[2]
  
  local train_size = self.h5_file:read(self.image_paths.train):dataspaceSize()
  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val):dataspaceSize()[1],
  }
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]

  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end
  
  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end

  self.rgb_to_gray = torch.FloatTensor{0.2989, 0.5870, 0.1140}
end


function DataLoader_video_fake:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader_video_fake:setSplitIdxFromIter(split, it)
  self.split_idxs[split] = ((it - 1) * self.batch_size + 1) % (self.split_sizes[split] - (self.split_sizes[split] % self.batch_size) + self.batch_size)
  print('Resuming from mini-batch index ' .. self.split_idxs[split])
end

function DataLoader_video_fake:setSplitIdx(split, idx)
  self.split_idxs[split] = idx
  print('Resuming from mini-batch index ' .. self.split_idxs[split])
end

function DataLoader_video_fake:getSplitIdx(split)
  return self.split_idxs[split]
end

function DataLoader_video_fake:getBatch(split, mode, num, dtype)
  local path = self.image_paths[split]

  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  
  -- Load images out of the HDF5 file
  local images = self.h5_file:read(path):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)

  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end

  -- Preprocess images
  local imgs_pre = self.preprocess_fn(images):type(dtype)
  
  local b, w, h = images:size(1), tonumber(self.img_size_w), tonumber(self.img_size_h)
  
  local imgsList = {}
  local flowList = {}
  local certList = {}

  if mode == 'shift' then

    local displ_x, displ_y = math.floor(math.random() * 32) - 16, math.floor(math.random() * 32) - 16
    local offs_x, offs_y = 16, 16
    
    local scale_net = nn.SpatialUpSamplingBilinear({oheight=h + offs_y * num, owidth=w + offs_x * num}):type(dtype)
    imgs_pre = scale_net:forward(imgs_pre)
    
    for i=0,num do
      local crop_net = nn.Sequential()
        :add(nn.Narrow(4, math.max(-displ_x * (num-i), 0) + math.max(displ_x * i, 0) + 1, w))
        :add(nn.Narrow(3, math.max(-displ_y * (num-i), 0) + math.max(displ_y * i, 0) + 1, h)):type(dtype)
      local imgs_i_pre = crop_net:forward(imgs_pre)
      table.insert(imgsList, imgs_i_pre)
    end

    local flow = torch.Tensor(1,2,1,1)
    flow[1][1][1][1], flow[1][2][1][1] = displ_y, displ_x
    flow = torch.repeatTensor(flow, b, 1, w, h):type(dtype):contiguous()
    
    local cert = torch.Tensor(b, 1, w, h):fill(1):type(dtype)
    
    if displ_x > 0 then cert[{{},{},{},{math.min(w-displ_x,w),w}}]:zero() end
    if displ_x < 0 then cert[{{},{},{},{1,math.max(-displ_x,1)}}]:zero() end
    if displ_y > 0 then cert[{{},{},{math.min(h-displ_y,h),h},{}}]:zero() end
    if displ_y < 0 then cert[{{},{},{1,math.max(-displ_y,1)},{}}]:zero() end
        
    for i=1,num do
      table.insert(flowList, flow)
      table.insert(certList, cert)
    end

  elseif mode == 'zoom_out' then

    local displ_x, displ_y = math.floor(math.random() * 32) - 16, math.floor(math.random() * 32) - 16

    for i=0,num do
      local crop_net = nn.Sequential()
        :add(nn.Narrow(4, math.max(-displ_x * (num-i), 0) + 1, w - math.abs(displ_x * (num-i))))
        :add(nn.Narrow(3, math.max(-displ_y * (num-i), 0) + 1, h - math.abs(displ_y * (num-i)))):type(dtype)
      local scale_net = nn.SpatialUpSamplingBilinear({oheight=h, owidth=w}):type(dtype)
      local imgs_i_pre = scale_net:forward(crop_net:forward(imgs_pre))
      table.insert(imgsList, imgs_i_pre)
    end

    local cert = torch.Tensor(b, 1, w, h):fill(1):type(dtype)

    if displ_x > 0 then cert[{{},{},{},{math.min(w-displ_x,w),w}}]:zero() end
    if displ_x < 0 then cert[{{},{},{},{1,math.max(-displ_x,1)}}]:zero() end
    if displ_y > 0 then cert[{{},{},{math.min(h-displ_y,h),h},{}}]:zero() end
    if displ_y < 0 then cert[{{},{},{1,math.max(-displ_y,1)},{}}]:zero() end

    -- x/y grids
    local grid_x = torch.ger( torch.linspace(-math.max(-displ_y,0),math.max(displ_y,0),w), torch.ones(h) )
    local grid_y = torch.ger( torch.ones(w), torch.linspace(-math.max(-displ_x,0),math.max(displ_x,0),h) )

    -- Apply scale
    local flow_scale = torch.FloatTensor()
    flow_scale:resize(2,w,h)
    flow_scale[1] = grid_x
    flow_scale[2] = grid_y

    local flow = flow_scale:view(1, 2, w, h):expand(b, 2, w, h):type(dtype):contiguous()
    for i=1,num do
      table.insert(flowList, flow)
      table.insert(certList, cert)
    end

  elseif mode == 'single_image' then

    local scale_net = nn.SpatialUpSamplingBilinear({oheight=w, owidth=h}):type(dtype)
    imgs_pre = scale_net:forward(imgs_pre)
  
    table.insert(imgsList, torch.zeros(b, 3, w, h):type(dtype))
    table.insert(imgsList, imgs_pre)
    table.insert(flowList, torch.zeros(b, 2, w, h):type(dtype):contiguous())
    table.insert(certList, torch.zeros(b, 1, w, h):type(dtype))
   
  elseif mode == 'vr' then
  
    if self.warpNet == nil then self.warpNet = nn.BilinearSamplerBDHW():type(dtype) end

    local rnd = math.random()

    -- We need to create warp maps that take a normal image and create something like a perspective transformed
    -- image border you would get if you transform the borders of cube map projections from one side to the other.
    -- We do this for all four sides of an image.
    local map_first, map_second = nil, nil
    local crop_net = nil
    if rnd < 0.25 then
      if self.map_first_left == nil or self.map_first_left:size(1) ~= b then
        self.map_first_left = addBatchDim(vr.make_perspective_warp_map_left(images:size(3), 70, images:size(4), 0):type(dtype), b):contiguous()
      end
      if self.map_second_left == nil or self.map_second_left:size(1) ~= b  then
        self.map_second_left = vr.make_perspective_warp_map_right(h, 64, w, 0, 0):type(dtype)
        -- Morror the projection to be on the correct side
        self.map_second_left[2]:add(-w+64)
        self.map_second_left = addBatchDim(self.map_second_left, b):contiguous()
      end
      map_first = self.map_first_left
      map_second = self.map_second_left
      crop_net = nn.Sequential():add(nn.Narrow(4, images:size(4)-64, 64)):add(nn.Narrow(3, 65, -65)):type(dtype)
    elseif rnd < 0.5 then
      if self.map_first_right == nil or self.map_first_right:size(1) ~= b then
        self.map_first_right = addBatchDim(vr.make_perspective_warp_map_right(images:size(3), 70, images:size(4), 0):type(dtype), b):contiguous()
      end
      if self.map_second_right == nil or self.map_second_right:size(1) ~= b then
        self.map_second_right = addBatchDim(vr.make_perspective_warp_map_left(h, 64, w, 0, 0):type(dtype), b):contiguous()
      end
      map_first = self.map_first_right
      map_second = self.map_second_right
      crop_net = nn.Sequential():add(nn.Narrow(4, 1, 64)):add(nn.Narrow(3, 65, -65)):type(dtype)
    elseif rnd < 0.75 then
      if self.map_first_top == nil or self.map_first_top:size(1) ~= b then
        self.map_first_top = addBatchDim(vr.make_perspective_warp_map_top(images:size(4), 70, images:size(3), 0):type(dtype), b):contiguous()
      end
      if self.map_second_top == nil or self.map_second_top:size(1) ~= b then
        self.map_second_top = vr.make_perspective_warp_map_bottom(w, 64, h, 0, 0):type(dtype)
        -- Morror the projection to be on the correct side
        self.map_second_top[1]:add(-h+64)
        self.map_second_top = addBatchDim(self.map_second_top, b):contiguous()
      end
      map_first = self.map_first_top
      map_second = self.map_second_top
      crop_net = nn.Sequential():add(nn.Narrow(3, images:size(3)-64, 64)):add(nn.Narrow(4, 65, -65)):type(dtype)
    else
      if self.map_first_bottom == nil or self.map_first_bottom:size(1) ~= b then
        self.map_first_bottom = addBatchDim(vr.make_perspective_warp_map_bottom(images:size(4), 70, images:size(3), 0):type(dtype), b):contiguous()
      end
      if self.map_second_bottom == nil or self.map_second_bottom:size(1) ~= b then
        self.map_second_bottom = addBatchDim(vr.make_perspective_warp_map_top(w, 64, h, 0, 0):type(dtype), b):contiguous()
      end
      map_first = self.map_first_bottom
      map_second = self.map_second_bottom
      crop_net = nn.Sequential():add(nn.Narrow(3, 1, 64)):add(nn.Narrow(4, 65, -65)):type(dtype)
    end

    -- Hack: Geometric transformation maps are hardcoded for an image size of 384x384, so we need to make sure our image has those dimensions.
    imgs_pre_384 = imgs_pre
    if imgs_pre:size(2) ~= 384 or imgs_pre:size(3) ~= 384 then
      local scale_net = nn.SpatialUpSamplingBilinear({oheight=384, owidth=384}):type(dtype)
      imgs_pre_384 = scale_net:forward(imgs_pre)
    end
    
    local imgs1 = self.warpNet:forward( { imgs_pre_384, map_first } )
    local imgs1_cropped = crop_net:forward(imgs1):clone()
    table.insert(imgsList, imgs1_cropped)

    table.insert(flowList, map_second)
    
    local cert_pre = torch.ones(b, 1, images:size(3), images:size(4)):type(dtype)
    local cert_cropped = crop_net:forward(cert_pre):clone():contiguous()
    local cert = self.warpNet:forward( { cert_cropped, map_second } ):clone()
    table.insert(certList, cert)

    local scale_net = nn.SpatialUpSamplingBilinear({oheight=w, owidth=h}):type(dtype)
    local imgs2 = scale_net:forward(imgs_pre)
    table.insert(imgsList, imgs2)
        
  end
  
  return imgsList, flowList, certList
  
end

