require 'torch'
require 'hdf5'

local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'

local DataLoader_video_real = torch.class('DataLoader_video_real')


function DataLoader_video_real:__init(opt)
  assert(opt.h5_file_video, 'Must provide h5_file')
  assert(opt.batch_size, 'Must provide batch size')
  self.preprocess_fn = preprocess[opt.preprocessing].preprocess

  self.h5_file = hdf5.open(opt.h5_file_video, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  
  self.image_paths = {
    train = '/train',
    val = '/val',
  }
  
  self.img_size_h = opt.train_img_size:split(':')[1]
  self.img_size_w = opt.train_img_size:split(':')[2]
  
  local train_size = self.h5_file:read(self.image_paths.train .. "/frames1"):dataspaceSize()
  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val .. "/frames1"):dataspaceSize()[1],
  }
  self.num_channels = train_size[3]
  self.image_height = train_size[4]
  self.image_width = train_size[5]

  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end
  
  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end

  self.rgb_to_gray = torch.FloatTensor{0.2989, 0.5870, 0.1140}
end


function DataLoader_video_real:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader_video_real:setSplitIdx(split, it)
  self.split_idxs[split] = ((it - 1) * self.batch_size + 1) % (self.split_sizes[split] - (self.split_sizes[split] % self.batch_size) + self.batch_size)
  print('Resuming from mini-batch index ' .. self.split_idxs[split])
end

function DataLoader_video_real:getSplitIdx(split)
  return self.split_idxs[split]
end

function DataLoader_video_real:getBatch(split, sequence_length, dtype)
  local path_frames = self.image_paths[split] .. "/frames1"
  local path_flows = self.image_paths[split] .. "/flow"
  local path_certs = self.image_paths[split] .. "/cert"

  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
                        
  local frames_tabl, flows_tabl, certs_tabl = {}, {}, {}
   
  -- Load frames
  for i=1,sequence_length+1 do              
    local frames = self.h5_file:read(path_frames):partial(
                    {start_idx, end_idx},
                    {i, i},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)
    local frames_pre = self.preprocess_fn(frames:view(frames:size(1), self.num_channels, self.image_height, self.image_width)):type(dtype)
    table.insert(frames_tabl, frames_pre)
  end
        
  -- Load flow        
  for i=1,sequence_length do                
    local flow = self.h5_file:read(path_flows):partial(
                  {start_idx, end_idx},
                  {i, i},
                  {1, 2},
                  {1, self.image_height},
                  {1, self.image_width}):float()
    local flow_new = torch.Tensor(flow:size(1), flow:size(3), flow:size(4), flow:size(5))
    flow_new[{{},2,{},{}}] = flow[{{},1,1,{},{}}]
    flow_new[{{},1,{},{}}] = flow[{{},1,2,{},{}}]
    flow_new = flow_new:type(dtype):contiguous()
    table.insert(flows_tabl, flow_new)
  end
  
  -- Load occlusions (certainty)
  for i=1,sequence_length do                 
    local cert = self.h5_file:read(path_certs):partial(
                  {start_idx, end_idx},
                  {i, i},
                  {1, self.image_height},
                  {1, self.image_width}):type(dtype):div(255)        
    cert = cert:view(cert:size(1), 1, cert:size(3), cert:size(4))
    table.insert(certs_tabl, cert)
  end
   
  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end

  return frames_tabl, flows_tabl, certs_tabl
end

