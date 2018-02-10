require 'torch'
require 'nn'
require 'image'

require 'fast_artistic_video.ShaveImage'
require 'fast_artistic_video.TotalVariation'
require 'fast_artistic_video.InstanceNormalization'
require 'fast_artistic_video.PerceptualCriterion'

local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'

-- local optnet = require 'optnet'

-- General style transfer algorithm with an image prior, can be used for several applications like video style transfer
-- (prior = last image warped) or 360 deg videos in cube face format (prior = neighboring cube face border), etc.
-- opt should have the following properties:
--  opt.gpu [int]
--  opt.backend [string]
--  opt.use_cudnn [bool]
--  opt.cudnn_benchmark [int]
--  opt.model_img [string]
--  opt.model_vid [string]
--  opt.create_inconsistent [bool]
--  opt.evaluate [bool]
--  opt.content_layers [int,...,int]
--  opt.style_layers [int,...,int]
--  opt.loss_network [string]
--  opt.style_image [string]
--  opt.backward [bool]
--  opt.num_frames [int]
--  opt.occlusions_min_filter [int]
--  opt.continue_with [int]
function run_fast_neural_video(opt, func_load_image, func_load_cert, func_eval, func_make_last_frame_warped, func_is_single_image, func_save_image)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

  local function get_model(path)
    local ok, checkpoint = pcall(function() return torch.load(path) end)
    if not ok then
      print('ERROR: Could not load model from ' .. path)
      exit()
    end
    print('Model loaded.')
    local model = checkpoint.model
    model:evaluate()
    model:type(dtype)
    if use_cudnn then
      require 'cudnn'
      cudnn.convert(model, cudnn)
      if opt.cudnn_benchmark == 0 then
        cudnn.benchmark = false
        cudnn.fastest = true
      end
    end
    return model
  end
  
  local model_img = nil
  local model_vid = nil
  
  if opt.model_img ~= 'self' then
    model_img = get_model(opt.model_img)
    if not opt.create_inconsistent then
      model_vid = get_model(opt.model_vid)
    end
  else
    model_vid = get_model(opt.model_vid)
  end
  
  local preprocess_method = 'vgg'
  local preprocess = preprocess[preprocess_method]

  -- Set up the perceptual loss network
  local percep_crit = nil
  if opt.evaluate then
      -- Parse layer strings and weights
    opt.content_layers, opt.content_weights =
      utils.parse_layers(opt.content_layers, opt.content_weights)
    opt.style_layers, opt.style_weights =
      utils.parse_layers(opt.style_layers, opt.style_weights)
    local loss_net = torch.load(opt.loss_network)
    local crit_args = {
      cnn = loss_net,
      style_layers = opt.style_layers,
      style_weights = opt.style_weights,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
      agg_type = opt.style_target_type,
    }
    percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)
    -- Load the style image and set it
    local style_image = image.load(opt.style_image, 3, 'float')
    style_image = image.scale(style_image, opt.style_image_size)
    local H, W = style_image:size(2), style_image:size(3)
    style_image = preprocess.preprocess(style_image:view(1, 3, H, W))
    percep_crit:setStyleTarget(style_image:type(dtype))
  end

  -- Calculates style and content loss
  local function evaluate_image(content_img, stylized_img)
    local H, W = content_img:size(2), content_img:size(3)
    local target = {content_target=preprocess.preprocess(content_img:view(1, 3, H, W)):type(dtype)}
    percep_crit:forward(preprocess.preprocess(stylized_img:view(1, 3, H, W)):type(dtype), target)
    return percep_crit.total_content_loss, percep_crit.total_style_loss
  end

  local function generate_fill(b, c, h, w, cert)
    local rndTensor = torch.rand(b, c, h, w):type(cert:type())
    rndTensor = preprocess.preprocess(rndTensor)
    local cert_inv = torch.mul(torch.add(cert, -1), -1)
    if opt.fill_occlusions == 'vgg-mean' then
      return torch.zeros(b, c, h, w):type(cert:type())
    elseif opt.fill_occlusions == 'uniform-random' then
      return torch.cmul(rndTensor, cert_inv)
    end
  end

  --optimizedImgModel = false

  local function run_image(img)
    local time1 = os.clock()

    local H, W = img:size(2), img:size(3)
    local origType = img:type()
    -- Downscale the image if too large
    if opt.scale_factor ~= 1 then
      img = image.scale(img:float(), W * opt.scale_factor, H * opt.scale_factor, 'bicubic'):type(origType)
    end
    local img_pre = preprocess.preprocess(img:view(1, 3, H * opt.scale_factor, W * opt.scale_factor)):type(dtype)
    local img_out = nil

    if model_img == nil then
      -- In this case, the video model processes the frame
      local input_tmp = torch.cat(img_pre, generate_fill(1, 3, H, W, torch.zeros(img_pre:size()):type(dtype)), 2)
      -- Just mask everything as "uncertain" because we have no prior frame
      input_tmp = torch.cat(input_tmp,  torch.zeros(img_pre:size(1), 1, img_pre:size(3), img_pre:size(4)):type(dtype), 2)
      img_out = model_vid:forward(input_tmp)
    else
      --if not optimizedImgModel then
      --  local opts = {inplace=true, mode='inference', reuseBuffers=true, removeGradParams=true}
      --  optnet.optimizeMemory(model_img, img_pre, opts)
      --  optimizedImgModel = true
      --  print("optimized img model")
      --end
      img_out = model_img:forward(img_pre)
    end

    img_out = preprocess.deprocess(img_out)[1]
    if opt.scale_factor ~= 1 then
      img_out = image.scale(img_out:float(), W, H, 'bicubic'):type(dtype)
    end

    local time2= os.clock()
    print("Elapsed time for stylizing frame independently:" .. (time2 - time1))

    return img_out
  end

  -- Process a subsequent frame
  local function run_next_image(H, W, new_content_img, cert_mask, i)
    local prev_warped_pre, flow_mask = func_make_last_frame_warped(opt, i, dtype, cert_mask)
    
    local time1 = os.clock()
    
    local prev_warped = preprocess.preprocess(prev_warped_pre:view(1, 3, H, W))
    local prev_warped_masked = torch.cmul(prev_warped, cert_mask:expand(1, 3, H, W))
    local new_content_img_pre = preprocess.preprocess(new_content_img:view(1, 3, H, W)):type(dtype)
    local input_mask = flow_mask == nil and cert_mask or torch.cmin(cert_mask, flow_mask:view(1,1,H,W))
    local input = torch.cat(new_content_img_pre, torch.add(generate_fill(1, 3, H, W, cert_mask:expand(1, 3, H, W)), prev_warped_masked), 2)
    input = torch.cat(input, input_mask, 2)
    local img_out = model_vid:forward(input)
    local img_out = preprocess.deprocess(img_out)[1]
    
    local time2= os.clock()
    
    print("Elapsed time for stylizing frame:" .. (time2 - time1))

    return img_out
  end

  local eval_numbers_tabl, eval_numbers_sum_tabl = {}, {}
  local num_eval_numbers = 0
  local file = nil
  if opt.evaluate then
    file = io.open(opt.evaluation_file, "a")
  end

  local start_idx = opt.backward and opt.num_frames-1 or opt.continue_with
  local end_idx = opt.backward and 1 or opt.num_frames
  local inc = opt.backward and -1 or 1

  -- Main loop over all frames
  for i=start_idx, end_idx, inc do

    img = func_load_image(opt, i, dtype)
    if img == nil then break end
    local H, W = img:size(2), img:size(3)

    local next_img_stylized
    
    -- Process this frame
    if func_is_single_image(i, opt) then
      next_img_stylized = run_image(img)
    else
      local cert = func_load_cert(opt, i, dtype)   
      cert = utils.min_filter(cert, opt.occlusions_min_filter, dtype)
      next_img_stylized = run_next_image(H, W, img, cert:view(1,1,H,W), i)
    end

    func_save_image(opt, i, next_img_stylized, dtype)

    -- Evaluate style, content and temporal loss if requested
    if opt.evaluate then
      local numers_tabl = nil
      numers_tabl, num_eval_numbers = func_eval(opt, i, evaluate_image, dtype)
      for j=1,num_eval_numbers do
        if eval_numbers_tabl[j] == nil then
          table.insert(eval_numbers_tabl, tostring(numers_tabl[j]))
          table.insert(eval_numbers_sum_tabl, numers_tabl[j])
        else
          eval_numbers_tabl[j] = eval_numbers_tabl[j] .. ';' .. tostring(numers_tabl[j])
          eval_numbers_sum_tabl[j] =  eval_numbers_sum_tabl[j] + numers_tabl[j]
        end
      end
    end

    img_stylized = next_img_stylized
  end
  
  if opt.evaluate then
    for i=1,num_eval_numbers do
      file:write(eval_numbers_tabl[i], "\n")
    end
    for i=1,num_eval_numbers do
      file:write(eval_numbers_sum_tabl[i] / opt.num_frames, "\n")
    end
    file:close()
    print("File written")
  end

end
