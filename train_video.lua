require 'torch'
require 'optim'
require 'image'
require 'stn'

require 'fast_artistic_video.DataLoader_video_fake'
require 'fast_artistic_video.DataLoader_video_real'
require 'fast_artistic_video.PerceptualCriterion'

local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'
local models = require 'fast_artistic_video.models_video'

local cmd = torch.CmdLine()

--[[
Train a feedforward style transfer model
--]]

-- Generic options
cmd:option('-arch', 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3')
cmd:option('-use_instance_norm', 1)
cmd:option('-h5_file', '')
cmd:option('-h5_file_video', '')
cmd:option('-padding_type', 'reflect-start')
cmd:option('-tanh_constant', 150)
cmd:option('-preprocessing', 'vgg')
cmd:option('-resume_from_checkpoint', '')
cmd:option('-image_model', '')

cmd:option('-data_mix', 'shift:1,zoom_out:1,video:3')
cmd:option('-num_frame_steps', '0:1')
cmd:option('-reliable_map_min_filter', 7)
cmd:option('-fill_occlusions', 'vgg-mean', 'uniform-random|vgg-mean')

cmd:option('-train_img_size', '256:256')

cmd:option('-single_image_until', 0)

-- Generic loss function options
cmd:option('-pixel_loss_type', 'L2', 'L2|L1|SmoothL1')
cmd:option('-pixel_loss_weight', 50.0)
cmd:option('-percep_loss_weight', 1.0)
cmd:option('-tv_strength', 1e-6)

-- Options for feature reconstruction loss
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')

-- Options for style reconstruction loss
cmd:option('-style_image', '')
cmd:option('-style_image_size', 384)
cmd:option('-style_weights', '10.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')

-- Optimization
cmd:option('-num_iterations', 60000)
cmd:option('-batch_size', 4)
cmd:option('-learning_rate', '1e-3')
cmd:option('-lr_decay_every', -1)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-weight_decay', 0)

-- Checkpointing and loss printing
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-checkpoint_every', 1000)
cmd:option('-history_every', 100)
cmd:option('-num_val_batches', 100)
cmd:option('-images_every', 100, 'Save network output every iteration being a multiple of the given number')
cmd:option('-print_every', 10, 'Print loss every <n> iterations.')


-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')


 function main()
  local opt = cmd:parse(arg)

  -- Parse layer strings and weights
  opt.content_layers, opt.content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  opt.style_layers, opt.style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)

  -- Figure out preprocessing
  if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]

  -- Figure out the backend
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

  local resume_from_iteration = 1
  
  -- Build the model
  local model = nil
  if opt.resume_from_checkpoint ~= '' then
    print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
    local checkpoint = torch.load(opt.resume_from_checkpoint)
    model = checkpoint.model:type(dtype)
    resume_from_iteration = checkpoint.iter + 1
  else
    print('Initializing model from scratch')
    model = models.build_model(opt):type(dtype)
  end
  if use_cudnn then cudnn.convert(model, cudnn) end
  model:training()
  
  -- Set up the pixel loss function
  local pixel_crit
  if opt.pixel_loss_weight > 0 then
    if opt.pixel_loss_type == 'L2' then
      pixel_crit = nn.MSECriterion():type(dtype)
    elseif opt.pixel_loss_type == 'L1' then
      pixel_crit = nn.AbsCriterion():type(dtype)
    elseif opt.pixel_loss_type == 'SmoothL1' then
      pixel_crit = nn.SmoothL1Criterion():type(dtype)
    end
  end

  -- Set up the perceptual loss function
  local percep_crit
  if opt.percep_loss_weight > 0 then
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

  -- Prepare data mix
  local data_mix_probs = {}
  local data_mix_wheel = {}
  local data_mix_unique = {}
  local current_data_source = ""
  local data_mix_count = 0
  local use_fake_videos = false
  local use_real_videos = false
  for _, v in ipairs(opt.data_mix:split(',')) do
    local str_split = v:split(':')
    local data_source, count = str_split[1], tonumber(str_split[2])
    data_mix_probs[data_source] = count
    data_mix_count =  data_mix_count + count
    for i = 1,count do table.insert(data_mix_wheel, data_source) end
    table.insert(data_mix_unique, data_source)
    use_fake_videos = use_fake_videos or data_source ~= 'video'
    use_real_videos = use_real_videos or data_source == 'video'
  end
  
  -- Prepare multi frame steps
  local tabl_frame_steps = {}
  local tabl_frame_steps_size = 0
  for _, v in ipairs(opt.num_frame_steps:split(',')) do
    local str_split = v:split(':')
    local iter, num = tonumber(str_split[1]), tonumber(str_split[2])
    table.insert(tabl_frame_steps, { iter=iter, num=num })
    tabl_frame_steps_size = tabl_frame_steps_size + 1
  end
  
  -- Prepare learning rates
  local tabl_learning_rates = {}
  local tabl_learning_rates_size = 1
  local learning_rate_split = tostring(opt.learning_rate):split(',')
  table.insert(tabl_learning_rates, { iter=0, rate=tonumber(learning_rate_split[1]) })
  for i=2,#learning_rate_split do
    local str_split = learning_rate_split[i]:split(':')
    local iter, rate = tonumber(str_split[1]), tonumber(str_split[2])
    table.insert(tabl_learning_rates, { iter=iter, rate=rate })
    tabl_learning_rates_size = tabl_learning_rates_size + 1
  end

  -- Prepare data loaders
  local loader, loader_real = nil, nil
  if use_fake_videos then
    loader = DataLoader_video_fake(opt)
  end
  if use_real_videos then
    loader_real = DataLoader_video_real(opt)
  end
  
  local warpNet = nn.BilinearSamplerBDHW():type(dtype)
  
  paths.mkdir("debug")
  
  local params, grad_params = model:getParameters()
  
  -- Prepare pretrained image model
  local finishedModel = nil
  if opt.image_model ~= 'self' then
    local ok, finishedCheckpoint = pcall(function() return torch.load(opt.image_model) end)
    if not ok then
      print('ERROR: Could not load single-image model from ' .. opt.image_model)
      return
    end
    finishedModel = finishedCheckpoint.model
    finishedModel:evaluate()
    finishedModel:type(dtype)
    if use_cudnn then cudnn.convert(finishedModel, cudnn) end
  end
  
  local iteration = 0
  local num_frame_steps = 1
  local optimized, optimizedImgModel = false, false
  
  local function get_next_data_source()
    if iteration < opt.single_image_until then
      return 'single_image'
    else
      -- Select a random data source
      local idx = math.floor(math.random() * data_mix_count) + 1
      return data_mix_wheel[idx]
    end
  end
  
  local function generate_antimask(b, c, h, w, img, cert)
    local cert_inv = torch.mul(torch.add(cert, -1), -1)
    if opt.fill_occlusions == 'vgg-mean' then
      return torch.zeros(b, c, h, w):type(cert:type())
    elseif opt.fill_occlusions == 'uniform-random' then
      local rndTensor = torch.rand(b, c, h, w):type(cert:type())
      rndTensor = preprocess.preprocess(rndTensor)
      return torch.cmul(rndTensor, cert_inv)
    end
  end
  
  local function f(x)
    assert(x == params)
    grad_params:zero()
    
    current_data_source = get_next_data_source()

    local imgsList, flowList, certList = nil, nil, nil

    -- Determine number of steps
    for i=1,tabl_frame_steps_size do
      if iteration > tabl_frame_steps[i].iter then num_frame_steps = tabl_frame_steps[i].num else break end
    end
    -- If single image, set to 1 nevertheless
    local num_frame_steps_local = current_data_source == 'single_image' and 1 or num_frame_steps

    if current_data_source == 'video' then
      imgsList, flowList, certList = loader_real:getBatch('train', num_frame_steps, dtype)
    else  -- Either from single image or vr
      imgsList, flowList, certList = loader:getBatch('train', current_data_source, num_frame_steps, dtype)
      if current_data_source == 'vr' then 
        num_frame_steps_local = 1
      end
    end

    local b, c, h, w = imgsList[1]:size(1), imgsList[1]:size(2), imgsList[2]:size(3), imgsList[2]:size(4)
    
    for i=1,num_frame_steps_local do
      certList[i] = utils.min_filter(certList[i], opt.reliable_map_min_filter, dtype)
    end

    -- Create the stylized version of frame 1
    local out1 = nil
    if current_data_source == 'single_image' then
      out1 = torch.zeros(b, c, h, w):type(dtype)
    elseif finishedModel == nil then
      -- TODO: Crashes if using reflection padding and the very first model input is a VR image
      local input_tmp = torch.cat(imgsList[1], torch.zeros(b, c+1, imgsList[1]:size(3), imgsList[1]:size(4)):type(dtype), 2)     
      out1 = model:forward(input_tmp)
    else
      out1 = finishedModel:forward(imgsList[1])
    end
    
    -- Generate next frames stylized
    local out2 = nil
    local out1_warped_masked = nil
    
    for i=1,num_frame_steps_local do 
      -- Copy result from last stylization    
      if out2 ~= nil then out1 = out2:clone() end
      out1:contiguous()
      -- Warp last frame
      local out1_warped = warpNet:forward({out1, flowList[i]:contiguous()})
      -- Mask last frame with occlusions
      out1_warped_masked = torch.cmul(out1_warped, certList[i]:expand(b, c, h, w))
      -- How to fill the occlusions
      local antimask = generate_antimask(b, c, h, w, out1_warped_masked, certList[i]:expand(b, c, h, w))
      
      -- Save debug images
      if (iteration % opt.images_every == 1) then
        image.save('debug/out' .. i .. '.png', preprocess.deprocess(out1)[1])
        image.save('debug/out' .. i .. '_warped.png', preprocess.deprocess(out1_warped)[1])
        image.save('debug/out' .. i .. '_warped_masked.png', preprocess.deprocess(out1_warped_masked)[1])
        image.save('debug/in' .. i .. '.png', preprocess.deprocess(imgsList[i])[1])
        image.save('debug/mask' .. i .. '.png', certList[i][1]:float())
      end
      
      -- Create next frame
      local input = torch.cat(imgsList[i+1], torch.add(out1_warped_masked, antimask), 2)
      input = torch.cat(input, certList[i], 2)
           
      out2 = model:forward(input)
      -- This is a bit of a hack: if we are using reflect-start padding and the
      -- output is not the same size as the input, lazily add reflection padding
      -- to the start of the model so the input and output have the same size.
      if opt.padding_type == 'reflect-start' and h ~= out2:size(3) then
        local ph = (h - out2:size(3)) / 2
        local pw = (w - out2:size(4)) / 2
        local pad_mod = nn.SpatialReflectionPadding(pw, pw, ph, ph):type(dtype)
        model:insert(pad_mod, 1)
        out2 = model:forward(input)
      end
    end

    -- Mask frame 2
    local out2_masked = torch.cmul(out2, certList[num_frame_steps_local]:expand(b, c, h, w))

    -- Save debug images
    if (iteration % opt.images_every == 1) then
      image.save('debug/out' .. num_frame_steps_local+1 .. '.png', preprocess.deprocess(out2)[1])
      image.save('debug/out' .. num_frame_steps_local+1 .. '_masked.png', preprocess.deprocess(out2_masked)[1])
      image.save('debug/in' .. num_frame_steps_local+1 .. '.png', preprocess.deprocess(imgsList[num_frame_steps_local+1])[1])
    end

    local grad_out = nil
    -- Compute perceptual loss and gradient
    local percep_loss = 0
    if percep_crit then
      local target = {content_target=imgsList[num_frame_steps_local+1]}
      percep_loss = percep_crit:forward(out2, target)
      percep_loss = percep_loss * opt.percep_loss_weight
      local grad_out_percep = percep_crit:backward(out2, target)
      if grad_out then
        grad_out:add(opt.percep_loss_weight, grad_out_percep)
      else
        grad_out_percep:mul(opt.percep_loss_weight)
        grad_out = grad_out_percep
      end
    end
    
    -- Compute pixel loss (to previous frame warped) and gradient
    local pixel_loss = 0
    if pixel_crit then
      local pixel_loss = pixel_crit:forward(out2_masked, out1_warped_masked)
      pixel_loss = pixel_loss * opt.pixel_loss_weight
      local grad_out_pix = pixel_crit:backward(out2_masked, out1_warped_masked)
      if grad_out then
        grad_out:add(opt.pixel_loss_weight, grad_out_pix)
      else
        grad_out_pix:mul(opt.pixel_loss_weight)
        grad_out = grad_out_pix
      end
    end
    
    local loss = pixel_loss + percep_loss

    -- Run model backward
    local input = torch.cat(imgsList[num_frame_steps_local+1], out1_warped_masked, 2)
    input = torch.cat(input, certList[num_frame_steps_local], 2)
    model:backward(input, grad_out)

    -- Add regularization
    -- grad_params:add(opt.weight_decay, params)
 
    return loss, grad_params
  end

  local optim_state = {learningRate=opt.learning_rate}
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_last_history = {}
  local val_loss_history_ts = {}
  local percept_loss_history = nil
  percept_loss_history = {}
  for i, k in ipairs(opt.style_layers) do
    percept_loss_history[string.format('style-%d', k)] = {}
  end
  for i, k in ipairs(opt.content_layers) do
    percept_loss_history[string.format('content-%d', k)] = {}
  end
  local total_loss_avg, style_loss_avg, content_loss_avg = 0, {}, {}

  local style_weight = opt.style_weight
  for t = resume_from_iteration, opt.num_iterations do
    iteration = t
    -- Determine learning rate
    for i=1,tabl_learning_rates_size do
      if iteration > tabl_learning_rates[i].iter then optim_state.learningRate = tabl_learning_rates[i].rate else break end
    end
    
    local _, loss = optim.adam(f, params, optim_state)

    if t % opt.print_every == 0 then
      print(string.format('Iteration %d / %d, loss = %f',
            t, opt.num_iterations, loss[1]))
    end

    -- Accumulate losses
    total_loss_avg = total_loss_avg + loss[1]
    for i, k in ipairs(opt.style_layers) do
      style_loss_avg[string.format('style-%d', k)] = 
        (style_loss_avg[string.format('style-%d', k)] or 0) + percep_crit.style_losses[i]
    end
    for i, k in ipairs(opt.content_layers) do
      content_loss_avg[string.format('content-%d', k)] =
        (content_loss_avg[string.format('content-%d', k)] or 0) + percep_crit.content_losses[i]
    end
 
    -- Inseret losses into tables
    if t % opt.history_every == 0 then
      table.insert(train_loss_history, total_loss_avg / opt.history_every)
      total_loss_avg = 0
      for i, k in ipairs(opt.style_layers) do
        table.insert(percept_loss_history[string.format('style-%d', k)],
          style_loss_avg[string.format('style-%d', k)] / opt.history_every)
        style_loss_avg[string.format('style-%d', k)] = 0
      end
      for i, k in ipairs(opt.content_layers) do
        table.insert(percept_loss_history[string.format('content-%d', k)],
          content_loss_avg[string.format('content-%d', k)] / opt.history_every)
        content_loss_avg[string.format('content-%d', k)] = 0
      end
    end

    if t % opt.checkpoint_every == 0 then
      -- Check loss on the validation set
      if loader ~= nil then
        loader:reset('val')
      end
      loader_real:reset('val')
      model:evaluate()
      local val_loss, val_loss_last = 0, 0
      print 'Running on validation set ... '
      local val_batches = opt.num_val_batches
      for j = 1, val_batches do
        local val_loss_part, val_loss_last_part = 0, 0
        for _,data_mix_value in ipairs(data_mix_unique) do
          local num_frame_steps_local = data_mix_value == 'single_image' and 1 or tabl_frame_steps[tabl_frame_steps_size].num
          if data_mix_value == 'video' then
            imgsList, flowList, certList = loader_real:getBatch('val', num_frame_steps_local, dtype)
          else
            imgsList, flowList, certList = loader:getBatch('val', data_mix_value, num_frame_steps_local, dtype)
            if data_mix_value == 'vr' then num_frame_steps_local = 1 end
          end
          local b, c, h, w = imgsList[1]:size(1), imgsList[1]:size(2), imgsList[2]:size(3), imgsList[2]:size(4)
          for i=1,num_frame_steps_local do
            certList[i] = utils.min_filter(certList[i], opt.reliable_map_min_filter, dtype)
          end
          local out1 = nil
          if current_data_source == 'single_image' then
            out1 = torch.zeros(b, c, h, w):type(dtype)
          elseif finishedModel == nil then
            local input_tmp = torch.cat(imgsList[1], torch.zeros(b, c+1, imgsList[1]:size(3), imgsList[1]:size(4)):type(dtype), 2)      
            out1 = model:forward(input_tmp)
          else
            out1 = finishedModel:forward(imgsList[1])
          end
          local out2, out1_warped_masked = nil, nil
          local pixel_loss, percep_loss, pixel_loss_last, percep_loss_last  = 0, 0
          for i=1,num_frame_steps_local do
            if out2 ~= nil then out1 = out2:clone() end
            out1:contiguous()
            local out1_warped = warpNet:forward({out1, flowList[i]:contiguous()})
            local out1_warped_masked = torch.cmul(out1_warped, certList[i]:expand(b, c, h, w))
            local input = torch.cat(imgsList[i+1], out1_warped_masked, 2)
            input = torch.cat(input, certList[i], 2)
            out2 = model:forward(input)
            local out2_masked = torch.cmul(out2, certList[num_frame_steps_local]:expand(b, c, h, w))
            if pixel_crit then
              pixel_loss_last = pixel_crit:forward(out2_masked, out1_warped_masked) * pixel_loss
              pixel_loss = pixel_loss
                + pixel_loss_last
            end
            if percep_crit then
             percep_loss_last = percep_crit:forward(out2, {content_target=imgsList[i+1]}) * opt.percep_loss_weight
              percep_loss = percep_loss
                + percep_loss_last
            end
          end
          val_loss_last_part = val_loss_last_part + (data_mix_probs[data_mix_value] * (percep_loss_last + pixel_loss_last))
          val_loss_part = val_loss_part + (data_mix_probs[data_mix_value] * (percep_loss + pixel_loss)) / num_frame_steps_local
        end
        val_loss = val_loss + val_loss_part / data_mix_count
        val_loss_last = val_loss_last + val_loss_last_part / data_mix_count
      end

      val_loss = val_loss / val_batches
      print(string.format('val loss = %f', val_loss))
      table.insert(val_loss_history, val_loss)
      table.insert(val_loss_last_history, val_loss_last)
      table.insert(val_loss_history_ts, t)
      model:training()

      -- Save a JSON checkpoint
      local checkpoint = {
        opt=opt,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_loss_last_history=val_loss_last_history,
        val_loss_history_ts=val_loss_history_ts,
        percept_loss_history=percept_loss_history,
        iter=t,
      }
      local filename = string.format('%s.json', opt.checkpoint_name)
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)

      collectgarbage()
      
      -- Save a torch checkpoint; convert the model to float first
      model:clearState()
      
      collectgarbage()
      
      if use_cudnn then
        cudnn.convert(model, nn)
      end
      model:float()
      checkpoint.model = model
      filename = string.format('%s_%d.t7', opt.checkpoint_name, num_frame_steps)
      torch.save(filename, checkpoint)
      
      -- Convert the model back
      model:type(dtype)
      if use_cudnn then
        cudnn.convert(model, cudnn)
      end
      params, grad_params = model:getParameters()
      optimized = false
      
      collectgarbage()
    end

    if opt.lr_decay_every > 0 and t % opt.lr_decay_every == 0 then
      local new_lr = opt.lr_decay_factor * optim_state.learningRate
      optim_state = {learningRate = new_lr}
    end

  end

end


main()

