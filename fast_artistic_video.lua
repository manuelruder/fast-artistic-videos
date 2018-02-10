require 'torch'
require 'nn'
require 'image'

--require 'fast_artistic_video.ShaveImage'
--require 'fast_artistic_video.TotalVariation'
--require 'fast_artistic_video.InstanceNormalization'
--require 'fast_artistic_video.PerceptualCriterion'
require 'fast_artistic_video_core'


local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'

local flowFile = require 'flowFileLoader'

--[[
Use a trained feedforward model to stylize an entire video.
--]]

local cmd = torch.CmdLine()

-- Main options
cmd:option('-model_img', 'models/checkpoint-candy-image.t7')
cmd:option('-model_vid', 'models/checkpoint-candy-video.t7')
cmd:option('-num_frames', 9999, 'maximum number of frames to process')
cmd:option('-continue_with', 1, 'Continue with this frame')

cmd:option('-input_pattern', '')
cmd:option('-output_prefix', 'out')

-- Optical flow and consistency
cmd:option('-flow_pattern', '')
cmd:option('-occlusions_pattern', '')
cmd:option('-invert_occlusion', false)
cmd:option('-occlusions_min_filter', 7, 'Workaround for artifacts around the occlusions')
cmd:option('-fill_occlusions', 'vgg-mean', 'uniform-random|vgg-mean')
cmd:option('-fix_occlusions', false, 'Workaround for incomplete Sintel gt occlusion pattern')
cmd:option('-median_filter', 3, 'Postprocessing filter')

-- Processing options 
cmd:option('-scale_factor', 1, 'Scale the image before processing')
cmd:option('-backward', false, 'Do stylization backwards, from the last frame to the first one.')
cmd:option('-create_inconsistent', false, 'Ignore any prior images, generate frame by frame')

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)

-- Evaluation
cmd:option('-evaluate', false, 'Whether to evaluate the consistency and perceptual quality of the outputs')
cmd:option('-flow_pattern_eval', '')
cmd:option('-occlusions_pattern_eval', '')
cmd:option('-invert_occlusion_eval', false)
cmd:option('-fix_occlusions_eval', false, 'Workaround for incomplete Sintel gt occlusion pattern')
cmd:option('-backward_eval', false, 'Perform evaluation in backward direction')
cmd:option('-evaluation_file', 'evaluation.txt')
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')
cmd:option('-style_image', 'images/styles/candy.jpg')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '1.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')


local function getFormatedFlowFileName(pattern, fromIndex, toIndex)
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return flowFileName
end

function fix_occlusions(flow, occluded, dtype)
  -- Find regions of the image that are left blank by the warping, i.e. there are no correspondences
  -- Fixes an issue in Sintel gt occlusions where occlusions at the boundaries are not marked as such. 
  local tmp = torch.ones(occluded:size()):type(dtype)
  tmp = utils.warp_image(tmp, flow, dtype)
  tmp:add(-0.5):sign():cmax(0)
  occluded:cmul(tmp)
end

-- Keep history of previous frames, needed for evaluation
local last_frame_stylized = nil
local prev_last_frame_stylized = nil
local last_frame = nil

function func_load_image(opt, i, dtype)
  if not utils.file_exists(string.format(opt.input_pattern, i)) then return nil end
  last_frame = image.load(string.format(opt.input_pattern, i), 3)
  return last_frame
end

function func_load_cert(opt, i, dtype) 
  local flowFileName = getFormatedFlowFileName(opt.flow_pattern, i - 1, i)
  local certFileName = getFormatedFlowFileName(opt.occlusions_pattern, i - 1, i)
  utils.wait_for_file(certFileName)
  local cert = image.load(certFileName, 1):type(dtype)
  if opt.invert_occlusion then
    cert:add(-1):mul(-1)
  end
  if opt.fix_occlusions then
    local flow = flowFile.load(flowFileName):float()
    fix_occlusions(flow, cert, dtype)     
  end
  return cert
end

function func_load_flow_cert_eval(opt, i, dtype)
  local flowFileName_eval = getFormatedFlowFileName(opt.flow_pattern_eval, i - 1, i)
  local certFileName_eval = getFormatedFlowFileName(opt.occlusions_pattern_eval, i - 1, i)
  local flow_eval = flowFile.load(flowFileName_eval):type(dtype)
  local cert_eval = image.load(certFileName_eval, 1):type(dtype)
  if opt.invert_occlusion_eval then
    cert_eval:add(-1):mul(-1)
  end
  if opt.fix_occlusions_eval then
    fix_occlusions(flow_eval, cert_eval, dtype)
  end
  return flow_eval, cert_eval
end

function func_eval(opt, i, func_percept_loss, dtype)
  style_loss, content_loss = func_percept_loss(last_frame, last_frame_stylized)
  if i > 1 then
    local W, H = last_frame_stylized:size(3), last_frame_stylized:size(2)
    flow_eval, cert_eval = func_load_flow_cert_eval(opt, i, dtype)
    local pixel_crit = nn.MSECriterion():type(dtype)
    local temporal_loss = nil
    if opt.backward_eval then
      -- Warp previous image for evaluation
      local prev_warped_eval = utils.warp_image(last_frame_stylized, flow_eval, dtype):type(dtype)
      temporal_loss = pixel_crit:forward(
        torch.cmul(prev_warped_eval, cert_eval:expand(3,H,W)),
        torch.cmul(prev_last_frame_stylized, cert_eval:expand(3,H,W)))
    else
      local prev_warped_eval = utils.warp_image(prev_last_frame_stylized, flow_eval, dtype):type(dtype)
      temporal_loss = pixel_crit:forward(
        torch.cmul(prev_warped_eval, cert_eval:expand(3,H,W)),
        torch.cmul(last_frame_stylized, cert_eval:expand(3,H,W)))
    end
    return { style_loss, content_loss, temporal_loss }, 3
  else
    return { style_loss, content_loss, 0 }, 3
  end
end

function func_make_last_frame_warped(opt, i, dtype)
  local flowFileName = getFormatedFlowFileName(opt.flow_pattern, i - 1, i)
  local flow = flowFile.load(flowFileName):type(dtype)
  local frame_warped = utils.warp_image(last_frame_stylized, flow, dtype):type(dtype)
  return frame_warped, nil
end

function func_save_image(opt, i, img)
  local out_path = string.format("%s-%05d.png", opt.output_prefix, i)
  print('Writing output image to ' .. out_path)
  local out_dir = paths.dirname(out_path)
  if not path.isdir(out_dir) then
    paths.mkdir(out_dir)
  end
  image.save(out_path, img)
  prev_last_frame_stylized = last_frame_stylized
  last_frame_stylized = img:clone()
end

function func_is_single_image(i, opt) return i == 1 or opt.create_inconsistent end

local function main()
  local opt = cmd:parse(arg)

  if (opt.input_pattern == '') then
    error('Must give -input_pattern')
  end
  if (not opt.create_inconsistent) and (opt.flow_pattern == '' or opt.occlusions_pattern == '') then
    error('Must give -flow_pattern and -occlusions_pattern')
  end
  if opt.gpu >= 0 and opt.backend == "cuda" then
    require 'stn'
  end
  
  run_fast_neural_video(opt, func_load_image, func_load_cert, func_eval, func_make_last_frame_warped, func_is_single_image, func_save_image)

end


main()
