require 'torch'
require 'nn'
require 'image'

require 'fast_artistic_video.ShaveImage'
require 'fast_artistic_video.TotalVariation'
require 'fast_artistic_video.InstanceNormalization'
require 'fast_artistic_video.PerceptualCriterion'
require 'fast_artistic_video_core'

local utils = require 'fast_artistic_video.utils'
local preprocess = require 'fast_artistic_video.preprocess'
local flowFile = require 'flowFileLoader'
local vr = require 'fast_artistic_video.vr_helper'

--[[
Use a trained feedforward model to stylize a spherical video.
--]]

local cmd = torch.CmdLine()

-- Input options
cmd:option('-input_pattern', '')
cmd:option('-flow_pattern', '')
cmd:option('-occlusions_pattern', '')
cmd:option('-model_img', '')
cmd:option('-model_vid', '')

-- Processing options
cmd:option('-start_frame', 1)
cmd:option('-continue_with', 1)
cmd:option('-num_frames', 9999)
cmd:option('-invert_occlusions', false)
cmd:option('-fix_occlusions', false, 'Workaround for incomplete Sintel gt occlusion pattern')
cmd:option('-occlusions_min_filter', 7)
cmd:option('-smooth_certainty', false)
cmd:option('-fill_occlusions', 'vgg-mean', 'uniform-random|vgg-mean')
cmd:option('-create_inconsistent', false)
cmd:option('-create_inconsistent_border', false)
cmd:option('-backward', false)
cmd:option('-overlap_pixel_w', 20)
cmd:option('-overlap_pixel_h', 20)

-- Output options
cmd:option('-output_prefix', 'out')
cmd:option('-out_equi_w', 768)
cmd:option('-out_equi_h', 768)
cmd:option('-out_equi', false)
cmd:option('-out_cubemap', false)
cmd:option('-median_filter', 3)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)

-- Evaluation
cmd:option('-evaluate', false)
cmd:option('-evaluation_file', 'evaluation.txt')
cmd:option('-no_consistency_eval', false)
cmd:option('-flow_pattern_eval', '')
cmd:option('-occlusions_pattern_eval', '')
cmd:option('-invert_occlusions_eval', false)
cmd:option('-backward_eval', false)
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')
cmd:option('-style_image', '')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '5.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')
cmd:option('-fix_occlusions_eval', false, 'Workaround for incomplete Sintel gt occlusion pattern')

-- Keep track of other stylized cube faces
last_content = nil
last_segments = {}
prev_last_segments = {}
hplus, wplus = nil, nil
h, w = nil, nil

-- Precomputed maps, so we only have to compute them once
mask_left = nil
mask_right = nil
mask_top = nil
mask_bottom = nil
mask_all = nil
mask_all_div = nil
grad_mask_left, grad_mask_right, grad_mask_top, grad_mask_bottom = nil, nil, nil, nil
grad_mask_left_right, grad_mask_all = nil, nil
warp_map_left, warp_map_right, warp_map_top, warp_map_bottom = nil, nil, nil, nil
equi_map = nil
initialized = false  -- Whether maps are initialized

-- Layout:
--     2
-- 3 6 4 5   
--     1
-- 
-- Order of processing:
-- 6, 1, 2, 5, 3, 4
local proc_order = { 6, 1, 2, 5, 3, 4 }


local tan30deg = 0.5773502692

local function getFormatedFlowFileName(pattern, fromIndex, toIndex, modeIdx)
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return string.format(flowFileName, modeIdx)
end

function fix_occlusions(flow, disoccluded)
  for x=1, flow:size(3) do
    for y=1, flow:size(2) do
      if     flow[1][y][x] < 0
          or flow[1][y][x] > flow:size(2)
          or flow[2][y][x] < 0
          or flow[2][y][x] > flow:size(3) then
        disoccluded[1][y][x] = 0
      end
    end
  end  
end

function reverse_tensor(t, n)
  return t:index(n ,torch.linspace(t:size(n),1,t:size(n)):long())
end

function rotate90(t)
  return reverse_tensor(t:transpose(2,3), 2)
end

function rotateMinus90(t)
  return reverse_tensor(t:transpose(2,3), 3)
end

function rotate180(t)
  return reverse_tensor(reverse_tensor(t, 2), 3)
end

function combineSides(side1, side2, side3, side4, divisor)
  local result = torch.cdiv(side1, divisor)
  result = torch.add(result, torch.cdiv(side2, divisor))
  result = torch.add(result, torch.cdiv(side3, divisor))
  result = torch.add(result, torch.cdiv(side4, divisor))
  return result
end

function func_load_image(opt, i, dtype)
  local mode = (i-1) % 6
  local file_idx = math.floor((i-1) / 6) + opt.start_frame
  
  print(string.format(opt.input_pattern, file_idx, mode+1))
  
  if not utils.file_exists(string.format(opt.input_pattern, file_idx, proc_order[mode+1])) then return nil end
  
  local full_image =  image.load(string.format(opt.input_pattern, file_idx, proc_order[mode+1]), 3)
  
  if not initialized then
    -- Create warping maps and blending masks to transform edges of neighboring cube faces
    hplus, wplus = full_image:size(2), full_image:size(3)
    h, w = hplus - opt.overlap_pixel_h, wplus - opt.overlap_pixel_w
  
    -- Warping maps
    warp_map_left = vr.make_perspective_warp_map_left(hplus, opt.overlap_pixel_w, wplus):type(dtype):contiguous()
    mask_left = utils.warp_image(torch.ones(1, hplus, wplus):type(dtype), warp_map_left, dtype)
    warp_map_top = vr.make_perspective_warp_map_top(wplus, opt.overlap_pixel_h, hplus):type(dtype):contiguous()
    mask_top = utils.warp_image(torch.ones(1, hplus, wplus):type(dtype), warp_map_top, dtype)    
    warp_map_bottom = vr.make_perspective_warp_map_bottom(wplus, opt.overlap_pixel_h, hplus):type(dtype):contiguous()
    mask_bottom = utils.warp_image(torch.ones(1, hplus, wplus):type(dtype), warp_map_bottom, dtype)  
    warp_map_right = vr.make_perspective_warp_map_right(hplus, opt.overlap_pixel_w, wplus):type(dtype):contiguous()
    mask_right = utils.warp_image(torch.ones(1, hplus, wplus):type(dtype), warp_map_right, dtype)   
    mask_all_div = torch.cmax(mask_left + mask_right + mask_top + mask_bottom, 1)
    mask_all = torch.cmin(mask_left + mask_right + mask_top + mask_bottom, 1)
       
    local grad_width_h = opt.overlap_pixel_h - 10
    local grad_width_w = opt.overlap_pixel_w - 10
    -- Smooth blending masks
    grad_mask_left = torch.cat({ utils.make_gradient_mask_w_dec(1, hplus, grad_width_w), torch.zeros(1, hplus, wplus - grad_width_w) }, 3)
    grad_mask_right = torch.cat({ torch.zeros(1, hplus, wplus - grad_width_w):double(), utils.make_gradient_mask_w_inc(1, hplus, grad_width_w):double() }, 3)
    grad_mask_top = torch.cat({ utils.make_gradient_mask_h_dec(1, grad_width_h, wplus):double(), torch.zeros(1, hplus - grad_width_h, wplus):double() }, 2)
    grad_mask_bottom = torch.cat({ torch.zeros(1, hplus - grad_width_h, wplus):double(), utils.make_gradient_mask_h_inc(1, grad_width_h, wplus):double() }, 2)
    grad_mask_all = torch.cmax( torch.cmax(grad_mask_left, grad_mask_right), torch.cmax(grad_mask_top, grad_mask_bottom) )
    grad_mask_left_right = torch.cmax(grad_mask_left, grad_mask_right)
    
    -- If we want to transform the cube faces into a equirectangular map, precompute the corresponding transformation map
    if opt.out_equi then
      local r = math.floor(opt.median_filter/2)
      equi_map = vr.make_cube_to_equirectangular_map(hplus - 2*r, wplus - 2*r, opt.overlap_pixel_w - r, opt.overlap_pixel_h - r, opt.out_equi_w, opt.out_equi_h):type(dtype)
    end
    
    initialized = true
  end
  
  last_content = full_image:contiguous()
  return full_image:contiguous()
end

function func_load_cert(opt, i, dtype) 
  local mode = (i-1) % 6
  local file_idx = math.floor((i-1) / 6) + opt.start_frame
  local cert = torch.zeros(1, hplus, wplus):type(dtype)  
  local cert_border = torch.zeros(1, hplus, wplus):type(dtype)  
  
  -- Make borders as certain since we have a certain correspondence for them from the neighboring cube face
  if not opt.create_inconsistent_border then
    if mode == 1 or mode == 3 or mode == 4 or mode == 5 then
      cert_border = torch.cmax(cert_border, mask_left)
    end
    if mode == 2 or mode == 3 or mode == 4 or mode == 5 then
      cert_border = torch.cmax(cert_border, mask_right)
    end
    if mode == 4 or mode == 5 then
      cert_border = torch.cmax(cert_border, mask_top)
    end
    if mode == 4 or mode == 5 then
      cert_border = torch.cmax(cert_border, mask_bottom)
    end
  end
  
  -- Certainty map for the rest of the image (occlusions)
  if i >= 7 and not opt.create_inconsistent then
    local certFileName = getFormatedFlowFileName(opt.occlusions_pattern, file_idx-1, file_idx, proc_order[mode+1])
    utils.wait_for_file(certFileName)
    local cert_frame = image.load(certFileName, 1):type(dtype)
    cert = torch.cmax(cert_frame, cert_border)
  else
    cert = cert_border
  end

  return cert
end

function func_make_last_frame_warped(opt, i, dtype, cert)  
  collectgarbage()
  local mode = (i-1) % 6
  local file_idx = math.floor((i-1) / 6) + opt.start_frame
  local border = torch.zeros(3, hplus, wplus):type(dtype)
  local result = torch.zeros(3, hplus, wplus):type(dtype)
  local gradMask = nil
  
  -- Neighboring cube faces
  if not opt.create_inconsistent_border then
    if mode == 1 then
      border = utils.warp_image(last_segments[1], warp_map_left, dtype)
      gradMask = grad_mask_right
    elseif mode == 2 then
      border = utils.warp_image(last_segments[1], warp_map_right, dtype)
      gradMask = grad_mask_left
    elseif mode == 3 then
      border = utils.warp_image(last_segments[2], warp_map_left, dtype)
      border = torch.add(border, utils.warp_image(last_segments[3], warp_map_right, dtype))
      gradMask = grad_mask_left_right
    elseif mode == 4 then
      border = torch.cdiv(utils.warp_image(rotate90(last_segments[2]), warp_map_left, dtype), mask_all_div:expand(3, hplus, wplus))
      border = torch.add(border, torch.cdiv(utils.warp_image(rotateMinus90(last_segments[3]), warp_map_right, dtype), mask_all_div:expand(3, hplus, wplus)))
      border = torch.add(border, torch.cdiv(utils.warp_image(last_segments[4], warp_map_top, dtype), mask_all_div:expand(3, hplus, wplus)))
      border = torch.add(border, torch.cdiv(utils.warp_image(rotate180(last_segments[1]), warp_map_bottom, dtype), mask_all_div:expand(3, hplus, wplus)))
      gradMask = grad_mask_all
    elseif mode == 5 then
      border = torch.cdiv(utils.warp_image(rotateMinus90(last_segments[2]), warp_map_left, dtype), mask_all_div:expand(3, hplus, wplus))
      border = torch.add(border, torch.cdiv(utils.warp_image(rotate90(last_segments[3]), warp_map_right, dtype), mask_all_div:expand(3, hplus, wplus)))
      border = torch.add(border, torch.cdiv(utils.warp_image(rotate180(last_segments[1]), warp_map_top, dtype), mask_all_div:expand(3, hplus, wplus)))
      border = torch.add(border, torch.cdiv(utils.warp_image(last_segments[4], warp_map_bottom, dtype), mask_all_div:expand(3, hplus, wplus)))
      gradMask = grad_mask_all
    end
  end

  -- Starting with the second cube face of the second frame, blend the last frame prior with neighboring cube face prior.
  if i >= 7 and not opt.create_inconsistent then
    local flowFileName = getFormatedFlowFileName(opt.flow_pattern, file_idx-1, file_idx, proc_order[mode+1])
    utils.wait_for_file(flowFileName)
    local flow = flowFile.load(flowFileName):type(dtype)
    local last_frame_warped = utils.warp_image(prev_last_segments[mode+1], flow, dtype)

    local cert_inv = torch.csub(torch.ones(cert:size()):type(cert:type()), cert)
    local grad_masks = { grad_mask_right, grad_mask_left, grad_mask_left_right, grad_mask_all, grad_mask_all }
    local masks = { mask_left, mask_right, mask_left + mask_right, mask_all, mask_all }
    if mode == 0 then
      result = last_frame_warped
    else
      local grad_mask = grad_masks[mode]:type(dtype)
      local mask = torch.cmul( torch.cmax(grad_mask, torch.ceil(grad_mask):cmul(cert_inv)), masks[mode] )
      local anti_mask = torch.csub(torch.ones(mask:size()):type(dtype), mask)
      result = torch.cmul(last_frame_warped, anti_mask:expand(3, hplus, wplus)) + torch.cmul(border, mask:expand(3, hplus, wplus))
    end
  else
    result = border
  end
  
  if opt.smooth_certainty then
    return result, gradMask:type(dtype):add(-0.5):cmax(0.0):sign():cmax(0.25)
  else
    return result
  end

end

function func_is_single_image(i, opt)
  if opt.create_inconsistent then
    return i % 6 == 1
  else
    return i == 1
  end
end

function evaluate_edge(img1, img2, edge)
  local loss = nn.MSECriterion():type(img1:type())
  if edge == 'left' then
    return loss:forward(img1[ { {}, {}, {1} } ], img2[ { {}, {}, {img2:size(3)} } ])
  elseif edge == 'top' then
    return loss:forward(img1[ { {}, {1}, {} } ], img2[ { {}, {img2:size(2)}, {} } ])
  end
end

function trim(t, opt)
  local oversize_w = opt.overlap_pixel_w/2
  local oversize_h = opt.overlap_pixel_h/2
  return t[ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ]
end

function evaluate_edge_top(img1, img2, edgeOther)
  local loss = nn.MSECriterion():type(img1:type())
  local side1 = img2[ { {}, {1}, {} } ]
  local side2 = nil
  if edgeOther == 'left' then
    side2 = img2[ { {}, {}, {1} } ]:transpose(2, 3)
  elseif edgeOther == 'right' then
    side2 = reverse_tensor(img2[ { {}, {}, {img2:size(3)} } ]:transpose(2, 3), 3)
  elseif edgeOther == 'top' then
    side2 = reverse_tensor(img2[ { {}, {1}, {} } ], 2)
  elseif edgeOther == 'bottom' then
    side2 = img2[ { {}, {img2:size(2)}, {} } ]
  end
  return loss:forward(side1, side2)
end

-- Evaluates gradients along cut edges.
function evaluate_gradient(img, mask)
  local conv_mask_x = torch.Tensor({{-1,0,1}})
  local conv_mask_y = torch.Tensor({{-1},{0},{1}})
  local gradient_x = 
      torch.cmax(
        torch.cmax(
          torch.conv2(img[1]:double(), conv_mask_x,'V'):abs(),
          torch.conv2(img[2]:double(), conv_mask_x,'V'):abs()),
        torch.conv2(img[3]:double(), conv_mask_x,'V'):abs())
  local gradient_y = 
      torch.cmax(
        torch.cmax(
          torch.conv2(img[1]:double(), conv_mask_y,'V'):abs(),
          torch.conv2(img[2]:double(), conv_mask_y,'V'):abs()),
        torch.conv2(img[3]:double(), conv_mask_y,'V'):abs())
  gradient_x:type(mask:type())
  gradient_y:type(mask:type())
  local gradient_mag = torch.sqrt(torch.add(torch.cmul(gradient_x[ { {2,gradient_x:size(1)-1}, {} } ], gradient_x[ { {2,gradient_x:size(1)-1}, {} } ]),
                                            torch.cmul(gradient_y[ { {}, {2,gradient_y:size(2)-1} } ], gradient_y[ { {}, {2,gradient_y:size(2)-1} } ])))
    
  local net1 = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1):type(mask:type())
  local net2 = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1):type(mask:type())
  
  local mask_gradient_x = net1:forward(torch.conv2(mask[1], conv_mask_x,'V'):abs():view(1,mask:size(2),mask:size(3)-2))
  local mask_gradient_y = net2:forward(torch.conv2(mask[1], conv_mask_y,'V'):abs():view(1,mask:size(2)-2,mask:size(3)))
  local mask_gradient_mag = torch.cmax(mask_gradient_x[ { {}, {2,mask_gradient_x:size(2)-1}, {} } ], mask_gradient_y[ { {}, {}, {2,mask_gradient_y:size(3)-1} } ])
  
  local masked_gradient_x = torch.cmul(gradient_x, mask_gradient_x)
  local masked_gradient_y = torch.cmul(gradient_y, mask_gradient_y)
  local masked_gradient_mag = torch.cmul(gradient_mag, mask_gradient_mag)
  
  local gradx_per_pixel_full = gradient_x:sum() / (gradient_x:size(1) * gradient_x:size(2))
  local grady_per_pixel_full = gradient_y:sum() / (gradient_y:size(1) * gradient_y:size(2))
  local gradmagg_per_pixel_full = (gradient_x:sum() + gradient_y:sum()) / (2*(gradient_mag:size(1) * gradient_mag:size(2)))
  
  local gradx_per_pixel_masked = masked_gradient_x:sum() / mask_gradient_x:sum()
  local grady_per_pixel_masked = masked_gradient_y:sum() / mask_gradient_y:sum()
  local gradmag_per_pixel_masked = (masked_gradient_x:sum() + masked_gradient_y:sum()) / (mask_gradient_x:sum() + mask_gradient_y:sum())
  
  local gradx_ratio, grady_ratio = gradx_per_pixel_masked / gradx_per_pixel_full, grady_per_pixel_masked / grady_per_pixel_full
  local gradmag_ratio = (gradx_ratio * mask_gradient_x:sum() + grady_ratio * mask_gradient_y:sum()) / (mask_gradient_x:sum() + mask_gradient_y:sum())
  
  return gradx_ratio, grady_ratio, gradmag_ratio
end

function load_flow_cert_eval(opt, file_idx, mode, dtype)
  local flowFileName_eval = getFormatedFlowFileName(opt.flow_pattern_eval == '' and opt.flow_pattern or opt.flow_pattern_eval, file_idx - 1, file_idx, proc_order[mode+1])
  local certFileName_eval = getFormatedFlowFileName(opt.occlusions_pattern_eval == '' and opt.occlusions_pattern or opt.occlusions_pattern_eval, file_idx - 1, file_idx, proc_order[mode+1])
  local flow_eval = flowFile.load(flowFileName_eval):type(dtype)
  local cert_eval = image.load(certFileName_eval, 1):type(dtype)
  if opt.invert_occlusions_eval then
    cert_eval:apply(function(x) return 1 - x end)
  end
  if opt.fix_occlusions_eval then
    fix_occlusions(flow_eval, cert_eval)
  end
  return flow_eval, cert_eval
end

function func_eval(opt, i, func_percept_loss, dtype)
  local mode = (i-1) % 6
  local file_idx = math.floor((i-1) / 6) + 1
  local gradx, grady, gradmag, edge = 0, 0, 0, 0
  if mode == 1 then
    gradx, grady, gradmag = evaluate_gradient(last_segments[2], utils.min_filter(mask_left, opt.reliable_map_min_filter, mask_left:type()):double())
    edge = evaluate_edge(trim(last_segments[1], opt), trim(last_segments[2], opt), 'left')
  elseif mode == 2 then
    gradx, grady, gradmag = evaluate_gradient(last_segments[3], utils.min_filter(mask_right, opt.reliable_map_min_filter, mask_right:type()):double())
    edge = evaluate_edge(trim(last_segments[3], opt), trim(last_segments[1], opt), 'left')
  elseif mode == 3 then
    gradx, grady, gradmag = evaluate_gradient(last_segments[4], utils.min_filter(torch.add(mask_right, mask_left), opt.reliable_map_min_filter, mask_right:type()):double())
    edge = (evaluate_edge(trim(last_segments[2], opt), trim(last_segments[4], opt), 'left')
          + evaluate_edge(trim(last_segments[2], opt), trim(last_segments[4], opt), 'left')) / 2
  elseif mode == 4 then
    gradx, grady, gradmag = evaluate_gradient(last_segments[mode+1], utils.min_filter(mask_all, opt.reliable_map_min_filter, mask_all:type()):double())
    edge = (evaluate_edge_top(trim(last_segments[1], opt), trim(last_segments[5], opt), 'top')
          + evaluate_edge_top(trim(last_segments[2], opt), trim(last_segments[5], opt), 'right')
          + evaluate_edge_top(trim(last_segments[3], opt), trim(last_segments[5], opt), 'left')
          + evaluate_edge_top(trim(last_segments[4], opt), trim(last_segments[5], opt), 'bottom')) / 4
  elseif mode == 5 then
    gradx, grady, gradmag = evaluate_gradient(last_segments[mode+1], utils.min_filter(mask_all, opt.reliable_map_min_filter, mask_all:type()):double())
  end
  
  local style_loss, content_loss = func_percept_loss(last_content, last_segments[mode+1])
  local temporal_loss = 0
  if i > 6 and not opt.no_consistency_eval then
    local pixel_crit = nn.MSECriterion():type(dtype)
    local flow_eval, cert_eval = load_flow_cert_eval(opt, file_idx, mode, dtype)
    local temporal_loss = nil
    if opt.backward_eval then
      -- Warp previous image for evaluation, this will also update the certainty mask
      local prev_warped_eval = utils.warp_image(last_segments[mode+1], flow_eval, dtype)
      temporal_loss = pixel_crit:forward(
        torch.cmul(prev_warped_eval, cert_eval:expand(3,hplus,wplus)),
        torch.cmul(prev_last_segments[mode+1], cert_eval:expand(3,hplus,wplus)))
    else
      local prev_warped_eval = utils.warp_image(prev_last_segments[mode+1], flow_eval, dtype)
      temporal_loss = pixel_crit:forward(
        torch.cmul(prev_warped_eval, cert_eval:expand(3,hplus,wplus)),
        torch.cmul(last_segments[mode+1], cert_eval:expand(3,hplus,wplus)))
    end
    return { gradx, grady, gradmag,  edge, style_loss, content_loss, temporal_loss }, 7
  elseif opt.no_consistency_eval then
    return { gradx, grady, gradmag, edge, style_loss, content_loss }, 6
  else
    return { gradx, grady, gradmag, edge, style_loss, content_loss, 0 }, 7
  end

end

function blend_other_sides(dtype)
  local result = {}
  local anti_mask = torch.csub(torch.ones(grad_mask_all:size()), grad_mask_all):type(dtype):expand(3, hplus, wplus)
  local mask = grad_mask_all:expand(3, hplus, wplus):type(dtype)
  
  local bordersFront = combineSides(
    utils.warp_image(last_segments[2], warp_map_right, dtype),
    utils.warp_image(last_segments[3], warp_map_left, dtype),
    utils.warp_image(rotate180(last_segments[5]), warp_map_bottom, dtype),
    utils.warp_image(rotate180(last_segments[6]), warp_map_top, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[1] = torch.cmul(last_segments[1], anti_mask) + torch.cmul(bordersFront, mask)
  
  local bordersLeft = combineSides(
    utils.warp_image(last_segments[1], warp_map_left, dtype),
    utils.warp_image(last_segments[4], warp_map_right, dtype),
    utils.warp_image(rotateMinus90(last_segments[5]), warp_map_bottom, dtype),
    utils.warp_image(rotate90(last_segments[6]), warp_map_top, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[2] = torch.cmul(last_segments[2], anti_mask) + torch.cmul(bordersLeft, mask)
  
  local bordersRight = combineSides(
    utils.warp_image(last_segments[1], warp_map_right, dtype),
    utils.warp_image(last_segments[4], warp_map_left, dtype),
    utils.warp_image(rotate90(last_segments[5]), warp_map_bottom, dtype),
    utils.warp_image(rotateMinus90(last_segments[6]), warp_map_top, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[3] = torch.cmul(last_segments[3], anti_mask) + torch.cmul(bordersRight, mask)
  
  local bordersBack = combineSides(
    utils.warp_image(last_segments[2], warp_map_left, dtype),
    utils.warp_image(last_segments[3], warp_map_right, dtype),
    utils.warp_image(last_segments[5], warp_map_bottom, dtype),
    utils.warp_image(last_segments[6], warp_map_top, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[4] = torch.cmul(last_segments[4], anti_mask) + torch.cmul(bordersBack, mask)
  
  local bordersTop = combineSides(
    utils.warp_image(rotate180(last_segments[1]), warp_map_bottom, dtype),
    utils.warp_image(rotate90(last_segments[2]), warp_map_left, dtype),
    utils.warp_image(rotateMinus90(last_segments[3]), warp_map_right, dtype),
    utils.warp_image(last_segments[4], warp_map_top, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[5] = torch.cmul(last_segments[5], anti_mask) + torch.cmul(bordersTop, mask)

  local bordersBottom = combineSides(
    utils.warp_image(rotate180(last_segments[1]), warp_map_top, dtype),
    utils.warp_image(rotateMinus90(last_segments[2]), warp_map_left, dtype),
    utils.warp_image(rotate90(last_segments[3]), warp_map_right, dtype),
    utils.warp_image(last_segments[4], warp_map_bottom, dtype),
    mask_all_div:expand(3, hplus, wplus):type(dtype))
  result[6] = torch.cmul(last_segments[6], anti_mask) + torch.cmul(bordersBottom, mask)

  
  return result
end

function func_save_image(opt, i, frame)
  local mode = (i-1) % 6
    
  -- Save image
  local oversize_w = opt.overlap_pixel_w/2 - math.floor(opt.median_filter/2)
  local oversize_h = opt.overlap_pixel_h/2 - math.floor(opt.median_filter/2)
  local file_idx = math.floor((i-1) / 6) + 1
  local out_path = opt.output_prefix .. file_idx .. "_" .. mode ..  ".png"
  local out_dir = paths.dirname(out_path)
  if not path.isdir(out_dir) then
    paths.mkdir(out_dir)
  end
  
  --print('Writing output image to ' .. out_path)
  --image.save(out_path, frame[ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ])
  --image.save(out_path, frame)

  last_segments[mode+1] = frame
  
  if mode == 5 then
    -- For the output, blend neighboring cube faces again to reduce artifacts
    prev_last_segments = blend_other_sides(frame:type())
    local min_filtered_sides = {}
    for j=1,6 do
      if opt.median_filter > 0 then
        table.insert(min_filtered_sides, utils.median_filter(prev_last_segments[j], opt.median_filter))
      else
        table.insert(min_filtered_sides, prev_last_segments[j])
      end
    end

    if opt.out_equi then
      local equi = utils.warp_image( torch.cat({min_filtered_sides[1], min_filtered_sides[2], min_filtered_sides[3], min_filtered_sides[4], rotate180(min_filtered_sides[5]), rotate180(min_filtered_sides[6])}, 3), equi_map, frame:type())
      image.save(string.format("%s-%05d_equi.png", opt.output_prefix, file_idx), equi)
    end
    if opt.out_cubemap then
      local cubemap = torch.cat( {
        min_filtered_sides[4][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ],
        min_filtered_sides[1][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ], 
        rotate90(min_filtered_sides[5][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ]),
        rotateMinus90(min_filtered_sides[6][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ]),
        min_filtered_sides[3][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ],
        min_filtered_sides[2][ { {}, {oversize_h+1,hplus-oversize_h}, {oversize_w+1,wplus-oversize_w} } ] }, 3)
      image.save(string.format("%s-%05d_cubemap.png", opt.output_prefix, file_idx), cubemap)
    end

  end

end

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

  opt.num_frames = opt.num_frames * 6
  opt.scale_factor = 1
  if opt.continue_with > 1 then
    for i=1,6 do
      local out_path = opt.output_prefix .. opt.continue_with .. "_" .. i-1 ..  ".png"
      prev_last_segments[i] = image.load(out_path, 3)
      prev_last_segments[i] = prev_last_segments[i]:cuda()
      prev_last_segments[i] = prev_last_segments[i]:contiguous()
    end
    opt.continue_with = (opt.continue_with-1) * 6 + 1
  end
  
  run_fast_neural_video(opt, func_load_image, func_load_cert, func_eval, func_make_last_frame_warped, func_is_single_image, func_save_image)

end


main()
