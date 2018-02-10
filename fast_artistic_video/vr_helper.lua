local M = {}

function M.make_perspective_warp_map_left(height, crop_w, orig_width, oversize_h, oversize_w)
  if oversize_h == nil then oversize_h = crop_w / 2 end
  if oversize_w == nil then oversize_w = crop_w / 2 end
  local width = height / 2 / ((2*oversize_h+height)/height)
  local max_resize_factor = (width+oversize_h)/width
  width = width - (max_resize_factor-1)/max_resize_factor * oversize_h
  
  local map = torch.Tensor(2, height, orig_width):fill(99999)
  local mid_y = height / 2
  
  for x=width-crop_w+1,width do
    local resize_factor_h = (x+oversize_h)/width
    local resize_factor_w = (x+oversize_w)/width
    for y=1,height do
      map[1][y][x-(width-crop_w)+orig_width-crop_w] = (mid_y-y) * (-1 / resize_factor_h + 1)
      map[2][y][x-(width-crop_w)+orig_width-crop_w] = (width-x-oversize_w) * (resize_factor_w-1)/resize_factor_w -orig_width+crop_w
    end
  end
  
  return map
end

function M.make_perspective_warp_map_right(height, crop_w, org_width, oversize_h, oversize_w)
  if oversize_h == nil then oversize_h = crop_w / 2 end
  if oversize_w == nil then oversize_w = crop_w / 2 end
  local width = height / 2 / ((2*oversize_h+height)/height)
  local max_resize_factor = (width+oversize_h)/width
  width = width - (max_resize_factor-1)/max_resize_factor * oversize_h
  
  local map = torch.Tensor(2, height, org_width):fill(99999)
  local mid_y = height / 2
  
  for x=1,crop_w do
    local resize_factor_h = (width-x+oversize_h)/width
    local resize_factor_w = (width-x+oversize_w)/width
    for y=1,height do
      map[1][y][x] = (mid_y-y) * (-1 / resize_factor_h + 1)
      map[2][y][x] = -(x-oversize_w) * (resize_factor_w-1)/resize_factor_w + org_width-crop_w
    end
  end
  
  return map
end

function M.make_perspective_warp_map_top(width, crop_h, orig_height, oversize_w, oversize_h)
  if oversize_h == nil then oversize_h = crop_h / 2 end
  if oversize_w == nil then oversize_w = crop_h / 2 end
  local height = width / 2 / ((2*oversize_w+width)/width)
  local max_resize_factor = (height+oversize_w)/height
  height = height - (max_resize_factor-1)/max_resize_factor * oversize_w
  
  
  local map = torch.Tensor(2, orig_height, width):fill(99999)
  local mid_x = width / 2
  
  for y=height-crop_h+1,height do
    local resize_factor_w = (y+oversize_w)/height
    local resize_factor_h = (y+oversize_h)/height
    for x=1,width do
      map[1][y-(height-crop_h)+orig_height-crop_h][x] = (height-y-oversize_h) * (resize_factor_h-1)/resize_factor_h -orig_height+crop_h
      map[2][y-(height-crop_h)+orig_height-crop_h][x] = (mid_x-x) * (-1 / resize_factor_w + 1)
    end
  end
  
  return map
end


function M.make_perspective_warp_map_bottom(width, crop_h, orig_height, oversize_w, oversize_h)
  if oversize_h == nil then oversize_h = crop_h / 2 end
  if oversize_w == nil then oversize_w = crop_h / 2 end
  local height = width / 2 / ((2*oversize_w+width)/width)
  local max_resize_factor = (height+oversize_w)/height
  height = height - (max_resize_factor-1)/max_resize_factor * oversize_w
  
  
  local map = torch.Tensor(2, orig_height, width):fill(99999)
  local mid_x = width / 2
  
  for y=1,crop_h do
    local resize_factor_w = (height-y+oversize_w)/height
    local resize_factor_h = (height-y+oversize_h)/height
    for x=1,width do
      map[1][y][x] = -(y-oversize_h) * (resize_factor_h-1)/resize_factor_h + orig_height-crop_h
      map[2][y][x] = (mid_x-x) * (-1 / resize_factor_w + 1)
    end
  end
  
  return map
end

-- Source: https://stackoverflow.com/a/34427087
function M.make_cube_to_equirectangular_map(w_plus_overlap, h_plus_overlap, overlap_w, overlap_h, out_w, out_h)
  
  local map = torch.Tensor(2, out_h, out_w)
  
  local cubeFaceWidth = w_plus_overlap - overlap_w
  local cubeFaceHeight = h_plus_overlap - overlap_h
  
  --u,v: Normalised texture coordinates, from 0 to 1, starting at lower left corner
  --phi, theta: Polar coordinates
  for j=0, out_h-1 do
    --Rows start from the bottom
    local v = 1 - (j / out_h)
    local theta = v * math.pi

    for i=0,out_w-1 do
      --Columns start from the left
      local u = (i / out_w)
      local phi = u * 2 * math.pi

      --Unit vector
      local x = math.sin(phi) * math.sin(theta) * -1
      local y = math.cos(theta)
      local z = math.cos(phi) * math.sin(theta) * -1

      local a = math.max(math.abs(x), math.abs(y), math.abs(z))
      --Vector Parallel to the unit vector that lies on one of the cube faces
      local xa, ya, za = x / a, y /a, z / a

      local xPixel, yPixel
      local xOffset, yOffset

      -- Assuming layout f, l, r, b, u, d
      if xa == 1 then
        --Right
        xPixel = (((za + 1) / 2) - 1) * cubeFaceWidth
        xOffset = 2 * w_plus_overlap
        yPixel = ((ya + 1) / 2) * cubeFaceHeight
        yOffset = 0
      elseif xa == -1 then
        --Left
        xPixel = ((za + 1) / 2) * cubeFaceWidth
        xOffset = 1 * w_plus_overlap
        yPixel = (((ya + 1) / 2)) * cubeFaceHeight
        yOffset = 0
      elseif ya == 1 then
        --Up
        xPixel = ((xa + 1) / 2) * cubeFaceWidth
        xOffset = 5 * w_plus_overlap
        yPixel = (((za + 1) / 2) - 1) * cubeFaceHeight
        yOffset = 0
      elseif ya == -1 then
        --Down
        xPixel = ((xa + 1) / 2) * cubeFaceWidth
        xOffset = 4 * w_plus_overlap
        yPixel = ((za + 1) / 2) * cubeFaceHeight
        yOffset = 0
      elseif za == 1 then
        --Front
        xPixel = ((xa + 1) / 2) * cubeFaceWidth
        xOffset = 0 * w_plus_overlap
        yPixel = ((ya + 1) / 2) * cubeFaceHeight
        yOffset = 0
      elseif za == -1 then
        --Back
        xPixel = (((xa + 1) / 2) - 1) * cubeFaceWidth
        xOffset = 3 * w_plus_overlap
        yPixel = ((ya + 1) / 2) * cubeFaceHeight
        yOffset = 0
      else
          print("Unknown face, something went wrong")
          xPixel = 0
          yPixel = 0
          xOffset = 0
          yOffset = 0
      end

      xPixel = math.abs(xPixel)
      yPixel = math.abs(yPixel)

      xPixel = xPixel + xOffset + overlap_w/2
      yPixel = yPixel + yOffset + overlap_h/2

      map[1][j+1][i+1] = yPixel - j
      map[2][j+1][i+1] = xPixel - i
    end
  end
  
  return map

end


return M
