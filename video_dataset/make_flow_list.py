from scipy.misc import imread, imresize
import os
import sys

if (len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help")) or len(sys.argv) < 2:
  print("Usage: make_flow_list.py <folder_to_extracted_dataset> <output_folder> [<num_tuples_per_scene> [<num_frames_per_tuble>]")
  print("Selects and extracts <num_tuples_per_scene> tuples consisting of <num_frames_per_tuble> consecutive frames from each scene of the hollywood2 dataset by amount of motion in the scene and creates a flownet2 compatible list of optical flows to compute.")
  
  sys.exit(1)

folder = sys.argv[1]
out_folder = sys.argv[2]

frames_folder = out_folder
videos_folder = os.path.join(folder, "AVIClipsScenes")
bounds_folder = os.path.join(folder, "ShotBoundsScenes")
flow_folder = out_folder
n_frames = 4  # Number of consecutive frames per sample.
n_tuples = 5  # Number of tuples selected per scene.
if len(sys.argv) > 3:
  n_tuples = int(sys.argv[3])
if len(sys.argv) > 4:
  n_frames = int(sys.argv[4])-1

if not os.path.exists(out_folder): os.mkdir(out_folder)
if not os.path.exists(flow_folder): os.mkdir(flow_folder)

opt_flow_list = []

for videofile in os.listdir(videos_folder):
  if not videofile.endswith('.avi'):
    continue
  subfolder = videofile[:-4]
  bounds_file = os.path.join(bounds_folder, subfolder + ".sht")
  file = open(bounds_file)
  bounds = [int(x)+1 for x in file.read().strip().split(' ') if len(x) > 0]
  file.close()
  
  frame_subfolder = os.path.join(frames_folder, subfolder)
  if not os.path.exists(frames_folder): os.mkdir(frames_folder)
  if not os.path.exists(frame_subfolder): os.mkdir(frame_subfolder)

  os.system('ffmpeg -i "%s" -vf "scale=-1:256,scale=\'max(in_w,384)\':-1,crop=384:256:(in_w-384)/2:(in_h-256)/2" "%s/frame_%%04d.png"' % (os.path.join(videos_folder, videofile), frame_subfolder))
  
  num_frames = len([name for name in os.listdir(frame_subfolder) if os.path.isfile(os.path.join(frame_subfolder, name))])
  bounds.append(num_frames)
  bounds = [ 1 ] + bounds
  
  keep_frames = []
  
  # Make output flow directory
  if not os.path.exists(os.path.join(flow_folder,  subfolder, 'flow')):
    os.mkdir(os.path.join(flow_folder,   subfolder, 'flow'))
  
  # Loop over all scenes
  for i in range(1, len(bounds)):
    diffs = []
    # Loop over all frames in that scene
    for j in range(bounds[i-1], bounds[i]-n_frames, n_frames):
      first_frame = imread( os.path.join(frames_folder, subfolder, "frame_%04d.png" % j ) )
      last_frame = imread( os.path.join(frames_folder, subfolder, "frame_%04d.png" % (j + n_frames) ) )
      diff = (first_frame - last_frame).sum()
      diffs.append((j, diff))
    diffs = sorted(diffs, key=lambda x : x[1])
    # Loop over the n_tuples most differing tuples
    for j in range(min(n_tuples, len(diffs))):
      # Loop over the individual frames in that tuple
      for k in range(n_frames):
        keep_frames.append( diffs[j][0]+k ) 
        # The first frame of a sequence starts with "s_", so it can be identified later
        opt_flow_list.append( os.path.join(frames_folder, subfolder, 'frame_%04d.png' % (diffs[j][0] + k)) + ' ' +
                              os.path.join(frames_folder, subfolder, 'frame_%04d.png' % (diffs[j][0] + k+1)) + ' ' +
                              os.path.join(flow_folder,   subfolder, 'flow/%s%04d_%04d.flo' % ('s_' if k == 0 else '', diffs[j][0]+k, diffs[j][0]+k+1)) ) 
        opt_flow_list.append( os.path.join(frames_folder, subfolder, 'frame_%04d.png' % (diffs[j][0] + k+1)) + ' ' +
                              os.path.join(frames_folder, subfolder, 'frame_%04d.png' % (diffs[j][0] + k)) + ' ' +
                              os.path.join(flow_folder,   subfolder, 'flow/%s%04d_%04d.flo' % ('s_' if k == 0 else '', diffs[j][0]+k+1, diffs[j][0]+k)) ) 
      keep_frames.append( diffs[j][0]+n_frames )
  # Delete all frames which are not used to construct the database
  for j in range(1,num_frames+1):
    if not j in keep_frames:
      os.remove( os.path.join(frames_folder, subfolder, "frame_%04d.png" % j) )

file = open(os.path.join(out_folder, "flowlist.txt"),"w")
for opt_flow_pair in opt_flow_list:
  file.write(opt_flow_pair + '\n')
file.close()
