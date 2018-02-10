set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}


# Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ] || [ "$#" -ge 4 ]; then
   echo "Usage: ./stylizeVideo.sh <path_to_video> <path_to_pretrained_model> [<path_to_firstframe_model>]"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
model_vid_path=$2
model_img_path=${3:-self}

# Create output folder
mkdir -p $filename


echo ""
read -p "In case of multiple GPUs, enter the zero-indexed ID of the GPU to use here, or enter -1 for CPU mode (slow!).\
 [0] $cr > " gpu
gpu=${gpu:-0}

if [ $gpu -ge 0 ]; then
  echo ""
  read -p "Which backend do you want to use? \
  For Nvidia GPUs it is recommended to use cudnn if installed. If not, use nn. \
  For non-Nvidia GPU, use opencl (not tested). Note: You have to have the given backend installed in order to use it. [cudnn] $cr > " backend
  backend=${backend:-cudnn}

  if [ "$backend" == "cudnn" ]; then
    backend="cuda"
    use_cudnn=1
  elif [ "$backend" == "nn" ]; then
    backend="cuda"
    use_cudnn=0
  elif [ "$backend" == "opencl" ]; then
    use_cudnn=0 
  else
    echo "Unknown backend."
    exit 1
  fi
else
  backend="nn"
  use_cudnn=0
fi

echo ""
read -p "Please enter a resolution at which the video should be processed, \
in the format w:h, or leave blank to use the original resolution. \
If you run out of memory, reduce the resolution. $cr > " resolution

echo ""
read -p "Please enter a downsampling factor (on a log scale, integer) for the matching algorithm used by DeepFlow. \
If you run out of main memory or optical flow estimation is too slow, slightly increase this value, \
otherwise the default value will be fine. [2] $cr > " opt_res

# Save frames of the video as individual image files
if [ -z $resolution ]; then
  $FFMPEG -i $1 ${filename}/frame_%05d.ppm
  resolution=default
else
  $FFMPEG -i $1 -vf scale=$resolution ${filename}/frame_%05d.ppm
fi

echo ""
echo "Starting optical flow computation as a background task..."
# This launches optical flow computation
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d.ppm ./${filename}/flow_$resolution 1 ${opt_res:-2} &

echo "Starting video stylization..."
# Perform style transfer
th fast_artistic_video.lua \
-input_pattern ${filename}/frame_%05d.ppm \
-flow_pattern ${filename}/flow_${resolution}/backward_[%d]_{%d}.flo \
-occlusions_pattern ${filename}/flow_${resolution}/reliable_[%d]_{%d}.pgm \
-output_prefix ${filename}/out \
-backend $backend \
-use_cudnn $use_cudnn \
-gpu $gpu \
-model_vid $model_vid_path \
-model_img $model_img_path


# Create video from output images.
$FFMPEG -i ${filename}/out-%05d.png ${filename}-stylized.$extension
