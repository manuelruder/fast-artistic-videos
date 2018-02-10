set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

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
echo "Starting vr video extraction..."
nice bash transformVRVideo.sh $1 ${filename}

echo ""
echo "Starting optical flow computation..."
# This launches optical flow computation
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-1.ppm ./${filename}/flow_768-1 1
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-2.ppm ./${filename}/flow_768-2 1
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-3.ppm ./${filename}/flow_768-3 1
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-4.ppm ./${filename}/flow_768-4 1
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-5.ppm ./${filename}/flow_768-5 1
bash makeOptFlow_flownet.sh ./${filename}/frame_%05d-6.ppm ./${filename}/flow_768-6 1

echo "Starting occlusion estimator as a background task..."
# Hack: Run makeOptFlow again, this will skip flow computation since flow files are already present
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-1.ppm ./${filename}/flow_768-1 1 &
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-2.ppm ./${filename}/flow_768-2 1 &
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-3.ppm ./${filename}/flow_768-3 1 &
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-4.ppm ./${filename}/flow_768-4 1 &
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-5.ppm ./${filename}/flow_768-5 1 &
nice bash makeOptFlow_deepflow.sh ./${filename}/frame_%05d-6.ppm ./${filename}/flow_768-6 1 &

echo "Starting vr video stylization..."
# Perform style transfer
th fast_artistic_video_vr.lua \
-input_pattern ${filename}/frame_%05d-%d.ppm \
-flow_pattern ${filename}/flow_768-%d/backward_[%d]_{%d}.flo \
-occlusions_pattern ${filename}/flow_768-%d/reliable_[%d]_{%d}.pgm \
-output_prefix ${filename}/out \
-backend $backend \
-use_cudnn $use_cudnn \
-gpu $gpu \
-model_vid $model_vid_path \
-model_img $model_img_path \
-overlap_pixel_h 128 \
-overlap_pixel_w 128 \
-out_equi \
-out_equi_w 2560 \
-out_equi_h 1440 \
-fill_occlusions uniform-random

# Create video from output images.
./ffmpeg -i ${filename}/out-%05d_equi.png ${filename}-stylized.$extension
