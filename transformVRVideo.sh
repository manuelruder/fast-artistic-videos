set -e

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ] && [ "$#" -ne 4 ] && [ "$#" -ne 6 ]; then
  echo "Usage: <input_video> <output_folder> [<start_in_mm:ss> [<length_in_mm:ss> [<cube_edge_length> <expand_coef>]]]"
  exit 1
fi

start=${3:-00:00}
length=${4:-0}
length_param=""
if [ $length -ne 0 ]; then
  length_param = "-t 00:$length"
fi
cube_edge_length=${5:-768}
expand_coef=${6:-1.2}

mkdir -p $2
./ffmpeg -ss 00:$start -i $1 $length_param -filter_complex "[0:v]transform360=input_stereo_format=MONO:cube_edge_length=$cube_edge_length:interpolation_alg=LANCZOS4:enable_low_pass_filter=1:enable_multi_threading=1:num_horizontal_segments=32:num_vertical_segments=15:adjust_kernel=1:expand_coef=$expand_coef,split=6[side1][side2][side3][side4][side5][side6];[side1]crop=(iw/3):(ih/2):0:0[out1];[side2]crop=(iw/3):(ih/2):(iw/3):0[out2];[side3]crop=(iw/3):(ih/2):(2*iw/3):0[out3];[side4]crop=(iw/3):(ih/2):0:(ih/2)[out4];[side5]crop=(iw/3):(ih/2):(iw/3):(ih/2)[out5];[side6]crop=(iw/3):(ih/2):(2*iw/3):(ih/2)[out6]" \
-map "[out1]" $2/frame_%05d-1.ppm \
-map "[out2]" $2/frame_%05d-2.ppm \
-map "[out3]" $2/frame_%05d-3.ppm \
-map "[out4]" $2/frame_%05d-4.ppm \
-map "[out5]" $2/frame_%05d-5.ppm \
-map "[out6]" $2/frame_%05d-6.ppm \

# Expand coeff / video size -> Overlapping width (which has to be passed to fast_artistic_video_vr.lua)
#1.1667 / 768  -> Overlap 110
#1.1498 / 768  -> Overlap 100
#1.2    / 768  -> Overlap 128
#1.2503 / 1024 -> Overlap 205
#1.1083 / 1024 -> Overlap 100
