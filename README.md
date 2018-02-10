# fast-artistic-videos

This is the source code for fast video style transfer described in

**[Artistic style transfer for videos and spherical images](https://arxiv.org/abs/1708.04538)**
<br>
Manuel Ruder,
Alexey Dosovitskiy,
[Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/)
<br>

The paper builds on
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
by Gatys et al. and [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/jcjohnson/fast-neural-style)
by Johnson et al. and our code is based on Johnson's implementation [fast-neural-style](https://github.com/jcjohnson/fast-neural-style).

It is a successor of our previous work [Artistic style transfer for videos](https://github.com/manuelruder/artistic-videos) and runs several orders of magnitudes faster.

If you find this code useful for your research, please cite

```
@inproceedings{Ruder2017,
  title={Artistic style transfer for videos and spherical images},
  author={Manuel Ruder, Alexey Dosovitskiy, Thomas Brox},
  booktitle={CoRR},
  year={2017}
}
```

## Setup
Code for inference in implemented in [Torch](http://torch.ch/).

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
```

### (Optional) GPU Acceleration

If you have an NVIDIA GPU, you can accelerate all operations with CUDA.

First [install CUDA](https://developer.nvidia.com/cuda-downloads), then
update / install the following packages:

```bash
luarocks install cutorch
luarocks install cunn
```

Also install `stnbhwd` (GPU accelerated warping) included in this repository:

```
cd stnbhwd
luarocks make stnbdhw-scm-1.rockspec
```

### (Optional) cuDNN

When using CUDA, you can use cuDNN to accelerate convolutions and save memory.

First [download cuDNN](https://developer.nvidia.com/cudnn) and copy the
libraries to `/usr/local/cuda/lib64/`. Then install the Torch bindings for cuDNN:

```bash
luarocks install cudnn
```

### Optical flow estimator

To benefit from faster execution times, a fast optical flow estimator is required.

There are sample scripts in our repository for either DeepFlow or FlowNet 2.0. DeepFlow is slower but comes as a standalone executable and is therefore very easy to install (just download the executable). Execution time will still be a lot faster than with our optimization-based approach. Faster execution times can be reached with FlowNet 2.0 which runs on a GPU as well, given you have a sufficient fast GPU. It was used for the experiments in the paper.

#### DeepFlow setup instructions

Just download both [DeepFlow](http://lear.inrialpes.fr/src/deepflow/) and [DeepMatching](http://lear.inrialpes.fr/src/deepmatching/) (CPU version) and place the static binaries (`deepmatching-static` and `deepflow2-static`) in the root directory of this repository.

#### FlowNet 2.0 setup instructions

Go to [flownet2 (GitHub)](https://github.com/lmb-freiburg/flownet2) and follow the instructions there on how to download, compile and use the source code and pretrained models. Since FlowNet is build upon Caffe, you may also want to read [Caffe | Installation](http://caffe.berkeleyvision.org/installation.html) for a list of dependencies (unfortunately, they are not listed in the flownet2 repository). There is also a Dockerfile for easy installation of the complete code in one step:
[flownet2-docker (GitHub)](https://github.com/lmb-freiburg/flownet2-docker)

Then, open `run-flownet-multiple.sh` and set the directory to the FlowNet files and models.

If you have struggles installing Caffe, there is also a TensorFlow implementation: [FlowNet2 (TensorFlow)](https://github.com/sampepose/flownet2-tf). However, you will have to adapt the scripts in this repository accordingly.

*Please don't ask me for support installing FlowNet 2.0. Ask the original authors or use DeepFlow.*

### Pretrained Models
Download all pretrained video style transfer models by running the script

```bash
bash models/download_models.sh
```

This will download 6 video model and 6 image model files (~300MB) to the folder `models/`.

You can download pretrained spherical video models with `download_models_vr.sh`, it will download 2 models (~340MB). These models are larger because we used more filters. We later found that less filters can archive similar performance, but didn't retrain the spherical video models.


## Running on new videos

You can use the scripts `stylizeVideo_*.sh <path_to_video> <path_to_video_model> [<path_to_image_model>]` to easily stylize videos using pretrained models. Choose one of the optical flow methods and specify one of the models we provide, see above. If image model is not specified, it will use the video model to generate the first frame (by marking everything as occluded). It will do all the preprocessing steps for you. For longer videos, make sure to have enough disk space available. This script will extract the video into uncompressed image files.

For advances users, videos can be stylized with `fast_artistic_videos.lua`.

You must specify the following options:
 - ```model_vid``` Path to a pretrained video model.
 - ```model_img``` Path to a separate pretrained image model which will be used to stylize the first frame, or 'self', if the video model should be used for this (the whole scene will be marked as uncertain in this case)
 - ```input_pattern``` File path pattern of the input frames, e.g. ```video/frame_%04d.png```
 - ```flow_pattern``` A file path pattern for files that store the backward flow between the frames. The placeholder in square brackets refers to the frame position where the optical flow starts and the placeholder in braces refers to the frame index where the optical flow points to. For example `flow_[%02d]_{%02d}.flo` means the flow files are named flow_02_01.flo, flow_03_02.flo, etc. If you use the script included in this repository (makeOptFlow.sh), the filename pattern will be `backward_[%d]_{%d}.flo`.
 - ```occlusions_pattern``` A file path pattern for the occlusion maps between two frames. These files should be a grey scale image where a white pixel indicates a high flow weight and a black pixel a low weight, respective. Same format as above. If you use the script, the filename pattern will be `reliable_[%d]_{%d}.pgm`.
 - ```output_prefix``` File path pattern of the output, e.g. `stylized/frame_%04d.png`
 
By default this script runs on CPU; to run on GPU, add the flag `-gpu`
specifying the GPU on which to run.

Other useful options:
 - ```occlusions_min_filter```: Width of a min filter applied to the occlusion map, can help to remove artifacts around occlusions. (Default: `7`)
 - ```median_filter```: Width of a median filter applied to the output. Can reduce network noise. (Default: `3`)
 - ```continue_with```: Continue with the given frame index, if the previous frames are already stylized and available at the output location.
 - ```num_frames```: Maximum number of frames to process. (Default: `9999`)
 - ```backward```: Process in backward direction, from the last frame to the first one.
 
To use this script for evaluation, specify `-evaluate` and give the following options:
 - ```flow_pattern_eval```: Ground truth optical flow.
 - ```occlusions_pattern_eval```: Ground truth occlusion maps.
 - ```backward_eval```: 'Perform evaluation in backward direction. Useful if only forward flow is available as it is the case for the Sintel dataset.
 - ```evaluation_file```: File to write the results in.
 - ```loss_network```: Path to a pretrained network used to compute style and content similarity, e.g. VGG-16.
 - ```content_layers```: Content layer indices to compute content similarity.
 - ```style_layers```: Style layer indices.
 - ```style_image```: Style image to be used for evaluation.
 - ```style_image_size```
 
## Running on spherical videos

To stylize spherical videos, frames must present as cube map projections with overlapping borders. Most commonly, however, spherical videos are encoded as equirectangular projection. Therefore, a reporjection becomes necessary.

### Reprojection software

[Transform360](https://github.com/facebook/transform360) can do the necessary transformations. To install, follow the instruction in their repository.

### Example script

Given a successful Transform360 compilation and a vr video in equirectangular projection (most common format), you can use the script `stylizeVRVideo_[deepflow|flownet].sh <path_to_equirectangular_projection_video> <path_to_pretrained_vr_model>`. Make sure to place the  `ffmpeg` binary dropped by Transform360 in the root directory of this repository. As above, also make sure to have enough disk space available for longer videos.

### Advanced usage

See the example scripts above for a preprocessing pipeline. Each cube face must be stored in a separate file.

`fast_artistic_videos_vr.lua` has similar options than the video script with the following differences:

 - The arguments given for ```input_pattern```, ```flow_pattern``` and ```occlusions_pattern``` must have another placeholder for the cube face id, for example `frame_%05d-%d.ppm` and `backward_[%d]_{%d}-%d.flo`.
 - ```overlap_pixel_w```: Horizontal overlapping region.
 - ```overlap_pixel_h```: Vertical overlapping region.
 - ```out_cubemap```: Whether the individual cube faces should be combined to one file.
 - ```out_equi```: Whether an additional equirectangular projection should be created from the output. Increases processing time. If this option is present, the size of the projection can be specified with `out_equi_w` and `out_equi_w`.
 - ```create_inconsistent_border```: No border consistency (for benchmarking purposes).

## Training new models

Not (yet) available.

## Contact

For issues or questions related to this implementation, please use the [issue tracker](https://github.com/manuelruder/fast-artistic-videos/issues). For everything else, including licensing issues, please email us. Our contact details can be found in [our paper](https://arxiv.org/abs/1708.04538).

## License

Free for personal or research use; for commercial use please contact us. Since our algorithm is based on Johnson's implementation, see also [fast-neural-style #License](https://github.com/jcjohnson/fast-neural-style#license).

