#include "utils.h"
// Bilinear sampling is done in BDHW (coalescing is not obvious in BDHW)
// we assume BDHW format in inputImages
// we assume B(YX)HW format on grids

struct stride_t {
  int batch;
  int channels;
  int height;
  int width;
};

__device__ void BilinearSamplerBDHW_getTopLeft(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   point = floor(x);
   weight = 1 - (x - point);
}

__device__ bool BilinearSamplerBDHW_between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__device__ int BilinearSamplerBDHW_toAddress(int b, int ch, int y, int x, stride_t stride) {
  return b*stride.batch + ch*stride.channels + y*stride.height + x*stride.width;
}

__device__ void BilinearSamplerBDHW_sumReduceShMem(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}



__global__ void BilinearSamplerBDHW_bilinearSamplingFromGrid(float* inputImages_data, stride_t inputImages_stride,
                                         float* grids_data, stride_t grids_stride,
                                         float* output_data, stride_t output_stride,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_height, int output_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)
      
   const int xOut = (blockIdx.y/output_height)*blockDim.y*blockDim.x + threadIdx.x*blockDim.y + threadIdx.y;

   const bool withinImageBounds = xOut < output_width;

   const int width = inputImages_width;
   const int height = inputImages_height;

   const int ch = blockIdx.x;
   const int yOut = blockIdx.y % output_height;
   const int b = blockIdx.z;

   float yf,xf;

   if(!withinImageBounds) return;
   yf = grids_data[BilinearSamplerBDHW_toAddress(b, 0, yOut, xOut, grids_stride)] + yOut;
   xf = grids_data[BilinearSamplerBDHW_toAddress(b, 1, yOut, xOut, grids_stride)] + xOut;
   
   int yInTopLeft, xInTopLeft;
   float yWeightTopLeft, xWeightTopLeft;
   BilinearSamplerBDHW_getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   BilinearSamplerBDHW_getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);
   
   const int outAddress = BilinearSamplerBDHW_toAddress(b, ch, yOut, xOut, output_stride);
   const int inTopLeftAddress = BilinearSamplerBDHW_toAddress(b, ch, yInTopLeft, xInTopLeft, inputImages_stride);
   const int inTopRightAddress = inTopLeftAddress + inputImages_stride.width;
   const int inBottomLeftAddress = inTopLeftAddress + inputImages_stride.height;
   const int inBottomRightAddress = inBottomLeftAddress + inputImages_stride.width;

   float v=0;
   float inTopLeft=0;
   float inTopRight=0;
   float inBottomLeft=0;
   float inBottomRight=0;

   bool topLeftIsIn = BilinearSamplerBDHW_between(xInTopLeft, 0, width-1) && BilinearSamplerBDHW_between(yInTopLeft, 0, height-1);
   bool topRightIsIn = BilinearSamplerBDHW_between(xInTopLeft+1, 0, width-1) && BilinearSamplerBDHW_between(yInTopLeft, 0, height-1);
   bool bottomLeftIsIn = BilinearSamplerBDHW_between(xInTopLeft, 0, width-1) && BilinearSamplerBDHW_between(yInTopLeft+1, 0, height-1);
   bool bottomRightIsIn = BilinearSamplerBDHW_between(xInTopLeft+1, 0, width-1) && BilinearSamplerBDHW_between(yInTopLeft+1, 0, height-1);

   // Interpolation
  if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress];
  if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress];
  if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress];
  if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress];

  v = xWeightTopLeft * yWeightTopLeft * inTopLeft
    + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
    + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
    + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;
  
  output_data[outAddress] = v;
}

static int cunn_BilinearSamplerBDHW_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");


   dim3 blocks(output->size[1], output->size[2] * ((output->size[3]+511)/512), output->size[0]);
   dim3 threads(32,16);

   /* assume BDHW */
   BilinearSamplerBDHW_bilinearSamplingFromGrid <<< blocks, threads, 0, THCState_getCurrentStream(state) >>> (THCudaTensor_data(state, inputImages), 
    (stride_t){ THCudaTensor_stride(state, inputImages, 0), 
      THCudaTensor_stride(state, inputImages, 1), 
      THCudaTensor_stride(state, inputImages, 2), 
      THCudaTensor_stride(state, inputImages, 3) },
      THCudaTensor_data(state, grids),  
    (stride_t){ THCudaTensor_stride(state, grids, 0), 
      THCudaTensor_stride(state, grids, 1),
      THCudaTensor_stride(state, grids, 2), 
      THCudaTensor_stride(state, grids, 3) },
    THCudaTensor_data(state, output),
    (stride_t){ THCudaTensor_stride(state, output, 0), 
      THCudaTensor_stride(state, output, 1),
      THCudaTensor_stride(state, output, 2), 
      THCudaTensor_stride(state, output, 3) },
    THCudaTensor_size(state, inputImages, 1),
    THCudaTensor_size(state, inputImages, 2), 
    THCudaTensor_size(state, inputImages, 3),
    THCudaTensor_size(state, output, 2),
    THCudaTensor_size(state, output, 3));


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}


template<bool onlyGrid> __global__ void BilinearSamplerBDHW_backwardBilinearSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int gradOutput_width)
{
  // Not implemented 
  printf("error in BilinearSampler.backwardBilinearSampling: Not implemented");
  THError("aborting"); 
}





static int cunn_BilinearSamplerBDHW_updateGradInput(lua_State *L)
{
  // Not implemented 
  printf("error in BilinearSampler.updateGradInput: Not implemented");
  THError("aborting");
}


static int cunn_BilinearSamplerBDHW_updateGradInputOnlyGrid(lua_State *L)
{
  // Not implemented 
  printf("error in BilinearSampler.updateGradInput: Not implemented");
  THError("aborting");
}



static const struct luaL_Reg cunn_BilinearSamplerBDHW__ [] = {
  {"BilinearSamplerBDHW_updateOutput", cunn_BilinearSamplerBDHW_updateOutput},
  {"BilinearSamplerBDHW_updateGradInput", cunn_BilinearSamplerBDHW_updateGradInput},
  {"BilinearSamplerBDHW_updateGradInputOnlyGrid", cunn_BilinearSamplerBDHW_updateGradInputOnlyGrid},
  {NULL, NULL}
};

static void cunn_BilinearSamplerBDHW_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_BilinearSamplerBDHW__, "nn");
  lua_pop(L,1);
}
