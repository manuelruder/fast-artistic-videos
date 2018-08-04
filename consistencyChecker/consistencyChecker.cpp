// consistencyChecker
// Check consistency of forward flow via backward flow.
//
// (c) Manuel Ruder, Alexey Dosovitskiy, Thomas Brox 2016

#include <algorithm>
#include <assert.h>
#include "CTensor.h"
#include "CFilter.h"

// Which certainty value motion boundaries should get. Value between 0 (uncertain) and 255 (certain).
#define MOTION_BOUNDARIE_VALUE 255


// readMiddlebury
bool readMiddlebury(const char* filename, CTensor<float>& flow) {
  FILE *stream = fopen(filename, "rb");
  if (stream == 0) {
    std::cout << "Could not open " << filename << std::endl;
    return false;
  }
  float help;
  int dummy;
  dummy = fread(&help,sizeof(float),1,stream);
  int xSize,ySize;
  dummy = fread(&xSize,sizeof(int),1,stream);
  dummy = fread(&ySize,sizeof(int),1,stream);
  flow.setSize(xSize,ySize,2);
  for (int y = 0; y < flow.ySize(); y++)
    for (int x = 0; x < flow.xSize(); x++) {
      dummy = fread(&flow(x,y,0),sizeof(float),1,stream);
      dummy = fread(&flow(x,y,1),sizeof(float),1,stream);
    }
  fclose(stream);
  return true;
}

// computeCorners --------------------------------------------------------------
void computeCorners(const CTensor<float>& image, CMatrix<float>* corners, float rho) {
  corners->setSize(image.xSize(),image.ySize());
  int xSize = image.xSize();
  int ySize = image.ySize();
  int aSize = xSize*ySize;
  // Compute gradient
  CTensor<float> dx(xSize,ySize,image.zSize());
  CTensor<float> dy(xSize,ySize,image.zSize());
  CDerivative<float> aDerivative(3);
  NFilter::filter(image,dx,aDerivative,1,1);
  NFilter::filter(image,dy,1,aDerivative,1);
  // Compute second moment matrix
  CMatrix<float> dxx(xSize,ySize,0);
  CMatrix<float> dyy(xSize,ySize,0);
  CMatrix<float> dxy(xSize,ySize,0);
  int i2 = 0;
  for (int k = 0; k < image.zSize(); k++)
    for (int i = 0; i < aSize; i++,i2++) {
      dxx.data()[i] += dx.data()[i2]*dx.data()[i2];
      dyy.data()[i] += dy.data()[i2]*dy.data()[i2];
      dxy.data()[i] += dx.data()[i2]*dy.data()[i2];
    }
  // Smooth second moment matrix
  NFilter::recursiveSmoothX(dxx,rho);
  NFilter::recursiveSmoothY(dxx,rho);
  NFilter::recursiveSmoothX(dyy,rho);
  NFilter::recursiveSmoothY(dyy,rho);
  NFilter::recursiveSmoothX(dxy,rho);
  NFilter::recursiveSmoothY(dxy,rho);
  // Compute smallest eigenvalue
  for (int i = 0; i < aSize; i++) {
    float a = dxx.data()[i];
    float b = dxy.data()[i];
    float c = dyy.data()[i];
    float temp = 0.5*(a+c);
    float temp2 = temp*temp+b*b-a*c;
    if (temp2 < 0.0f) corners->data()[i] = 0.0f;
    else corners->data()[i] = temp-sqrt(temp2);
  }
}

void checkConsistency(const CTensor<float>& flow1, const CTensor<float>& flow2, bool useStrcuture, const CMatrix<float>& structure, CMatrix<float>& reliable) {
  int xSize = flow1.xSize(), ySize = flow1.ySize();
  int size = xSize * ySize;
  CTensor<float> flow_dx(xSize,ySize,2);
  CTensor<float> flow_dy(xSize,ySize,2);
  CDerivative<float> derivative(3);
  NFilter::filter(flow1,flow_dx,derivative,1,1);
  NFilter::filter(flow1,flow_dy,1,derivative,1);
  CMatrix<float> motionEdge(xSize,ySize,0);
  for (int i = 0; i < size; i++) {
    motionEdge.data()[i] += flow_dx.data()[i]*flow_dx.data()[i];
    motionEdge.data()[i] += flow_dx.data()[size+i]*flow_dx.data()[size+i];
    motionEdge.data()[i] += flow_dy.data()[i]*flow_dy.data()[i];
    motionEdge.data()[i] += flow_dy.data()[size+i]*flow_dy.data()[size+i];
  }

  float structureAvg = 0;
  if (useStrcuture)
    structureAvg = structure.avg();
 
  for (int ay = 0; ay < flow1.ySize(); ay++)
    for (int ax = 0; ax < flow1.xSize(); ax++) {
      float bx = ax+flow1(ax, ay, 0);
      float by = ay+flow1(ax, ay, 1);
      int x1 = floor(bx);
      int y1 = floor(by);
      int x2 = x1 + 1;
      int y2 = y1 + 1;
      if (x1 < 0 || x2 >= xSize || y1 < 0 || y2 >= ySize)
      { reliable(ax, ay) = 0.0f; continue; }
      float alphaX = bx-x1; float alphaY = by-y1;
      float a = (1.0-alphaX) * flow2(x1, y1, 0) + alphaX * flow2(x2, y1, 0);
      float b = (1.0-alphaX) * flow2(x1, y2, 0) + alphaX * flow2(x2, y2, 0);
      float u = (1.0-alphaY)*a+alphaY*b;
      a = (1.0-alphaX) * flow2(x1, y1, 1) + alphaX * flow2(x2, y1, 1);
      b = (1.0-alphaX) * flow2(x1, y2, 1) + alphaX * flow2(x2, y2, 1);
      float v = (1.0-alphaY)*a+alphaY*b;
      float cx = bx+u;
      float cy = by+v;
      float u2 = flow1(ax,ay,0);
      float v2 = flow1(ax,ay,1);
      // Avoid false detections in homogeneous regions
      float structureTerm = 0;
      if (useStrcuture)
        structureTerm = 4.0f/structureAvg * std::max(0.0f, structureAvg/2.0f - structure(ax, ay));
      if (((cx-ax) * (cx-ax) + (cy-ay) * (cy-ay)) >= 0.01*(u2*u2 + v2*v2 + u*u + v*v) + structureTerm + 0.5f) {
        reliable(ax, ay) = 0.0f;
        continue;
      }
      if (motionEdge(ax, ay) > 0.01 * (u2*u2+v2*v2) + 0.002f) {
        reliable(ax, ay) = MOTION_BOUNDARIE_VALUE;
        continue;
      }
    }
}

int main(int argc, char** args) {
  assert(argc >= 4);

  //printf(args[1]);

  CTensor<float> flow1,flow2;
  readMiddlebury(args[1], flow1);
  readMiddlebury(args[2], flow2);
  assert(flow1.xSize() == flow2.xSize());
  assert(flow1.ySize() == flow2.ySize());


  int xSize = flow1.xSize(), ySize = flow1.ySize();

  // Check consistency of forward flow via backward flow and exclude motion boundaries
  CMatrix<float> reliable(xSize, ySize, 255.0f);
  reliable.writeToPGM(args[3]);
  
  if (argc >= 5) {
    CMatrix<float> structure;
    CTensor<float> image;
    image.readFromPPM(args[4]);
    computeCorners(image, &structure, 3.0f);
    structure.normalize(0.0f, 1.0f);
    checkConsistency(flow1, flow2, true, structure, reliable);
  } else {
    CMatrix<float> structure;
    checkConsistency(flow1, flow2, false, structure, reliable);
  }

  printf(args[3]);

  
  reliable.clip(0.0f, 255.0f);
 
  reliable.writeToPGM(args[3]);
}