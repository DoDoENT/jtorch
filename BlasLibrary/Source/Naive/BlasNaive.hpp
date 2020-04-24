#pragma once

namespace BlasNaive {

void im2col(float* imgData, int channels, int height, int width, int kernelH, int kernelW,
              int padH, int padW, int strideH, int strideW, float* colData);

}
