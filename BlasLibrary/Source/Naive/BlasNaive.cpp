#include "BlasNaive.hpp"

namespace BlasNaive {

void im2col(float* imgData, int channels, int height, int width, int kernelH, int kernelW,
            int padH, int padW, int strideH, int strideW, float* colData) {

    int colHeight = (height + 2 * padH - kernelH) / strideH + 1;
    int colWidth = (width + 2 * padW - kernelW) / strideW + 1;
    int colChannels = channels * kernelH * kernelW;
    for (int c = 0; c < colChannels; ++c) {

        int offsetW = c % kernelW;
        int offsetH = (c / kernelW) % kernelH;
        int imChannel = c / kernelH / kernelW;
        for (int h = 0; h < colHeight; ++h) {
          for (int w = 0; w < colWidth; ++w) {

            int hPad = h * strideH - padH + offsetH;
            int wPad = w * strideW - padW + offsetW;
            if (hPad >= 0 && hPad < height && wPad >= 0 && wPad < width)
              colData[(c * colHeight + h) * colWidth + w] =
                imgData[(imChannel * height + hPad) * width + wPad];
            else
              colData[(c * colHeight + h) * colWidth + w] = 0;
          }
        }
      }
}

}
