#pragma once

namespace Blas {

void gemm(char transA, char transB, int m, int n, int k, float alpha, float* a, int lda,
            float* b, int ldb, float beta, float* c, int ldc);

void im2col(float* imgData, int channels, int height, int width, int kernelH, int kernelW,
              int padH, int padW, int strideH, int strideW, float* colData);


}
