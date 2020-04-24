#include "Blas.hpp"
#include "Naive/BlasNaive.hpp"

#include "Eigen/BlasEigen.hpp"


namespace Blas {

void gemm(char transA, char transB, int m, int n, int k, float alpha, float* a, int lda,
            float* b, int ldb, float beta, float* c, int ldc) {

    BlasEigen::gemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

}

void im2col(float* imgData, int channels, int height, int width, int kernelH, int kernelW,
              int padH, int padW, int strideH, int strideW, float* colData) {

    BlasNaive::im2col(imgData, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW, colData);
}

}
