#include "BlasEigen.hpp"
#include "BlasHeader.hpp"  // for eigen_gemm

namespace BlasEigen {

void gemm(char transA, char transB, int m, int n, int k, float alpha, float* a, int lda,
            float* b, int ldb, float beta, float* c, int ldc) {

    eigen_gemm(&transA, &transB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

}
