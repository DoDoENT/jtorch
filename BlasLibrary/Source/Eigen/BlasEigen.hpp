#pragma once

namespace BlasEigen {

void gemm(char transA, char transB, int m, int n, int k, float alpha, float* a, int lda,
            float* b, int ldb, float beta, float* c, int ldc);

}
