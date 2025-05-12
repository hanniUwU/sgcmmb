#ifndef CUTLASS_SGEMM_H
#define CUTLASS_SGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

cudaError_t cutlass_sgemm_nn(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif // CUTLASS_GEMM_H

