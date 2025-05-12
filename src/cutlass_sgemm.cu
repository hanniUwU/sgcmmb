#include "cutlass_sgemm.h"

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm.h>

extern "C" cudaError_t cutlass_sgemm_nn(
    int M, int N, int K,
    float alpha,
    const float *A, int lda,
    const float *B, int ldb,
    float beta,
    float *C, int ldc)
{
    // Use CUTLASS column-major layout for A, B, C
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, ColumnMajor,
        float, ColumnMajor,
        float, ColumnMajor
    >;
    CutlassGemm gemm_op;

    // Build the CUTLASS GEMM arguments (M,N,K dimensions, A, B, C pointers, scalars)
    typename CutlassGemm::Arguments args(
        { M, N, K },
        { A, lda },
        { B, ldb },
        { C, ldc },   // source C (used if beta≠0)
        { C, ldc },   // destination D (we overwrite C with output)
        { alpha, beta }
    );

    // Launch CUTLASS kernel (runs on GPU)
    cutlass::Status status = gemm_op(args);
    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}
