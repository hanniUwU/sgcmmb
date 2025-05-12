#ifndef CUDA_SGEMM_H
#define CUDA_SGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

void kernel_sgemm_launch(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2);

#ifdef __cplusplus
}
#endif

#endif // CUDA_SGEMM_H



