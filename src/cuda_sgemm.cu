#include <cuda_runtime.h>
#define BLOCK_SIZE 1
__global__ void kernel_sgemm(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2) {

	// each thread computes one element of M3
	// by accumulating results into M3_value
	float M3_value = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t k = 0; k < d1; k++) {
		M3_value += M1_device[row + k*d1] * M2_device[k + col*d2];
	}
	M3_device[row + col*d1] += M3_value;
}

extern "C" void kernel_sgemm_launch(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2) {
	
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((d2)/block.x, (d1)/block.y);
	kernel_sgemm<<<grid, block>>>(M1_device, M2_device, M3_device, d1, d2);
	cudaDeviceSynchronize();
}
