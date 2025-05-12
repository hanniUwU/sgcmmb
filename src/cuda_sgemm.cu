#define BLOCK_SIZE 16

extern "C" {
	void kernel_sgemm_launch(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2);
}

__global__ void kernel_sgemm(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2) {

	// each thread computes one element of M3
	// by accumulating results into M3_value
	float M3_value = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t k = 0; k < d1; k++) {
		M3_value += M1[row + k*d1] * M2[k + col*d2];
	}
	M3_device[row + col*d1] = M3_value;
}

void kernel_sgemm_launch(float* M1_device, float* M2_device, float* M3_device, size_t d1, size_t d2) {
	dim3 block(16, 16);
	dim3 grid((d1+15)/16, (d2+15)/16);
	sgemm_kernel<<<grid, block>>>(M1_device, M2_device, M3_device, d1, d2);
	cudaDeviceSynchronize();
}
