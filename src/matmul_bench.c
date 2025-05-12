#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutensor.h>
#include <assert.h>
#include <sys/stat.h>
#include <math.h>

#include "cuda_sgemm.h"
#include "cutlass_sgemm.h"

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUTENSOR_CHECK(x) do { \
    cutensorStatus_t err = (x); \
    if (err != CUTENSOR_STATUS_SUCCESS) { \
        fprintf(stderr, "cuTENSOR error at %s:%d: %s\n", __FILE__, __LINE__, cutensorGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>

    static double get_time(void) {
        LARGE_INTEGER freq, ctr;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&ctr);
        return (double)ctr.QuadPart / (double)freq.QuadPart;
    }
#else
    #include <time.h>

    static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
#endif

//static const size_t matrix_sizes[] = {1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14};
//static const size_t matrix_sizes[] = {1<<10, 1<<11, 1<<12, 1<<13, 1<<14};
static const size_t matrix_sizes[] = {1<<2};
//static const size_t matrix_sizes[] = {1<<12};
//static const size_t matrix_sizes[] = {54, 150, 1200, 5596};
#define BLOCKSIZE 64
#define N_MEASURED 10

void random_matrix(float* M_host, size_t d1, size_t d2) {

    for (size_t j = 0; j < d2; j++) {
    for (size_t i = 0; i < d1; i++) {
        M_host[i + j*d1] = 2.0f * ((float) rand() / RAND_MAX - 0.5f);
    }}
}

void printM_f(float* M_host, char* desc, size_t d1, size_t d2) {

    printf("\t --- printing %zux%zu Matrix (float) ---\n", d1, d2);
    for (size_t j = 0; j < d2; j++) {
    	for (size_t i = 0; i < d1; i++) {
	    printf("%8.2f ", M_host[i + j*d1]);
    	}
	printf("\n");
    }
}

void matmul_naive(float* M1_host, float* M2_host, float* M3_host, size_t d1, size_t d2, size_t d3) {

    for (size_t j = 0; j < d3; j++) {
    for (size_t k = 0; k < d2; k++) {
    for (size_t i = 0; i < d1; i++) {
        M3_host[i + j*d1] += M1_host[i + k*d1] * M2_host[k + j*d2];
    }}}
}

void matmul_cached(float* M1_host, float* M2_host, float* M3_host, int d1, int d2, int d3) {

    for (size_t k1 = 0; k1 < d2; k1 += BLOCKSIZE) {
    for (size_t j1 = 0; j1 < d3; j1 += BLOCKSIZE) {
        size_t k2 = k1 + BLOCKSIZE, j2 = j1 + BLOCKSIZE;
        for (size_t j = j1; j < j2; j++) {
        for (size_t k = k1; k < k2; k++) {
        for (size_t i = 0;  i < d1; i++) {
            M3_host[i + j*d1] += M1_host[i + k*d1] * M2_host[k + j*d2];
        }}}
    }}
}

extern void sgemm_(const char*, const char*, const unsigned*, const unsigned*,
                   const unsigned*, const float*, const float*, const unsigned*,
                   const float*, const unsigned*, const float*, float*, const unsigned*);

void matmul_BLAS(float* M1_host, float* M2_host, float* M3_host, unsigned d1, unsigned d2, unsigned d3) {
    const float alpha = 1.0f, beta = 1.0f;
    sgemm_("N", "N", &d1, &d3, &d2, &alpha, M1_host, &d1, M2_host, &d2, &beta, M3_host, &d1);
}

void matmul_CUDA(float* M1_host, float* M2_host, float* M3_host, size_t d1, size_t d2, size_t d3, double time_ms[]) {

    // cuda: device memory allocation
    float* M1_device;
    float* M2_device;
    float* M3_device;
    size_t szA = d1 * d2 * sizeof(float);
    size_t szB = d2 * d3 * sizeof(float);
    size_t szC = d1 * d3 * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &M1_device, szA));
    CUDA_CHECK(cudaMalloc((void**) &M2_device, szB));
    CUDA_CHECK(cudaMalloc((void**) &M3_device, szC));
    CUDA_CHECK(cudaMemcpy(M1_device, M1_host, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M2_device, M2_host, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(M3_device, 0, szC));

    // measured runs
    for (size_t i = 0; i < N_MEASURED; i++) {
    	cudaEventRecord(start, 0);
	launch_sgemm_kernel(M1_device, M2_device, M3_device, d1, d2);
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop); // important since gpu is async
    	float ms = 0;
    	cudaEventElapsedTime(&ms, start, stop);
    	time_ms[i] = ms;
    }

}

void matmul_cuBLAS(float* M1_host, float* M2_host, float* M3_host, size_t d1, size_t d2, size_t d3, double time_ms[]) {
    float* M1_device;
    float* M2_device;
    float* M3_device;
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaMalloc((void**) &M1_device, (size_t) (d1 * d2 * sizeof *M1_device)));
    CUDA_CHECK(cudaMalloc((void**) &M2_device, (size_t) (d2 * d3 * sizeof *M2_device)));
    CUDA_CHECK(cudaMalloc((void**) &M3_device, (size_t) (d1 * d3 * sizeof *M3_device)));
    CUDA_CHECK(cudaMemcpy(M1_device, M1_host , (size_t) (d1 * d2 * sizeof *M1_device), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M2_device, M2_host , (size_t) (d2 * d3 * sizeof *M2_device), cudaMemcpyHostToDevice));

    cublasCreate(&handle);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // measured runs
    for (size_t i = 0; i < N_MEASURED; i++) {
    	cudaEventRecord(start, 0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d1, d3, d2, &((const float) {1.0f}),
                    M1_device, d1,
                    M2_device, d2,                                &((const float) {1.0f}),
                    M3_device, d1);
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop); // important since gpu is async
    	float ms = 0;
    	cudaEventElapsedTime(&ms, start, stop);
    	time_ms[i] = ms;
    }

    // copy back to host to verify
    cudaMemcpy(M3_host, M3_device, (size_t) (d1 * d3 * sizeof *M3_device), cudaMemcpyDeviceToHost);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(M1_device);
    cudaFree(M2_device);
    cudaFree(M3_device);
}

void matmul_cuTENSOR(float* M1_host, const float* M2_host, float* M3_host, size_t d1, size_t d2, size_t d3, double* time_ms)
{
    // cuda: device memory allocation
    float* M1_device;
    float* M2_device;
    float* M3_device;
    size_t szA = d1 * d2 * sizeof(float);
    size_t szB = d2 * d3 * sizeof(float);
    size_t szC = d1 * d3 * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &M1_device, szA));
    CUDA_CHECK(cudaMalloc((void**) &M2_device, szB));
    CUDA_CHECK(cudaMalloc((void**) &M3_device, szC));
    CUDA_CHECK(cudaMemcpy(M1_device, M1_host, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M2_device, M2_host, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(M3_device, 0, szC));

    // cutensor: handle creation
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorCreate(&handle));

    // cutensor: tensor descriptors initialization
    cutensorTensorDescriptor_t descA, descB, descC, descD;
    uint32_t rank = 2;
    uint32_t alignment = 256;
    const int64_t extA[2] = { d1, d2 };
    const int64_t strA[2] = {  1, d1 };
    const int64_t extB[2] = { d2, d3 };
    const int64_t strB[2] = {  1, d2 };
    const int64_t extC[2] = { d1, d3 };
    const int64_t strC[2] = {  1, d1 };

    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle, &descA, rank, extA, strA, CUTENSOR_R_32F, alignment));
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle, &descB, rank, extB, strB, CUTENSOR_R_32F, alignment));
    CUTENSOR_CHECK(cutensorCreateTensorDescriptor(handle, &descC, rank, extC, strC, CUTENSOR_R_32F, alignment));

    int32_t modeA[2] = { 0, 1 };
    int32_t modeB[2] = { 1, 2 };
    int32_t modeC[2] = { 0, 2 };

    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
    cutensorOperationDescriptor_t contrDesc;
    CUTENSOR_CHECK(cutensorCreateContraction(handle, &contrDesc,
		    descA, modeA, CUTENSOR_OP_IDENTITY,
		    descB, modeB, CUTENSOR_OP_IDENTITY,
    		    descC, modeC, CUTENSOR_OP_IDENTITY,
	            descC, modeC, descCompute));

    cutensorPlanPreference_t planPref;
    CUTENSOR_CHECK(cutensorCreatePlanPreference(handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSize = 0;
    CUTENSOR_CHECK(cutensorEstimateWorkspaceSize(handle, contrDesc, planPref, CUTENSOR_WORKSPACE_MIN, &workspaceSize));

    // plan creation
    cutensorPlan_t plan;
    CUTENSOR_CHECK(cutensorCreatePlan(handle, &plan, contrDesc, planPref, workspaceSize));

    void* workspace = NULL;
    if (workspaceSize) CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    cudaStream_t stream = 0;

    float alpha = 1.0f;
    float beta = 1.0f;

    // measured runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (size_t i = 0; i < N_MEASURED; i++) {
        cudaEventRecord(start, 0);
	CUTENSOR_CHECK(cutensorContract(handle, plan, &alpha, M1_device, M2_device, &beta, M3_device, M3_device, workspace, workspaceSize, stream));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        time_ms[i] = ms;
    }

    // copy result back
    cudaMemcpy(M3_host, M3_device, szC, cudaMemcpyDeviceToHost);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(M1_device);
    cudaFree(M2_device);
    cudaFree(M3_device);
    if (workspace) cudaFree(workspace);

   CUTENSOR_CHECK(cutensorDestroy(handle));
}

void matmul_CUTLASS(float* M1_host, const float* M2_host, float* M3_host, size_t d1, size_t d2, size_t d3, double* time_ms)
{
    // cuda: device memory allocation
    float* M1_device;
    float* M2_device;
    float* M3_device;
    size_t szA = d1 * d2 * sizeof(float);
    size_t szB = d2 * d3 * sizeof(float);
    size_t szC = d1 * d3 * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**) &M1_device, szA));
    CUDA_CHECK(cudaMalloc((void**) &M2_device, szB));
    CUDA_CHECK(cudaMalloc((void**) &M3_device, szC));
    CUDA_CHECK(cudaMemcpy(M1_device, M1_host, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(M2_device, M2_host, szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(M3_device, 0, szC));

    float alpha = 1.0f;
    float beta = 1.0f;
    // measured runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (size_t i = 0; i < N_MEASURED; i++) {
        cudaEventRecord(start, 0);
        cutlass_sgemm_nn(d1, d3, d2, alpha, M1_device, d1, M2_device, d2, beta, M3_device, d1);
        cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	time_ms[i] = ms;
    }

    // copy result back
    cudaMemcpy(M3_host, M3_device, szC, cudaMemcpyDeviceToHost);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(M1_device);
    cudaFree(M2_device);
    cudaFree(M3_device);
}

void stats_calculate(double* time_ms, size_t d1, size_t d2, size_t d3, char* mode) {

    char folder_path[256];
    char file_path[512];
    snprintf(folder_path, sizeof(folder_path), "results/float");
    mkdir(folder_path, 0777);
    snprintf(file_path, sizeof(file_path), "%s/%s_A%zux%zu_B%zux%zu_C%zux%zu.dat", folder_path, mode, d1, d2, d2, d3, d1, d3);
    FILE* data = fopen(file_path, "w");

    double sum = 0;
    double sum2 = 0;
    // trim, allowing the caches to warm up
    size_t trim = 5;

    for (size_t i = trim; i < N_MEASURED; ++i) {
	sum  += time_ms[i];
	sum2 += time_ms[i] * time_ms[i];
    }
    double mean_ms  = sum / (N_MEASURED-trim);
    double var_ms  = (sum2 - sum*sum/(N_MEASURED-trim)) / (N_MEASURED-trim-1);
    double std_ms  = sqrt(var_ms);
    double gflops = (double) (2.0 * d1 * d2 * d3) / (mean_ms * 1e6);
    printf("avg = %lf ms ± %lf ms (%zu runs)\n", mean_ms, std_ms, N_MEASURED-trim);
    printf("throughput = %lf GFlops\n", gflops);

    for (size_t i = trim; i < N_MEASURED; i++) {
        fprintf(data, "%e\t%e\n", time_ms[i], (double) (2.0 * d1 * d2 * d3) / (time_ms[i] * 1e6));
    }
    fclose(data);
}

int main(void) {
    srand(42);

    for (size_t i = 0; i < ARRAY_LENGTH(matrix_sizes); i++) {
    for (size_t j = i; j < ARRAY_LENGTH(matrix_sizes); j++) {

        size_t d1 = matrix_sizes[i];
        //size_t d2 = matrix_sizes[j];
	size_t d2 = 6000;
        size_t d3 = d1;

        float* M1_host = calloc((size_t) d1 * d2, (size_t) sizeof *M1_host);
        float* M2_host = calloc((size_t) d2 * d3, (size_t) sizeof *M2_host);
        float* M3_host = calloc((size_t) d1 * d3, (size_t) sizeof *M3_host);

        printf("\nMatrix A: %zux%zu, B: %zux%zu, C: %zux%zu\n", d1, d2, d2, d3, d1, d3);
        size_t bytes_allocated = (d1*d2 + d2*d3 + d1*d3) * sizeof *M1_host;
        printf("Memory used: %.2fMiB = %.2fGiB\n", bytes_allocated/ (1024.0 * 1024.0), bytes_allocated / (1024.0*1024.0*1024.0));

        random_matrix(M1_host, d1, d2);
        random_matrix(M2_host, d2, d3);

	double cpu_ms[N_MEASURED] = {};
        double t0, t1;

	/*
        // NAIVE
	printf("NAIVE:\n");
        memset(M3_host, 0, d1 * (size_t) d3 * sizeof *M3_host);
	// measured runs
        for (int i = 0; i < N_MEASURED; ++i) {
            t0 = get_time();
            matmul_BLAS(M1_host, M2_host, M3_host, d1, d2, d3);
            t1 = get_time();
            cpu_ms[i] = (t1 - t0) * 1e3;
        }
	stats_calculate(cpu_ms, d1, d2, d3, "NAIVE");
	*/
        // BLAS
	printf("BLAS:\n");
        memset(M3_host, 0, d1 * (size_t) d3 * sizeof *M3_host);
        for (int i = 0; i < N_MEASURED; ++i) {
	    cpu_ms[i] = 0;
	}

	// measured runs
        for (int i = 0; i < N_MEASURED; ++i) {
            t0 = get_time();
            matmul_BLAS(M1_host, M2_host, M3_host, d1, d2, d3);
            t1 = get_time();
            cpu_ms[i] = (t1 - t0) * 1e3;
        }
	stats_calculate(cpu_ms, d1, d2, d3, "BLAS");

	// cuBLAS
	printf("cuBLAS:\n");
        memset(M3_host, 0, d1 * (size_t) d3 * sizeof *M3_host);
        double gpu_ms[N_MEASURED];
       	matmul_cuBLAS(M1_host, M2_host, M3_host, d1, d2, d3, gpu_ms);
	stats_calculate(gpu_ms, d1, d2, d3, "cuBLAS");

        // cuTENSOR
	printf("cuTENSOR:\n");
        memset(M3_host, 0, d1 * (size_t) d3 * sizeof *M3_host);
	for (size_t i = 0; i < N_MEASURED; i++) {
	    gpu_ms[i] = 0;
	}
        matmul_cuTENSOR(M1_host, M2_host, M3_host, d1, d2, d3, gpu_ms);
	stats_calculate(gpu_ms, d1, d2, d3, "cuTENSOR");

	// CUTLASS
	printf("CUTLASS:\n");
        memset(M3_host, 0, d1 * (size_t) d3 * sizeof *M3_host);
	for (size_t i = 0; i < N_MEASURED; i++) {
	    gpu_ms[i] = 0;
	}
        matmul_CUTLASS(M1_host, M2_host, M3_host, d1, d2, d3, gpu_ms);
	stats_calculate(gpu_ms, d1, d2, d3, "CUTLASS");

        free(M1_host);
        free(M2_host);
        free(M3_host);
    }}

    return 0;
}

