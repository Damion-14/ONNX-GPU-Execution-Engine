#include "kernels.cuh"
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>

namespace onnx_runner {

// ============================================================================
// Mul (Element-wise Multiplication)
// ============================================================================

__global__ void mulKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void mulScalarKernel(const float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

void mulCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] * B[i];
    }
}

void launchMulKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        mulCPU(A, B, C, size, num_threads);
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        mulKernel<<<grid_size, block_size>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }
}

void launchMulScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] * scalar;
        }
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        mulScalarKernel<<<grid_size, block_size>>>(A, scalar, C, size);
        cudaDeviceSynchronize();
    }
}

// ============================================================================
// Div (Element-wise Division)
// ============================================================================

__global__ void divKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / B[idx];
    }
}

__global__ void divScalarKernel(const float* A, float scalar, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] / scalar;
    }
}

void divCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] / B[i];
    }
}

void launchDivKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        divCPU(A, B, C, size, num_threads);
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        divKernel<<<grid_size, block_size>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }
}

void launchDivScalarKernel(const float* A, float scalar, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] / scalar;
        }
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        divScalarKernel<<<grid_size, block_size>>>(A, scalar, C, size);
        cudaDeviceSynchronize();
    }
}

// ============================================================================
// Pow (Element-wise Power)
// ============================================================================

__global__ void powKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = powf(A[idx], B[idx]);
    }
}

__global__ void powScalarKernel(const float* A, float exponent, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = powf(A[idx], exponent);
    }
}

void powCPU(const float* A, const float* B, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::pow(A[i], B[i]);
    }
}

void launchPowKernel(const float* A, const float* B, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        powCPU(A, B, C, size, num_threads);
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        powKernel<<<grid_size, block_size>>>(A, B, C, size);
        cudaDeviceSynchronize();
    }
}

void launchPowScalarKernel(const float* A, float exponent, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < size; ++i) {
            C[i] = std::pow(A[i], exponent);
        }
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        powScalarKernel<<<grid_size, block_size>>>(A, exponent, C, size);
        cudaDeviceSynchronize();
    }
}

// ============================================================================
// Sqrt (Element-wise Square Root)
// ============================================================================

__global__ void sqrtKernel(const float* A, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = sqrtf(A[idx]);
    }
}

void sqrtCPU(const float* A, float* C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; ++i) {
        C[i] = std::sqrt(A[i]);
    }
}

void launchSqrtKernel(const float* A, float* C, int size, bool use_cpu, int num_threads) {
    if (use_cpu) {
        sqrtCPU(A, C, size, num_threads);
    } else {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        sqrtKernel<<<grid_size, block_size>>>(A, C, size);
        cudaDeviceSynchronize();
    }
}

} // namespace onnx_runner
