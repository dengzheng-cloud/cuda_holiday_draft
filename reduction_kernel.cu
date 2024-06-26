#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

#define ROW_NUM 16
#define COL_NUM 512

template <typename T>
__device__ void warp_reduction(volatile T* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}


std::vector<float> init_random_vector(int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    std::vector<float> random_vector(length);

    for (int i = 0; i < length; i++) {
        random_vector[i] = dis(gen);
    }
    return random_vector;
}

std::vector<float> cpu_verification(const std::vector<float>& in, int row, int col) {
    assert(row * col == in.size());
    std::vector<float> ret;
    for (int i = 0; i < row; i++) {
        float temp = 0.f;
        for (int j = 0; j < col; j ++) {
            temp += in[i * col + j];
        }
        ret.push_back(temp);
    }
    return ret;
}

template <typename T>
__global__ void reduction0(T* __restrict__ in, T* __restrict__ out) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = in[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// split one line one block here into one line two block
template <typename T>
__global__ void reduction1(T* __restrict__ in, T* __shared__ out) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + tid;
    sdata[tid] = in[i] + in[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}


// split one line one block here into one line two block
template <typename T>
__global__ void reduction2(T* __restrict__ in, T* __restrict__ out) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x * 2 + tid;
    sdata[tid] = in[i] + in[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) { warp_reduction<T>(sdata, tid); }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}


template <typename T, int block_size>
__device__ void warp_reduction1(volatile T* sdata, int tid) {
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid +  8];
    if (block_size >=  8) sdata[tid] += sdata[tid +  4];
    if (block_size >=  4) sdata[tid] += sdata[tid +  2];
    if (block_size >=  2) sdata[tid] += sdata[tid +  1];
}



template <typename T, int block_size>
__global__ void reduction3(T* __restrict__ in, T* __restrict__ out, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int offset = blockIdx.x * COL_NUM;
    unsigned int i   = tid;
    
    while (i < n) {
        sdata[tid] += in[offset + i] + in[offset + i + block_size]; 
        i += block_size * 2;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) { warp_reduction1<T, block_size>(sdata, tid); }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}



int main() {
    std::vector<float> h_in = init_random_vector(ROW_NUM * COL_NUM);
    std::vector<float> h_out(ROW_NUM, 0.f);

    const int block_size = 64;

    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float) * ROW_NUM * COL_NUM);
    cudaMalloc(&d_out, sizeof(float) * ROW_NUM);

    cudaMemcpy(d_in, h_in.data(), sizeof(float) * h_in.size(), cudaMemcpyHostToDevice);
    // reduction0<<<ROW_NUM, COL_NUM, COL_NUM * sizeof(float)>>>(d_in, d_out);
    // reduction1<<<ROW_NUM, COL_NUM / 2, COL_NUM / 2 * sizeof(float)>>>(d_in, d_out);
    // reduction2<<<ROW_NUM, COL_NUM / 2, COL_NUM / 2 * sizeof(float)>>>(d_in, d_out);
    reduction3<float, block_size><<<ROW_NUM, block_size, block_size * sizeof(float)>>>(d_in, d_out, COL_NUM);
    auto cpu_result = cpu_verification(h_in, ROW_NUM, COL_NUM);


    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, sizeof(float) * h_out.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_out.size(); i++) {
        std::cout << "cuda : " <<h_out[i] << ", cpu : " << cpu_result[i] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
