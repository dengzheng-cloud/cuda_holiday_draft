#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <chrono>

#define ROW_NUM 16
#define COL_NUM 256

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

template<typename T>
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

int main() {
    std::vector<float> h_in = init_random_vector(ROW_NUM * COL_NUM);
    std::vector<float> h_out(ROW_NUM, 0.f);

    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(float) * ROW_NUM * COL_NUM);
    cudaMalloc(&d_out, sizeof(float) * ROW_NUM);

    cudaMemcpy(d_in, h_in.data(), sizeof(float) * h_in.size(), cudaMemcpyHostToDevice);
    reduction0<<<ROW_NUM, COL_NUM, COL_NUM * sizeof(float)>>>(d_in, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, sizeof(float) * h_out.size(), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_out.size(); i++) {
        std::cout << h_out[i] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
