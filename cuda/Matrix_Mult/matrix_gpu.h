#ifndef __MATRIX_GPU_H__
#define __MATRIX_GPU_H__

#include <cuda_runtime.h>

__global__ void MatAdd(const  float* A, const  float* B, float* C, const  int dim, const int n);

__global__ void MatSub(const float* A, const float* B, float* C, const int dim, const int n);

__global__ void MatCopy(const float* A, float* B, const int dim, const int n);

__global__ void MatMul(const float* A, const float* B, float* C, int n);

void Strassens_GPU(const float *d_A, const float *d_B, float* d_C, const int n, const int threshold);

#endif