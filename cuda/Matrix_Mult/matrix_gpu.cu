#include <cuda_runtime.h>
#include "matrix_gpu.h"

__global__ void MatAdd(const  float* A, const  float* B, float* C, const  int dim, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dim && j < dim){
        C[i * dim + j] = A[i * n + j] + B[i * n + j];
    }
}

__global__ void MatSub(const float* A, const float* B, float* C, const int dim, const int n)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dim && j < dim){
        C[i * dim + j] = A[i * n + j] - B[i * n + j]; 
    }
}

__global__ void MatCopy(const float* A, float* B, const int dim, const int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < dim && j < dim){
        B[i * dim + j] = A[i * n + j]; 
    }
}

__global__ void MatCombine(const  float* C11, const  float* C12, const float* C21, const float* C22, float* C, const int m, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < m){
				C[i * n + j] = C11[i * m + j];				
				C[i * n + j + m] = C12[i * m + j];			
				C[(i + m) * n + j] = C21[i * m + j];		
				C[(i + m) * n + j + m] = C22[i * m + j];
    }
}

__global__ void MatMul(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0;
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void Strassens_GPU(const float *d_A, const float *d_B, float* d_C, const int n, const int threshold){

    if (n == threshold){
        /* GPU kernel for Matrix Multiplication */
        dim3 threads_per_block(32 ,32);
        dim3 num_blocks((n + threads_per_block.x - 1) / threads_per_block.x, (n + threads_per_block.y - 1) / threads_per_block.y);
        MatMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    }
    else{
        /* Recursive Step */
        int m = n/2;
        
        dim3 threads_per_block(32 ,32);
        dim3 num_blocks((m + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);
        //allocate cuda memory for variables

        float *M1, *M2, *M3, *M4, *M5, *M6, *M7;
        float *a0, *b0;
        
        cudaMalloc((void **)&a0, m*m*sizeof(float));
        cudaMalloc((void **)&b0, m*m*sizeof(float));

        /* M1 = (A11 + A22)(B11 + B12)  */
        cudaMalloc((void **)&M1, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (&d_A[0], &d_A[m * (n+1)], a0, m, n);
        MatAdd<<<num_blocks, threads_per_block>>> (&d_B[0], &d_B[m * (n+1)], b0, m, n);
        Strassens_GPU(a0, b0, M1, m, threshold);

        /* M2 = (A21 + A22)B11 */
        cudaMalloc((void **)&M2, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (&d_A[m*n], &d_A[m * (n+1)], a0, m, n);
        MatCopy<<<num_blocks, threads_per_block>>> (&d_B[0], b0, m, n);
        Strassens_GPU(a0, b0, M2, m, threshold);

        /* M3 = A11(B12 - B22)*/
        cudaMalloc((void **)&M3, m*m*sizeof(float));
        MatCopy<<<num_blocks, threads_per_block>>> (&d_A[0], a0, m, n);
        MatSub<<<num_blocks, threads_per_block>>> (&d_B[m], &d_B[m * (n+1)], b0, m, n);
        Strassens_GPU(a0, b0, M3, m, threshold);

        /* M4 = A22(B21-B11) */
        cudaMalloc((void **)&M4, m*m*sizeof(float));
        MatCopy<<<num_blocks, threads_per_block>>> (&d_A[m*(n+1)], a0, m, n);
        MatSub<<<num_blocks, threads_per_block>>> (&d_B[m*n], &d_B[0], b0, m, n);
        Strassens_GPU(a0, b0, M4, m, threshold);

        /* M5 = (A11 + A12)B22 */
        cudaMalloc((void **)&M5, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (&d_A[0], &d_A[m], a0, m, n);
        MatCopy<<<num_blocks, threads_per_block>>> (&d_B[m*(n+1)], b0, m, n);
        Strassens_GPU(a0, b0, M5, m, threshold);

        /* M6 = (A21 - A11)(B11 + B12)*/
        cudaMalloc((void **)&M6, m*m*sizeof(float));
        MatSub<<<num_blocks, threads_per_block>>> (&d_A[m*n], &d_A[0], a0, m, n);
        MatAdd<<<num_blocks, threads_per_block>>> (&d_B[0], &d_B[m], b0, m, n);
        Strassens_GPU(a0, b0, M6, m, threshold);

        /* M7 = (A12 - A22)(B21 + B22)*/
        cudaMalloc((void **)&M7, m*m*sizeof(float));
        MatSub<<<num_blocks, threads_per_block>>> (&d_A[m], &d_A[m*(n+1)], a0, m, n);
        MatAdd<<<num_blocks, threads_per_block>>> (&d_B[m*n], &d_B[m*(n+1)], b0, m, n);
        Strassens_GPU(a0, b0, M7, m, threshold);

        /* Compute output Matrix C */
        
        float *C11, *C12, *C21, *C22;
        
        /* C11 = M1 + M4 - M5 + M7  */
        cudaMalloc((void **)&C11, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (M1, M4, a0, m, m);
        MatSub<<<num_blocks, threads_per_block>>> (M7, M5, b0, m, m);
        MatAdd<<<num_blocks, threads_per_block>>> (a0, b0, C11, m, m);

        /* C12 = M3 + M5 */
        cudaMalloc((void **)&C12, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (M3, M5, C12, m, m);

        /* C21 = M2 + M4 */
        cudaMalloc((void **)&C21, m*m*sizeof(float));
        MatAdd<<<num_blocks, threads_per_block>>> (M2, M4, C21, m, m);

        /* C22 = M1 - M2 + M3 + M6 */
        cudaMalloc((void **)&C22, m*m*sizeof(float));
        MatSub<<<num_blocks, threads_per_block>>> (M1, M2, a0, m, m);
        MatAdd<<<num_blocks, threads_per_block>>> (M3, M6, b0, m, m);
        MatAdd<<<num_blocks, threads_per_block>>> (a0, b0, C22, m, m);

        /* Combine matrices */
        MatCombine<<<num_blocks, threads_per_block>>>(C11, C12, C21, C22, d_C, m, n);
    }
}