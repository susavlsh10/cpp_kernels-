#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <cmath>

#include <cuda_runtime.h>
#include "matrix_cpu.h"

#define DEBUG 0

using namespace std;




__global__ void MatAdd(float* A, float* B, float* C, int dim){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < dim && j < dim){
        C[i * dim + j] = A[i * dim + j] + B[i * dim + j];
    }
}

__global__ void MatMul(float* A, float* B, float* C, int n)
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

void strassen_cpu(float* A, float* B, float* C, int n, int threshold) {
    /*
    A - > Matrix A
    B - > Matrix B
    C - > Result Matrix
    n - > current size of matrix for multiplication (n x n)
    threshold - > threshold for naive multiplication
    N - > Offset for rows in original matrix (Original size : N x N) 
    
    */
    
    if (n == threshold) {
        multiplyMatrices(A, B, C, n);
    }

    else {
        int m = n/2; 

        // allocate memory for variables
        float * M1 = new float[m*m];
        float * M2 = new float[m*m];
        float * M3 = new float[m*m];
        float * M4 = new float[m*m];
        float * M5 = new float[m*m];
        float * M6 = new float[m*m];
        float * M7 = new float[m*m];

        float *a0 = new float[m*m];              
        float *b0 = new float[m*m];                 

        /* M1 = (A11 + A22)(B11 + B22) */
        addMatrices(&A[0], &A[m * (n+1)], a0, m,  n);   // (A11 + A22)
        addMatrices(&B[0], &B[m * (n+1)], b0, m,  n);   // (B11 + B22)
        strassen_cpu(a0, b0, M1, m, threshold);     // (A11 + A22)(B11 + B22)

        /* M2 = (A21 + A22)B11 */
        addMatrices(&A[m*n], &A[m * (n+1)], a0, m,  n);   // (A21 + A22)
        MatrixCopy(b0, &B[0], m, m, n, n);
        strassen_cpu(a0, b0, M2, m, threshold);     // (A11 + A22)(B11 + B22)

        /* M3 = A11(B12 - B22)*/
        MatrixCopy(a0, &A[0], m, m, n, n);
        subMatrices(&B[m], &B[m*(n+1)], b0, m, n);
        strassen_cpu(a0, b0, M3, m, threshold); 

        /* M4 = A22(B21-B11)*/
        MatrixCopy(a0, &A[m* (n+1)], m, m, n, n);
        subMatrices(&B[m*n], &B[0], b0, m, n);
        strassen_cpu(a0, b0, M4, m, threshold); 

        /* M5 = (A11 + A12)B22 */
        addMatrices(&A[0], &A[m], a0, m,  n);  
        MatrixCopy(b0, &B[m* (n+1)], m, m, n, n);
        strassen_cpu(a0, b0, M5, m, threshold);

        /* M6 = (A21 - A11)(B11 + B12)*/
        subMatrices(&A[m*n], &A[0], a0, m, n);
        addMatrices(&B[0], &B[m], b0, m,  n);
        strassen_cpu(a0, b0, M6, m, threshold);

        /* M7 = (A12 - A22)(B21 + B22)*/
        subMatrices(&A[m], &A[m*(n+1)], a0, m, n);
        addMatrices(&B[m*n], &B[m*(n+1)], b0, m,  n);
        strassen_cpu(a0, b0, M7, m, threshold);
    
        /* Compute output Matrix C */

        /* C11 = M1 + M4 - M5 - M7  */
        float *C11 = new float[m*m];
        subMatrices(M7, M5, a0, m, m);
        addMatrices(M1, M4, b0, m, m);
        addMatrices(a0, b0, C11, m, m);
    
        /* C12 = M3 + M5 */
        float *C12 = new float[m*m];
        addMatrices(M3, M5, C12, m, m);

        /* C21 = M2 + M4 */
        float *C21 = new float[m*m];
        addMatrices(M2, M4, C21, m, m);

        /* C22 = M1 - M2 + M3 + M6 */
        float *C22 = new float[m*m];
        subMatrices(M1, M2, a0, m, m);
        addMatrices(M3, M6, b0, m, m);
        addMatrices(a0, b0, C22, m, m);

        for (int i = 0; i < m; ++i){
			for (int j = 0; j < m; ++j) {
				C[i * n + j] = C11[i * m + j];				
				C[i * n + j + m] = C12[i * m + j];			
				C[(i + m) * n + j] = C21[i * m + j];		
				C[(i + m) * n + j + m] = C22[i * m + j];	
			}
        }
        delete [] M1; delete [] M2; delete [] M3; delete [] M4; delete [] M5; delete [] M6; delete [] M7;
        delete [] a0; delete [] b0;
        delete [] C11; delete [] C12; delete [] C21; delete [] C22;            


    }
}






int main(int argc, char* argv[]) {
    // check if the correct number of command line arguments are provided
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <k> <k'>" << endl;
        return 1;
    }

    // convert command line arguments to integers
    const int k = atoi(argv[1]);
    const int k_i = atoi(argv[2]);

    const int N = pow(2,k);
    const int threshold = pow(2, k_i);
    //const int COLS = ROWS;

    //int size = ROWS * COLS;

    float* A, *B, *C, *C1;
    srand(time(NULL)); // set seed for random number generator

    A = alloc_matrix(N, N, 1); //CPU
    B = alloc_matrix(N, N, 1); //CPU
    C = alloc_matrix(N, N, 0); //CPU
    C1 = alloc_matrix(N, N, 0); //CPU

    // Allocate memory in GPU
    float * d_A, *d_B, *d_C;
    
    /*
    cudaMalloc((void **)&d_A, N*N*sizeof(float));
    cudaMalloc((void **)&d_B, N*N*sizeof(float));
    cudaMalloc((void **)&d_C, N*N*sizeof(float));

    //Copy the matrices to GPU memory
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    
    */

    
    

    //CPU operations
    multiplyMatrices(A, B, C1, N);
    
    strassen_cpu(A, B, C, N, threshold);


    //addMatrices(A, B, C1, N, N);
    
    /*
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y);

    */
 
    //launch the kernel
    //MatAdd<<<num_blocks, threads_per_block>>> (d_A, d_B, d_C, N);
    //MatMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    
    // Copy the result back to the host
    //cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Make sure the kernel is executed correctly
    bool result = Check_Matrices(C, C1, N, N);

    // print the matrix
    if (DEBUG){
        /*
        cout<< "\nMatrix A " << endl;
        printMatrix(A, N, N);

        cout<< "\nMatrix B " << endl;
        printMatrix(B, N, N);
    
        */


        cout<< "\nMatrix C " << endl;
        printMatrix(C, N, N); 

        cout<< "\nMatrix C1 " << endl;
        printMatrix(C1, N, N); 
    }
   
    // Free the memory on the device
    /*
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    */


    // Free the memory on the host
    delete [] A;
    delete [] B;
    delete [] C;

    return 0;
}
