#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <cmath>

#include <cuda_runtime.h>
#include "matrix_cpu.h"
#include "matrix_gpu.h"

#include <chrono>

#define DEBUG 0

using namespace std;


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
    const int K = pow(2, k_i);

    const int threshold = N/K;

    float* A, *B, *C, *C1, *C2, *C3;
    srand(time(NULL)); // set seed for random number generator

    A = alloc_matrix(N, N, 1); //CPU
    B = alloc_matrix(N, N, 1); //CPU
    C = alloc_matrix(N, N, 0); //CPU
    C1 = alloc_matrix(N, N, 0); //CPU
    C2 = alloc_matrix(N, N, 0); //CPU
    C3 = alloc_matrix(N, N, 0); //CPU

    // Allocate memory in GPU
    float * d_A, *d_B, *d_C;
    
    cudaMalloc((void **)&d_A, N*N*sizeof(float));
    cudaMalloc((void **)&d_B, N*N*sizeof(float));
    cudaMalloc((void **)&d_C, N*N*sizeof(float));

    //dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y);
        
    //Copy the matrices to GPU memory

    dim3 threads_per_block(32 ,32);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y);
    
    
    /* CPU based implementation  */
    auto start0 = std::chrono::high_resolution_clock::now();
        multiplyMatrices(A, B, C2, N);
    auto stop0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms0 = stop0 - start0;
    std::cout << "Traditional CPU time: " << elapsed_ms0.count() << " ms" << std::endl;
    

    auto start1 = std::chrono::high_resolution_clock::now();
        strassen_cpu(A, B, C3, N, threshold);
    auto stop1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms1 = stop1 - start1;
    std::cout << "Strassen's CPU time: " << elapsed_ms1.count() << " ms" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    MatMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    std::cout << "Traditional GPU time: " << elapsed_ms.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    Strassens_GPU(d_A, d_B, d_C, N, threshold);

    cudaMemcpy(C1, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = end - start;
    std::cout << "Strassen's GPU time: " << elapsed_ms.count() << " ms" << std::endl;

    // Make sure the kernel is executed correctly
    std::cout<<endl;
    bool result = Check_Matrices(C2, C3, N, N);
    bool result1 = Check_Matrices(C, C1, N, N);
    
    if (result== false || result1== false){
        std::cout<<" Error!! Matrices do not match. " << std::endl;
    }
    else{
        std::cout<<"Success!"<<std::endl;
    }
    
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


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    // Free the memory on the host
    delete [] A;
    delete [] B;
    delete [] C;
    delete [] C1;
    delete [] C2;
    delete [] C3;

    return 0;
}

/*
Archive
    //CPU operations
    
    //MatAdd<<<num_blocks, threads_per_block>>> (d_A, d_B, d_C, N);
    
    auto start = std::chrono::high_resolution_clock::now();
    //multiplyMatrices(A, B, C1, N);

    
    //Copy the matrices to GPU memory
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    //launch the kernel
    MatMul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    
    // Copy the result back to the host
    int m = N/2;
    float *M1;
    cudaMalloc((void **)&M1, m*m*sizeof(float));
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    
    



    
;




*/