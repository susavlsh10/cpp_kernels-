
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string.h>

#define DATA_SIZE 8192

cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size);

__global__ void addKernel(float *c, float *a, float *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}

float gen()
{
    static float i = 0;
    return ++i;
}

int main()
{
    
    int vec_size[] = {1024, 2048, 4096, 8192, 16384, 32768, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456};
	


    //float* cpu_result = new float[arraySize];
    char filename[100];
    char filename1[100];

    //for (int i=0; i<15; i++)
    //{
        int arraySize = vec_size[8];
        sprintf(filename, "/home/susavlsh10/kernels/cpp_kernels/float_save/float_%d.dat", arraySize);
        //sprintf(filename1, "/home/susavlsh10/kernels/cpp_kernels/float_save/float_%d_a.dat", arraySize);
        
        float* a = new float[arraySize];
        float* b = new float[arraySize];
        float* c = new float[arraySize];      

        std::ifstream fin;
        std::ifstream fin1;
        //auto start = std::chrono::high_resolution_clock::now();
        
        fin.open(filename, std::ios::in | std::ios::binary);
		
        if (!fin){
            std::cout<<"File cannot be opened 0."<<std::endl;
            return 1;
        }
		
        


        
    // cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
		
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    int blocksize, gridsize;
	int size = arraySize;
    blocksize = 1024;
    gridsize = (int)ceil(arraySize / blocksize);

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
       // goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       // goto Error;
    }

    
    
    // Copy input vectors from SSD to host memory then to GPU buffers.
	auto start = std::chrono::high_resolution_clock::now();
	
	fin.read((char*)a,sizeof(float)*arraySize);
    b = a;
	
    if(!fin.good()) {
        std::cout << "Error occurred at reading time!" << std::endl;
        return 1;
    }
  
	
	auto stop1 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start);    
	std::cout << arraySize <<": SSD -> CPU read  time: , " <<duration1.count() << ", microseconds" << std::endl;	

    auto start_x = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }
    auto stop_x = std::chrono::high_resolution_clock::now();
	auto duration_x = std::chrono::duration_cast<std::chrono::microseconds>(stop_x - start_x);    
	std::cout << arraySize <<": CPU -> GPU memcopy: , " <<duration_x.count() << ", microseconds" << std::endl;
    
    /*
    
    */
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       // goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //auto start = std::chrono::high_resolution_clock::now();
    addKernel<<<gridsize, blocksize>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
       // goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
       // goto Error;
    }

    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "GPU execution Time:" << duration.count() << " microseconds" << std::endl;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
       // goto Error;
    }
		
		
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);    
        std::cout << arraySize <<": CPU read -> CUDA (compute) -> CPU time: , " <<duration.count() << ", microseconds" << std::endl;
        
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

    //}
	
/*
    std::cout << "Vector C \n";
    for (int j = 0; j < arraySize; j++) {
        std::cout << c[j] << ",";
    }
    std::cout << std::endl;
*/
    //check accuracy
    
    /*
    for (int i = 0; i < arraySize; i++) {
        cpu_result[i] = a[i] * b[i];
    }

    bool match = true;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (cpu_result[i] != c[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << cpu_result[i]
                << " Device result = " << c[i] << std::endl;
            match = false;
            break;
        }
    }
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;   
    */
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    std::cout << "Vector multiply complete" << std::endl;

    return 0;
}

/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size)
{

 

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/