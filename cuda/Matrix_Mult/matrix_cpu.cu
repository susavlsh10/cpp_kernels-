#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <cmath>

#include "matrix_cpu.h"

void initializeMatrix(float *matrix, int rows, int cols, int INIT) {
    
    //srand(12345);
    if (INIT){
        // Initialize matrix with random values
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                //matrix[i * rows + j] =static_cast<float>(rand()) / RAND_MAX; //random floating point numbers
                matrix[i * rows + j] =rand() % 10; //random integers from 0 to 9
            }
        }
    }
    else{
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i * rows + j] = 0; 
            }
        }
    }
}

void printMatrix(float*matrix, int ROWS, int COLS){
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << matrix[i * ROWS + j] << " ";
        }
        std::cout << std::endl;
    }
}

float* alloc_matrix(int ROWS, int COLS, int INIT){
    // allocate memory for the matrix
    float* matrix = new float[ROWS*COLS];

    // initialize the matrix with random val ues
    initializeMatrix(matrix, ROWS, COLS, INIT);  

    return matrix;
}


void multiplyMatrices(float* A, float* B, float* C, int N) {

    // compute the multiplication of matrices A and B and store the result in C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                //C[i][j] += A[i][k] * B[k][j];
                C[i * N + j] += A[i * N + k]* B[k * N + j];
            }
        }
    }
}

void addMatrices(float *A, float* B, float* C, int m, int n){
    /* 
        A, B, C -> m x m matrices     
        n -> offset between rows of actual matrix
        if m == n -> full matrix addition
    */
    
    //compute matrices addition of A and B 
    for (int i = 0; i< m; i++){
        for(int j = 0; j< m; j++){
            C[i * m + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

void subMatrices(float *A, float* B, float* C, int m, int n){
    /* 
        A, B, C -> m x m matrices     
        n -> offset between rows of actual matrix
        if m == n -> full matrix addition
    */
    
    //compute matrices subtraction of A and B 
    for (int i = 0; i< m; i++){
        for(int j = 0; j< m; j++){
            C[i * m + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

bool Check_Matrices(float* X, float* Y, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (X[i * cols + j] != Y[i * cols + j])
            {
                std::cout << "Error : The matrices are not equal!" << std::endl;
                return false;
            }
        }
    }
    std::cout << "Success : The matrices are equal." << std::endl;
    return true;
}

void MatrixCopy(float *dst, const float *src, const int m_dst, const int n_dst, const int m_src, const int n_src) 
{
	int i, j;
	if (m_dst < m_src || n_dst < n_src)
		for (i = 0; i < m_dst; ++i)
			for (j = 0; j < n_dst; ++j)
				dst[i * n_dst + j] = src[i * n_src + j];
	else
		for (i = 0; i < m_src; ++i)
			for (j = 0; j < n_src; ++j)
				dst[i * n_dst + j] = src[i * n_src + j];
}