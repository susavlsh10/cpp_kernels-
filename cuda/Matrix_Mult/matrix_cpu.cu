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


void multiplyMatrices(const float* A, const float* B, float* C, const int N) {

    // compute the multiplication of matrices A and B and store the result in C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
             C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k]* B[k * N + j];
            }
        }
    }
}

void addMatrices(const float *A, const float* B, float* C, const int m, const int n){
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

void subMatrices(const float *A, const float* B, float* C, const int m, const int n){
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
                //std::cout << "Error : The matrices are not equal!" << std::endl;
                return false;
            }
        }
    }
    //std::cout << "Success : The matrices are equal." << std::endl;
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

void strassen_cpu(const float* A, const float* B, float* C, const int n, const int threshold) {
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

        /* C11 = M1 + M4 - M5 + M7  */
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

        delete [] M1; delete [] M2; delete [] M3; delete [] M4; delete [] M5; delete [] M6; delete [] M7;
        
        for (int i = 0; i < m; ++i){
			for (int j = 0; j < m; ++j) {
				C[i * n + j] = C11[i * m + j];				
				C[i * n + j + m] = C12[i * m + j];			
				C[(i + m) * n + j] = C21[i * m + j];		
				C[(i + m) * n + j + m] = C22[i * m + j];	
			}
        }
        
        delete [] a0; delete [] b0;
        delete [] C11; delete [] C12; delete [] C21; delete [] C22;            

    }
}