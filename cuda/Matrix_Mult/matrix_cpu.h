#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__


void initializeMatrix(float *matrix, int rows, int cols, int INIT);

void printMatrix(float*matrix, int ROWS, int COLS);

float* alloc_matrix(int ROWS, int COLS, int INIT);

void multiplyMatrices(const float* A, const float* B, float* C, const int N);


void addMatrices(const float *A, const float* B, float* C, const int m, const int n);

void subMatrices(const float *A, const float* B, float* C, const int m, const int n);

bool Check_Matrices(float* X, float* Y, int rows, int cols);

void MatrixCopy(float *dst, const float *src, const int m_dst, const int n_dst, const int m_src, const int n_src);

void strassen_cpu(const float* A, const float* B, float* C, const int n, const int threshold);

#endif