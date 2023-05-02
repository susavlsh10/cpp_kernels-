#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__


void initializeMatrix(float *matrix, int rows, int cols, int INIT);

void printMatrix(float*matrix, int ROWS, int COLS);

float* alloc_matrix(int ROWS, int COLS, int INIT);

void multiplyMatrices(float* A, float* B, float* C, int N);

void addMatrices(float *A, float* B, float* C, int m, int n);

void subMatrices(float *A, float* B, float* C, int m, int n);

bool Check_Matrices(float* X, float* Y, int rows, int cols);

void MatrixCopy(float *dst, const float *src, const int m_dst, const int n_dst, const int m_src, const int n_src);

#endif