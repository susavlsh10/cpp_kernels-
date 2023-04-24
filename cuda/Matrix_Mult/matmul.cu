#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <cmath>

#include <cuda_runtime.h>

#define DEBUG 1

using namespace std;

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
            cout << matrix[i * ROWS + j] << " ";
        }
        cout << endl;
    }
}

float* alloc_matrix(int ROWS, int COLS, int INIT){
    // allocate memory for the matrix
    float* matrix = new float[ROWS*COLS];

    // initialize the matrix with random values
    initializeMatrix(matrix, ROWS, COLS, INIT);  

    return matrix;
}


void multiplyMatrices(float* A, float* B, float* C, int nRowsA, int nColsA, int nColsB) {

    // compute the multiplication of matrices A and B and store the result in C
    for (int i = 0; i < nRowsA; i++) {
        for (int j = 0; j < nColsB; j++) {
            for (int k = 0; k < nColsA; k++) {
                //C[i][j] += A[i][k] * B[k][j];
                C[i * nRowsA + j] += A[i * nRowsA + k]* B[k * nRowsA + j];
            }
        }
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

    const int ROWS = pow(2,k);
    const int COLS = ROWS;

    float* A, *B, *C;
    srand(time(NULL)); // set seed for random number generator

    A = alloc_matrix(ROWS, COLS, 1);
    B = alloc_matrix(ROWS, COLS, 1);
    C = alloc_matrix(ROWS, COLS, 0);

    multiplyMatrices(A, B, C, ROWS,ROWS, ROWS);
    
    // print the matrix
    if (DEBUG){
        cout<< "\nMatrix A " << endl;
        printMatrix(A, ROWS, COLS);

        cout<< "\nMatrix B " << endl;
        printMatrix(B, ROWS, COLS);

        cout<< "\nMatrix C " << endl;
        printMatrix(C, ROWS, COLS); 
    }
   
    delete [] A;
    delete [] B;
    delete [] C;

    return 0;
}
