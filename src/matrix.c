#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

int print_matrix(Matrix *A) {
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < A->cols; j++) {
                printf("%f ", A->data[i][j]);
            }
            printf("\n");
        }
        return EXIT_SUCCESS;
    }
}

// Initializes Matrix A
/*
int new_matrix(Matrix *A, int rows, int cols) {
    A = (Matrix*) calloc(1, sizeof(*A));
    if (A == NULL) {
        return EXIT_FAILURE;
    }
    A->rows = rows;
    A->cols = cols;
    A->data = (double**) calloc(A->rows, sizeof(*A->data));

    for(int i = 0; i < A->rows; i++) {
        A->data[i] = (double*) calloc(A->cols, sizeof(**A->data));
        if (A->data[i] == NULL) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
*/

int new_matrix(Matrix **A, int rows, int cols) {
    
    *A = (Matrix*) calloc(1, sizeof(Matrix *));
    if (*A == NULL) {
        return EXIT_FAILURE;
    }
    (*A)->rows = rows;
    (*A)->cols = cols;
    //*A->data = (double**) calloc(*A->rows, sizeof(double *));
    (*A)->data = (double**) calloc(rows, sizeof(double **));

    for(int i = 0; i < rows; i++) {
        (*A)->data[i] = (double *) calloc(cols, sizeof(double *));
        //A->data[i] = (double*) calloc(A->cols, sizeof(**A->data));
        if ((*A)->data[i] == NULL) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

// Sets A to be zero matrix
int zero_matrix(Matrix *A) {
    // int status = new_matrix(A, rows, cols);
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = 0.0;
            }
        }
        return EXIT_SUCCESS;
    }
}

// Sets A to be identity
int eye_matrix(Matrix *A) {
    //int status = new_matrix(A, rows, rows);

    if (A == NULL) {
        return EXIT_FAILURE;
    }
    if (A->rows != A->cols) {
        return EXIT_FAILURE;
    }

    int status = zero_matrix(A);
    if (status == EXIT_FAILURE) {
        return EXIT_FAILURE;
    } else {
        for(int i = 0; i < A->rows; i++) {
            A->data[i][i] = 1.0;
        }
        return EXIT_SUCCESS;
    }
}

// Sets A to have random values
int rand_matrix(Matrix *A) {

    //int status = new_matrix(A, rows, cols);
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = (double) rand() / ((double) RAND_MAX + 1);
            }
        }
        return EXIT_SUCCESS;
    }
}

void free_matrix(Matrix *A) {
    for(int i = 0; i < A->rows; i++) {
        free(A->data[i]);
    }
    free(A->data);
    free(A);
}

// C <-- alpha * A + B
int axpy(Matrix *A, Matrix *B, Matrix *C, double alpha) {
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } else if ((A->rows != B->rows) 
            || (A->cols != B->cols)
            || (A->cols != C->cols)
            || (A->rows != C->rows)) {
        return EXIT_FAILURE;
    } else {
        //Matrix *sum = new_matrix(A->rows, A->cols);
        for(int i = 0; i < C->rows; i++) {
            for(int j = 0; j < C->cols; j++) {
                C->data[i][j] = alpha * A->data[i][j] + B->data[i][j];
            }
        }
        return EXIT_SUCCESS;
    }
}

// Dot product of two column vectors
int dot(Matrix *A, Matrix *B, double *d) {
    if (A == NULL || B == NULL || d == NULL) {
        return EXIT_FAILURE;
    } else if (A->rows != B->rows || A->cols != B->cols) {
        return EXIT_FAILURE;
    } else if (A->cols != 1 || B->cols != 1) { // Ensures column vectors
        return EXIT_FAILURE;
    } else {
        *d = 0.0;
        for (int i = 0; i < A->rows; i++) {
            *d += A->data[i][0] * B->data[i][0];
        }
        return EXIT_SUCCESS;
    }
}

// Norm of column vector
int norm(Matrix *A, double *d) {
    if (A == NULL || d == NULL) {
        return EXIT_FAILURE;
    } else if (A->cols != 1){
        return EXIT_FAILURE;
    } else {
        int status = dot(A, A, d);
        *d = sqrt(*d);
        return status;
    }
}

// "Dot Product" of two matrices
int mdot(Matrix *A, Matrix *B, double *d) {
    if (A == NULL || B == NULL || d == NULL) {
        return EXIT_FAILURE;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        return EXIT_FAILURE;
    } else {
        *d = 0.0;
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < A->cols; j++) {
                *d += (A->data[i][j]) * (B->data[i][j]);
            }
        }
        return EXIT_SUCCESS;
    }
}

// Matrix-vector product
// y <- alpha * Ax + beta * y
int gemv(Matrix *A, Matrix *x, Matrix *y, double alpha, double beta) {
    return EXIT_FAILURE;
}

// Matrix-matrix product
// C <-- alpha * AB + beta * C
// A is (M, N)
// B is (N, K)
// C is (M, K)
int gemm(int M, int N, int K, 
        Matrix *A, 
        Matrix *B,
        Matrix *C) {

    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } 
    if (A->rows != M || A->cols != N 
            || B->rows != N || B->rows != K 
            || C->rows != M || C->cols != K) {
        return EXIT_FAILURE;
    }

    int status = zero_matrix(A);
    if (status == EXIT_FAILURE) {
        return EXIT_FAILURE;
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                for (int k = 0; k < N; k++) {
                    C->data[i][j] += A->data[i][k] * B->data[k][j];
                }
            }
        }
        return EXIT_SUCCESS;
    }
}
