#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

/*
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;
*/

/*
void print_matrix(Matrix *A);
Matrix *new_matrix(int rows, int cols);
Matrix *rand_matrix(int rows, int cols);
void free_matrix(Matrix *A);
*/

void print_matrix(Matrix *A) {

    if (A == NULL) {
        printf("MATRIX IS NULL\n");
    } else {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < A->cols; j++) {
                printf("%f ", A->data[i][j]);
            }
            printf("\n");
        }
    }
}

Matrix *new_matrix(int rows, int cols) {

    Matrix *A = (Matrix*) calloc(1, sizeof(*A));
    if (A == NULL) {
        return NULL;
    }
    A->rows = rows;
    A->cols = cols;
    A->data = (double**) calloc(A->rows, sizeof(*A->data));

    for(int i = 0; i < A->rows; i++) {
        A->data[i] = (double*) calloc(A->cols, sizeof(**A->data));
        if (A->data[i] == NULL) {
            return NULL;
        }
    }

    return A;
}

Matrix *zero_matrix(int rows, int cols) {
    Matrix *A = new_matrix(rows, cols);
    if (A == NULL) {
        return NULL;
    } else {
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = 0.0;
            }
        }
        return A;
    }
}

// Square identity matrix
Matrix *eye_matrix(int rows) {
    Matrix *A = new_matrix(rows, rows);

    if (A == NULL) {
        return NULL;
    }

    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            if (i == j) {
                A->data[i][j] = 1.0;
            } else {
                A->data[i][j] = 0.0;
            }
        }
    }
    return A;
}

Matrix *rand_matrix(int rows, int cols) {

    Matrix *A = new_matrix(rows, cols);
    if (A == NULL) {
        return NULL;
    } else {
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = (double) rand() / ((double) RAND_MAX + 1);
            }
        }
        return A;
    }
}

void free_matrix(Matrix *A) {
    for(int i = 0; i < A->rows; i++) {
        free(A->data[i]);
    }
    free(A->data);
    free(A);
}

Matrix *axpy(Matrix *A, Matrix *B, double alpha) {
    if (A->rows != B->rows || A->cols != B->cols) {
        return NULL;
    } else {
        Matrix *sum = new_matrix(A->rows, A->cols);
        for(int i = 0; i < sum->rows; i++) {
            for(int j = 0; j < sum->cols; j++) {
                sum->data[i][j] = alpha * A->data[i][j] + B->data[i][j];
            }
        }
        return sum;
    }
}

// Dot product of two column vectors
double dot(Matrix *A, Matrix *B) {
    double out_val = 0.0;
    if (A == NULL || B == NULL) {
        return -1.0;
    } else if (A->rows != B->rows || A->cols != B->cols) {
        return -1.0;
    } else if (A->cols != 1 || B->cols != 1) { // Ensures column vectors
        return -1.0;
    } else {
        for (int i = 0; i < A->rows; i++) {
            out_val = out_val + A->data[i][0] * B->data[i][0];
        }
        return out_val;
    }
}

// Norm of column vector
double norm(Matrix *A) {
    if (A == NULL) {
        return -1.0;
    } else if (A->cols != 1){
        return -1.0;
    } else {
        return sqrt(dot(A, A));
    }
}

// "Dot Product" of two matrices
double mdot(Matrix *A, Matrix *B) {
    double out_val = 0.0;

    if (A == NULL || B == NULL) {
        return -1.0;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        return -1.0;
    } else {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < B->rows; j++) {
                out_val = out_val + A->data[i][j] * B->data[i][j];
            }
        }
    }

    return out_val;
}

// Matrix-vector product
// returns alpha * Ax + beta * y
//Matrix *gemv(int M, int N, int K, 
//        Matrix *A, 
//        Matrix *x, 
//        Matrix *y) {
//    return A;
//}

// Matrix-matrix product
// alpha * AB + beta * C
// C <-- AB
// A is (M, N)
// B is (N, K)
// C is (M, K)
Matrix *gemm(int M, int N, int K, 
        Matrix *A, 
        Matrix *B) {

    if (A->rows != M || A->cols != N || B->rows != N || B->rows != K) {
        return NULL;
    }

    Matrix *C = zero_matrix(M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            for (int k = 0; k < N; k++) {
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return C;
}
