#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "matrix.h"

//TODO Add better error handling: program should actually die if things mess up
//TODO Transition to like, not even using the matrix struct
//TODO Fix resource leaks
//
//

// Parses a matrix text file into A column_ordered matrix
int readtxt(char *fname, double *A, int A_rows, int A_cols) {
    FILE *pf;
    pf = fopen(fname, "r");
    if (pf == NULL) {
        printf("could not open file %s: %s\n", fname, strerror(errno));
        return EXIT_FAILURE;
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            fscanf(pf, "%lf", &A[i + j*A_rows]);
        }
    }
}


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

int print_ptr(double *A, int A_rows, int A_cols) {
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        // Data stored in column-major order
        //
        for (int i = 0; i < A_rows; i++) {
            for (int j = 0; j < A_cols; j++) {
                if (A[j*A_rows + i] >= 0.0) {
                    printf(" %f ", A[j*A_rows + i]);
                } else {
                    printf("%f ", A[j*A_rows + i]);
                }
            }
            printf("\n");
        }
        return EXIT_SUCCESS;
    }
}

// Initializes new matrix struct
int init_matrix(Matrix **A, int rows, int cols) {
    // TODO make sure this is always zero
    if (A == NULL) {
        return EXIT_FAILURE;
    }

    *A = (Matrix*) calloc(1, sizeof(Matrix *));
    if (*A == NULL) {
        return EXIT_FAILURE;
    }
    (*A)->rows = rows;
    (*A)->cols = cols;
    (*A)->data = (double**) calloc(rows, sizeof(double **));

    for(int i = 0; i < rows; i++) {
        (*A)->data[i] = (double *) calloc(cols, sizeof(double *));
        if ((*A)->data[i] == NULL) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

// Sets A to be zero matrix
int zero_matrix(Matrix *A) {
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
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        //srand(time(0));
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = (double) rand() / ((double) RAND_MAX + 1);
            }
        }
        return EXIT_SUCCESS;
    }
}

//stolen from https://stackoverflow.com/questions/33058848/generate-a-random-double-between-1-and-1/33059025
double randfrom(double min, double max) {

    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}
// Sets A to have random values
int rand_ptr(double *A, int size) {
    if (A == NULL) {
        return EXIT_FAILURE;
    } else {
        srand(time(0));
        for (int i = 0; i < size; i++) {
            //A[i] = (double) rand() / ((double) RAND_MAX + 1);
            A[i] = randfrom(-1.0, 1.0);
        }
        return EXIT_SUCCESS;
    }
}

int copy_matrix(Matrix *A, Matrix *A_copy) {
    //TODO: I don't think this one does anything, should
    //probably be deprecated
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    if (A->rows != A_copy->rows || A->cols != A_copy->cols) {
        return EXIT_FAILURE;
    }
    A_copy->rows = A->rows;
    A_copy->cols = A->cols;
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A_copy->data[i][j] = A->data[i][j];
        }
    }
    return EXIT_SUCCESS;
}

int copy_matrix_deep(Matrix **A, Matrix **A_copy) {
    //TODO: Is this a resource leak?
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    (*A_copy)->rows = (*A)->rows;
    (*A_copy)->cols = (*A)->cols;
    for(int i = 0; i < (*A)->rows; i++) {
        for(int j = 0; j < (*A)->cols; j++) {
            (*A_copy)->data[i][j] = (*A)->data[i][j];
        }
    }
    return EXIT_SUCCESS;
}

int copy_matrix_ptr(double **A, double **A_copy, int rows, int cols) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A_copy[i][j] = A[i][j];
        }
    }
    return EXIT_SUCCESS;
}

//Copy submatrix of A into A_copy
int copy_submatrix(double **A, double **A_copy, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    // TODO check dimensions somehow
    
    int copy_row_idx = 0;
    int copy_col_idx = 0;
    for (int i = sub_row_beg; i < sub_row_end; i++) {
        copy_col_idx = 0;
        if (sub_col_beg == sub_col_end) {
            A_copy[copy_row_idx][copy_col_idx] = A[i][sub_col_beg];
        }
        for (int j = sub_col_beg; j < sub_col_end; j++) {
            //A_copy->data[copy_row_idx][copy_col_idx] = A->data[i][j];
            A_copy[copy_row_idx][copy_col_idx] = A[i][j];
            copy_col_idx++;
        }
        copy_row_idx++; 
    }
    return EXIT_SUCCESS;
}

// copy whole matrix A into submatrix of A_copy
int copy_matrix_to_submatrix(double **A, double **A_copy, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    int row_idx = 0;
    int col_idx = 0;
    for (int i = sub_row_beg; i < sub_row_end; i++) {
        col_idx = 0;
        for (int j = sub_col_beg; j < sub_col_end; j++) {
            A_copy[i][j] = A[row_idx][col_idx];
            col_idx++;
        }
        row_idx++; 
    }
    return EXIT_SUCCESS;
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
    //TODO: Is this a resource leak?
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } else if ((A->rows != B->rows) 
            || (A->cols != B->cols)) {
        return EXIT_FAILURE;
    } else {
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, A->rows, A->cols);
        for(int i = 0; i < C->rows; i++) {
            for(int j = 0; j < C->cols; j++) {
                C_copy->data[i][j] = alpha * A->data[i][j] + B->data[i][j];
            }
        }
        copy_matrix(C_copy, C);
        free_matrix(C_copy);
        return EXIT_SUCCESS;
    }
}

// Take submatrix of A, add it to B, and put that in a submatrix of C.
int axpy_submatrix(double **A, double **B, double**C, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    }
    // TODO check dimensions somehow
    // TODO is this a resource leak?
    
    int B_row_idx = 0;
    int B_col_idx = 0;
    Matrix *C_copy = NULL;
    init_matrix(&C_copy, sub_row_end - sub_row_beg, sub_col_end - sub_col_beg);
    copy_submatrix(C, C_copy->data, sub_row_beg, sub_row_end, sub_col_beg, sub_col_end);
    // take a submatrix of R, and add it to B
    for (int i = sub_row_beg; i < sub_row_end; i++) {
        B_col_idx = 0;
        for (int j = sub_col_beg; j < sub_col_end; j++) {
            //C_copy->data[i][j] = A[i][j] + B[B_row_idx][B_col_idx];
            C_copy->data[B_row_idx][B_col_idx] = A[i][j] + B[B_row_idx][B_col_idx];
            B_col_idx++;
        }
        B_row_idx++;
    }
    copy_matrix_to_submatrix(C_copy->data, C, sub_row_beg, sub_row_end, sub_col_beg, sub_col_end);
    free_matrix(C_copy);
    return EXIT_SUCCESS;
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
// C <- alpha * AB + beta * C
// A is (M, N)
// B is (N, 1)
// C is (M, 1)
int gemv(Matrix *A, Matrix *B, Matrix *C, double alpha, double beta) {
    // TODO check dimensions
    // TODO add transpose
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } 
    else {
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, A->rows, 1);
        for (int i = 0; i < A->rows; i++) {
            for (int k = 0; k < A->cols; k++) {
                C_copy->data[i][0] += alpha * (A->data[i][k] * B->data[k][0]) + beta * C->data[i][0];
            }
        }
        copy_matrix_deep(&C_copy, &C);
        free_matrix(C_copy);
        return EXIT_SUCCESS;
    }
}

// Column - ordered
int gemm_ptr(double *A, int A_rows, int A_cols,
             double *B, int B_rows, int B_cols,
             double *C, int C_rows, int C_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i + j * C_rows] = 0.0;
            for (int k = 0; k < A_cols; k++) {
                C[i + j * C_rows] += A[i + k * A_rows] * B[k + j * B_rows];
            }
        }
    }
}

int axpy_ptr(int n, double *x, double *y, double *alpha) {
    for (int i = 0; i < n; i++) {
        y[i] = (*alpha) * x[i] + y[i];
    }
}

int nrm2_ptr(int n, double *x, double *result) {
    *result = 0.0;
    for (int i = 0; i < n; i++) {
        *result += x[i] * x[i];
    }
    *result = sqrt(*result);
}

// Matrix-matrix product
int gemm(matrix_operation_t transa, matrix_operation_t transb, Matrix *A, Matrix *B, Matrix *C, double alpha, double beta) {
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } 
    if (transa == MATRIX_OP_N && transb == MATRIX_OP_N) {
        // C <-- alpha * AB + beta * C
        // A is (M, N)
        // B is (N, K)
        // C is (M, K)
        if (A->cols != B->rows) {
            return EXIT_FAILURE;
        }
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, A->rows, B->cols);
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < B->cols; j++) {
                for (int k = 0; k < A->cols; k++) {
                    C_copy->data[i][j] += alpha * (A->data[i][k] * B->data[k][j]) + beta * C->data[i][j];
                }
            }
        }
        copy_matrix_deep(&C_copy, &C);
        free_matrix(C_copy);
        return EXIT_SUCCESS;
    } else if (transa == MATRIX_OP_T && transb == MATRIX_OP_N) {
        // C <-- A.T @ B
        // A.T is (N, M), A is (M, N)
        // B is (M, K)
        // C is (N, k)
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, A->cols, B->cols);
        for (int i = 0; i < A->cols; i++) {
            for (int j = 0; j < B->cols; j++) {
                for (int k = 0; k < B->rows; k++) {
                    C_copy->data[i][j] += alpha * (A->data[k][i] * B->data[k][j]); // + beta * C->data[i][j];
                }
            }
        }
        copy_matrix_deep(&C_copy, &C);
        free_matrix(C_copy);
        return EXIT_SUCCESS;
    } else if (transa == MATRIX_OP_N && transb == MATRIX_OP_T) {
        // C <-- A @ B.T
        // A is (M, N)
        // B is (K, N), B.T is (N, K)
        // C is (M, K)
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, A->rows, B->rows);
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < B->rows; j++) {
                for (int k = 0; k < B->cols; k++) {
                    C_copy->data[i][j] += alpha * (A->data[i][k] * B->data[j][k]); // + beta * C->data[i][j];
                }
            }
        }
        copy_matrix_deep(&C_copy, &C);
        free_matrix(C_copy);
        return EXIT_SUCCESS;
    } else if (transa == MATRIX_OP_T && transb == MATRIX_OP_T){
        return EXIT_FAILURE;
    } else {
        return EXIT_FAILURE;
    }
}

int is_zero(Matrix *A) {
    if (A == NULL) {
        return EXIT_FAILURE;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (A->data[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}
