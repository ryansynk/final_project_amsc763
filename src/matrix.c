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

// Initializes new matrix struct
int init_matrix(Matrix **A, int rows, int cols) {
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
        for(int i = 0; i < A->rows; i++) {
            for(int j = 0; j < A->cols; j++) {
                A->data[i][j] = (double) rand() / ((double) RAND_MAX + 1);
            }
        }
        return EXIT_SUCCESS;
    }
}

int copy_matrix(Matrix *A, Matrix *A_copy) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    if (A->rows != A_copy->rows || A->cols != A_copy->cols) {
        return EXIT_FAILURE;
    }
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A_copy->data[i][j] = A->data[i][j];
        }
    }
    return EXIT_SUCCESS;
}

int new_copy_matrix(Matrix *A, Matrix *A_copy, int rows, int cols) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            A_copy->data[i][j] = A->data[i][j];
        }
    }
    return EXIT_SUCCESS;
}

//int copy_submatrix(Matrix *A, Matrix *A_copy, int A_copy_rows, int A_copy_cols, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
int copy_submatrix(double **A, double **A_copy, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
    if (A == NULL || A_copy == NULL) {
        return EXIT_FAILURE;
    }
    // TODO check dimensions somehow
    
    int copy_row_idx = 0;
    int copy_col_idx = 0;
    for (int i = sub_row_beg; i < sub_row_end; i++) {
        copy_col_idx = 0;
        for (int j = sub_col_beg; j < sub_col_end; j++) {
            //A_copy->data[copy_row_idx][copy_col_idx] = A->data[i][j];
            A_copy[copy_row_idx][copy_col_idx] = A[i][j];
            copy_col_idx++;
        }
        copy_row_idx++; 
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

int axpy_submatrix(double **A, double **B, double**C, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end) {
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    }
    // TODO check dimensions somehow
    
    int B_row_idx = 0;
    int B_col_idx = 0;
    // take a submatrix of R, and add it to B
    for (int i = sub_row_beg; i < sub_row_end; i++) {
        B_col_idx = 0;
        for (int j = sub_col_beg; j < sub_col_end; j++) {
            C[i][j] = A[i][j] + B[B_row_idx][B_col_idx];
            B_col_idx++;
        }
        B_row_idx++;
    }
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
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } 
    else if (C->rows != A->rows || B->rows != A->cols) {
        return EXIT_FAILURE;
    }
    else if (B->cols != 1 || C->cols != 1) {
        return EXIT_FAILURE;
    } else {
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, C->rows, C->cols);
        // TODO: Do i need this?
        //zero_matrix(C_copy);
        for (int i = 0; i < C->rows; i++) {
            //for (int j = 0; j < C->cols; j++) {
                for (int k = 0; k < A->cols; k++) {
                    C_copy->data[i][0] += alpha * (A->data[i][k] * B->data[k][0]) + beta * C->data[i][0];
                }
           // }
        }
        copy_matrix(C_copy, C);
        return EXIT_SUCCESS;
    }
}

// Matrix-matrix product
// C <-- alpha * AB + beta * C
// A is (M, N)
// B is (N, K)
// C is (M, K)
int gemm(matrix_operation_t transa, matrix_operation_t transb, Matrix *A, Matrix *B, Matrix *C, double alpha, double beta) {
    if (A == NULL || B == NULL || C == NULL) {
        return EXIT_FAILURE;
    } 
    if (transa == MATRIX_OP_N && transb == MATRIX_OP_N) {
        if (C->rows != A->rows || C->cols != B->cols) {
            return EXIT_FAILURE;
        }
        Matrix *C_copy = NULL;
        init_matrix(&C_copy, C->rows, C->cols);
        zero_matrix(C_copy);
        for (int i = 0; i < C->rows; i++) {
            for (int j = 0; j < C->cols; j++) {
                for (int k = 0; k < A->cols; k++) {
                    //C->data[i][j] += alpha * (A->data[i][k] * B->data[k][j]) + beta * C->data[i][j];
                    C_copy->data[i][j] += alpha * (A->data[i][k] * B->data[k][j]) + beta * C->data[i][j];
                    //C->data[i][j] += alpha * (A->data[i][k] * B->data[k][j]);
                }
            }
        }
        copy_matrix(C_copy, C);
        return EXIT_SUCCESS;
    } else if (transa == MATRIX_OP_T && transb == MATRIX_OP_N) {
        return EXIT_FAILURE;
    } else if (transa == MATRIX_OP_N && transb == MATRIX_OP_T) {
        return EXIT_FAILURE;
    } else if (transa == MATRIX_OP_T && transb == MATRIX_OP_T){
        return EXIT_FAILURE;
    } else {
        return EXIT_FAILURE;
    }
}
