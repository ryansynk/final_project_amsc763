#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int house(Matrix *x, Matrix *v, double *beta) {
    int status = EXIT_SUCCESS;
    if ((v == NULL || x == NULL) 
            || (v->rows != x->rows) 
            || (v->cols != 1 || x->cols != 1)) {
        status = EXIT_FAILURE;
    } else {
        double norm_x = 0.0;
        status = status || norm(x, &norm_x);

        // Generate basis vector
        for (int i = 0; i < v->rows; i++) {
            v->data[i][0] = 0.0;
        }
        v->data[0][0] = 1.0;

        status = status || axpy(v, x, v, -norm_x);
        *beta = 0.0;
        if (!is_zero(v)) { // if v is entirely zero, then just return with beta as 0.0
            status = status || dot(v, v, beta);
            *beta = 2 / *beta;
        }
    }
    return status;
}

int qr(Matrix *A, Matrix *Q, Matrix *R) {
    int status = EXIT_SUCCESS;
    if (A == NULL || Q == NULL || R == NULL) {
        status = EXIT_FAILURE;
    } 
    if (Q->rows != A->rows || Q->cols != A->rows) {
        status = EXIT_FAILURE;
    } 
    if (R->rows != A->rows || R->cols != A->cols) {
        status = EXIT_FAILURE;
    }

    eye_matrix(Q);
    copy_matrix(A, R);

    Matrix *R_submatrix = NULL;
    Matrix *Q_submatrix = NULL;
    init_matrix(&R_submatrix, R->rows, R->cols);
    init_matrix(&Q_submatrix, Q->rows, Q->cols);

    Matrix *v_init = NULL;
    init_matrix(&v_init, A->rows, 1);
    double beta = 0.0;

    Matrix *x = NULL;
    init_matrix(&x, R->rows, 1);

    for (int k = 0; k < A->cols; k++) {
        copy_submatrix(R->data, x->data, k, R->rows, k, k);
        x->rows = R->rows - k;
        Matrix v = {v_init->rows - k, 1, v_init->data + k};
        status = status || house(x, &v, &beta);

        // Get R[k:m, k:n]
        copy_submatrix(R->data, R_submatrix->data, k, R->rows, k, R->cols);
        R_submatrix->rows = R->rows - k;
        R_submatrix->cols = R->cols - k;

        // Get Q[:, k:m]
        copy_submatrix(Q->data, Q_submatrix->data, 0, Q->rows, k, Q->cols);
        Q_submatrix->cols = Q->rows - k;

        gemm(MATRIX_OP_T, MATRIX_OP_N, &v, R_submatrix, R_submatrix, 1.0, 0.0);        // R[k:m, k:n] <-- v.T @ R[k:m, k:n]              
        gemm(MATRIX_OP_N, MATRIX_OP_N, &v, R_submatrix, R_submatrix, -beta, 0.0);      // R[k:m, k:n] <-- -beta * v @ R[k:m, k:n]        
        axpy_submatrix(R->data, R_submatrix->data, R->data, k, R->rows, k, R->cols);   // R[k:m, k:n] <-- R[k:m, k:n] (OLD) + R[k:m, k:n]

        gemv(Q_submatrix, &v, Q_submatrix, -beta, 0.0);                                // Q[:, k:m] <-- -beta (Q[:, k:m] @ v)      
        gemm(MATRIX_OP_N, MATRIX_OP_T, Q_submatrix, &v, Q_submatrix, 1.0, 0.0);        // Q[:, k:m] <-- Q[:, k:m] @ v.T           
        axpy_submatrix(Q->data, Q_submatrix->data, Q->data, 0, Q->rows, k, Q->cols);   // Q[:, k:m] <-- Q[:, k:m](OLD) + Q[:, k:m]
    }

    free_matrix(R_submatrix);
    free_matrix(Q_submatrix);
    free_matrix(v_init);
    free_matrix(x);

    return status;
}
