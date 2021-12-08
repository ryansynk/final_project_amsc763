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
        status = status || dot(v, v, beta);
        *beta = 2 / *beta;
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

    Matrix *v_init = NULL;
    init_matrix(&v_init, A->rows, 1);
    double beta = 0.0;

    for (int k = 0; k < A->cols; k++) {
        Matrix x = {A->rows - k, 1, A->data + k};
        Matrix v = {v_init->rows - k, 1, v_init->data + k};
        status = status || house(&x, &v, &beta);
    }
    return status;
}
