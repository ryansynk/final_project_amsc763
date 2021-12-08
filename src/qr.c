#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int house(Matrix *x, Matrix *v) {
    if ((v == NULL || x == NULL) 
            || (v->rows != x->rows) 
            || (v->cols != 1 || x->cols != 1)) {
        return EXIT_FAILURE;
    } else {
        double* norm_x = calloc(1, sizeof(double));
        norm(x, norm_x);
        double neg_norm_x = -1* (*norm_x);

        for (int i = 0; i < v->rows; i++) {
            v->data[i][0] = 0.0;
        }
        v->data[0][0] = 1.0;

        axpy(v, x, v, neg_norm_x);
        return EXIT_SUCCESS;
    }
}

int qr(Matrix *A, Matrix *Q, Matrix *R) {
    return EXIT_SUCCESS;
}
