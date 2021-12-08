#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test.h"
#include "matrix.h"
#include "qr.h"

int main(int argc, char *argv[]) {
    //Matrix *A = NULL;
    //new_matrix(&A, 3, 3);
    //print_matrix(A);

    Matrix *A = NULL;
    Matrix *B = NULL;
    Matrix *C = NULL;

    double *dot_val = calloc(1, sizeof(double));
    *dot_val = 0.0;
    int status;

    srand(time(NULL));

    new_matrix(&A, 3, 3);
    new_matrix(&B, 3, 3);
    new_matrix(&C, 3, 3);

    eye_matrix(A);
    /*
    A->data[0][0] = 1.0; A->data[0][1] = 0.0; A->data[0][2] = 0.0;
    A->data[1][0] = 0.0; A->data[1][1] = 1.0; A->data[1][2] = 0.0;
    A->data[2][0] = 0.0; A->data[2][1] = 0.0; A->data[2][2] = 1.0;
    */

    printf("A = \n");
    print_matrix(A);

    rand_matrix(B);
    printf("B = \n");
    print_matrix(B);

    axpy(A, B, C, 1.0);
    printf("A + B = \n");
    print_matrix(C);
    axpy(A, B, C, 2.0);
    printf("A + 2B = \n");
    print_matrix(C);

    mdot(A, B, dot_val);

    printf("mdot(A,B) = %f\n", *dot_val);

    qr(A, B, C);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
