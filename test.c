#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "test.h"
#include "matrix.h"

int main(int argc, char *argv[]) {
    //int m, n;
    Matrix *A;
    Matrix *B;
    Matrix *C;
    double dot_val;

    srand(time(NULL));

    A = new_matrix(3, 3);

    A->data[0][0] = 1.0; A->data[0][1] = 0.0; A->data[0][2] = 0.0;
    A->data[1][0] = 0.0; A->data[1][1] = 1.0; A->data[1][2] = 0.0;
    A->data[2][0] = 0.0; A->data[2][1] = 0.0; A->data[2][2] = 1.0;

    print_matrix(A);

    B = rand_matrix(3, 3);
    print_matrix(B);

    C = axpy(A, B, 1.0);
    print_matrix(C);
    C = axpy(A, B, 2.0);
    print_matrix(C);

    dot_val = dot(A, B);

    printf("dot(A,B) = %f\n", dot_val);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
