#include <stdio.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

int print_matrix(Matrix *A);
int new_matrix(Matrix **A, int rows, int cols);
int zero_matrix(Matrix *A);
int eye_matrix(Matrix *A);
int rand_matrix(Matrix *A);
void free_matrix(Matrix *A);
int axpy(Matrix *A, Matrix *B, Matrix *C, double alpha);
int dot(Matrix *A, Matrix *B, double *d);
int norm(Matrix *A, double *d);
int mdot(Matrix *A, Matrix *B, double *d);
int gemv(Matrix *A, Matrix *x, Matrix *y, double alpha, double beta);
int gemm(int M, int N, int K, Matrix *A, Matrix *B, Matrix *C);
