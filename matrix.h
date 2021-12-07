#include <stdio.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

void print_matrix(Matrix *A);
Matrix *new_matrix(int rows, int cols);
Matrix *rand_matrix(int rows, int cols);
Matrix *zero_matrix(int rows, int cols);
Matrix *eye_matrix(int rows);
void free_matrix(Matrix *A);
Matrix *axpy(Matrix *A, Matrix *B, double alpha);
double dot(Matrix *A, Matrix *B);
double norm(Matrix *A);
double mdot(Matrix *A, Matrix *B);
Matrix *gemv(Matrix *A, Matrix *B);
Matrix *gemm(int M, int N, int K, Matrix *A, Matrix *B);
