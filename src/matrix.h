#include <stdio.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

typedef enum {
    MATRIX_OP_N,
    MATRIX_OP_T
} matrix_operation_t;

int readtxt(char *fname, double *A, int A_rows, int A_cols);
int print_matrix(Matrix *A);
int print_ptr(double *A, int A_rows, int A_cols);
int init_matrix(Matrix **A, int rows, int cols);
int zero_matrix(Matrix *A);
int eye_matrix(Matrix *A);
int rand_matrix(Matrix *A);
int rand_ptr(double *A, int size);
int copy_matrix(Matrix *A, Matrix *A_copy);
int copy_submatrix(double **A, double **A_copy, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end);
void free_matrix(Matrix *A);
int axpy(Matrix *A, Matrix *B, Matrix *C, double alpha);
int axpy_submatrix(double **A, double **B, double**C, int sub_row_beg, int sub_row_end, int sub_col_beg, int sub_col_end);
int dot(Matrix *A, Matrix *B, double *d);
int norm(Matrix *A, double *d);
int mdot(Matrix *A, Matrix *B, double *d);
int gemv(Matrix *A, Matrix *x, Matrix *y, double alpha, double beta);
int gemm(matrix_operation_t transa, matrix_operation_t transb, 
        Matrix *A, Matrix *B, Matrix *C, 
        double alpha, double beta);
int gemm_ptr(double *A, int A_rows, int A_cols,
             double *B, int B_rows, int B_cols,
             double *C, int C_rows, int C_cols);
int is_zero(Matrix *A);


int axpy_ptr(int n, double *x, double *y, double *alpha);

int nrm2_ptr(int n, double *x, double *result);
