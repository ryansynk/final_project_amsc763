#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

/*
void print_matrix(Matrix *A);
Matrix *new_matrix(int rows, int cols);
Matrix *rand_matrix(int rows, int cols);
void free_matrix(Matrix *A);
*/

void print_matrix(Matrix *A) {
    //TODO check null
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            printf("%f ", A->data[i][j]);
        }
        printf("\n");
    }
}

Matrix *new_matrix(int rows, int cols) {
    //TODO check null
    Matrix *A = (Matrix*) calloc(1, sizeof(*A));
    if (A == NULL) {
        return NULL;
    }
    A->rows = rows;
    A->cols = cols;
    A->data = (double**) calloc(A->rows, sizeof(*A->data));

    for(int i = 0; i < A->rows; i++) {
        A->data[i] = (double*) calloc(A->cols, sizeof(**A->data));
        if (A->data[i] == NULL) {
            return NULL;
        }
    }

    return A;
}

Matrix *rand_matrix(int rows, int cols) {
    //TODO Check null
    Matrix *A = new_matrix(rows, cols);

    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            A->data[i][j] = (double) rand() / ((double) RAND_MAX + 1);
        }
    }

    return A;
}

void free_matrix(Matrix *A) {
    for(int i = 0; i < A->rows; i++) {
        free(A->data[i]);
    }
    free(A->data);
    free(A);
}

void add_matrix(Matrix *A, Matrix *B, Matrix *sum) {
    if (A->rows != B->rows || A->cols != B->cols || sum == NULL) {
        sum = NULL;
    } else {
        for(int i = 0; i < sum->rows; i++) {
            for(int j = 0; j < sum->cols; j++) {
                sum->data[i][j] = A->data[i][j] + B->data[i][j];
            }
        }
    }
}

double dot(Matrix *A, Matrix *B) {
    double out_val = 0.0;

    if (A == NULL || B == NULL) {
        return -1.0;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        return -1.0;
    } else {
        for (int i = 0; i < A->rows; i++) {
            for (int j = 0; j < B->rows; j++) {
                out_val = out_val + A->data[i][j] * B->data[i][j];
            }
        }
    }

    return out_val;
}

int main(int argc, char *argv[]) {
    int m, n;
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

    C = new_matrix(3, 3);

    add_matrix(A, B, C);
    print_matrix(C);

    dot_val = dot(A, B);

    printf("dot(A,B) = %f\n", dot_val);


    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}
