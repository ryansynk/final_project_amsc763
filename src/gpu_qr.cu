#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

// On-device householder transform
//__global__ void gpu_house(cublasHandle_t handle, int row_beg, int rows, int cols, double *d_x, double *d_v, double *beta, int col) {
__global__ void gpu_house(cublasHandle_t handle, int k, int rows, int cols, double *d_x, double *d_v, double *beta) {
    __syncthreads();
    if (d_v == NULL || d_x == NULL || beta == NULL) {
        return;
    }
    double norm_x = 0.0;
    *beta = 0.0;
    for (int i = 0; i < rows - k; i++) {
        __syncthreads();
        //printf("d_x[%d] = %f\n", i*cols, d_x[i*cols]);
        //(dev_v + k)[i] = (dev_R + k + k * R_cols)[i*R_cols];
        printf("(d_x + k + k * cols)[i*cols] = %f\n", (d_x + k + k * cols)[i*cols]);
        (d_v + k)[i] = (d_x + k + k * cols)[i*cols];
    __syncthreads();
        norm_x += (d_x + k + k * cols)[i*cols] * (d_x + k + k * cols)[i*cols];
    __syncthreads();
        *beta += (d_v + k)[i] * (d_v + k)[i];
    }

    __syncthreads();
    norm_x = sqrt(norm_x);
    printf("norm_x = %f\n", norm_x);
    if (norm_x == 0.0) {
        *beta = 0.0;
    } else {
        printf("norm_x = %f\n", norm_x);
        (d_v + k)[0] = (d_v + k)[0] - norm_x;
        *beta = -2.0 / *beta;
    }
}

// extern "C"
//int gpu_qr(Matrix *A, Matrix *Q, Matrix *R, int A_rows, int A_cols) {
extern "C" int gpu_qr(double *A, double *Q, double *R, int A_rows, int A_cols) {
    int status = EXIT_SUCCESS;
    if (A == NULL || Q == NULL || R == NULL) {
        status = EXIT_FAILURE;
    } 
    //if (Q->rows != A->rows || Q->cols != A->rows) {
    //    status = EXIT_FAILURE;
    //} 
    //if (R->rows != A->rows || R->cols != A->cols) {
    //    status = EXIT_FAILURE;
    //}
    //Q is an A_rows x A_rows matrix

    int Q_rows = A_rows;
    int Q_cols = A_rows;
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_rows; j++) {
            if (i == j) {
                Q[i * A_rows + j] = 1.0;
            } else {
                Q[i * A_rows + j] = 0.0;
            }
        }
    }

    int R_rows = A_rows;
    int R_cols = A_cols;
    for (int i = 0; i < A_rows * A_cols; i++) {
        R[i] = A[i];
    }
    //eye_matrix(Q);
    //copy_matrix(A, R);

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    double* dev_A = NULL;
    double* dev_Q = NULL;
    double* dev_R = NULL;
    double* dev_x = NULL;
    double* dev_v = NULL;
    double* dev_Rv = NULL;
    double* dev_beta = NULL;
    double* dev_alpha = NULL;
    double* dev_gamma = NULL;

    //double beta = 0.0;
    double alpha = 1.0;
    double gamma = 0.0;

    //cudaStat = cudaMalloc((double**)&dev_A, A->rows * A->cols * sizeof(double));
    //cudaStat = cudaMalloc((double**)&dev_Q, Q->rows * Q->cols * sizeof(double));
    //cudaStat = cudaMalloc((double**)&dev_R, R->rows * R->cols * sizeof(double));
    //cudaStat = cudaMalloc((double**)&dev_x, R->rows * sizeof(double));
    //cudaStat = cudaMalloc((double**)&dev_v, R->rows * sizeof(double));
    //cudaStat = cudaMalloc((double**)&dev_Rv, R->rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_A, A_rows * A_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_Q, Q_rows * Q_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_R, R_rows * R_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_x, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_v, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_Rv, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_beta, sizeof(double));
    cudaStat = cudaMalloc(&dev_alpha, sizeof(double));
    cudaStat = cudaMalloc(&dev_gamma, sizeof(double));

    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed\n");
        return EXIT_FAILURE;
    }

    //cudaStat = cudaMemcpy(dev_A, *(A->data), A->rows * A->cols * sizeof(double), cudaMemcpyHostToDevice);
    //cudaStat = cudaMemcpy(dev_Q, *(Q->data), Q->rows * Q->cols * sizeof(double), cudaMemcpyHostToDevice);
    //cudaStat = cudaMemcpy(dev_R, *(R->data), R->rows * R->cols * sizeof(double), cudaMemcpyHostToDevice);

    cudaStat = cudaMemcpy(dev_A, A, A_rows * A_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_Q, Q, Q_rows * Q_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_R, R, R_rows * R_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_gamma, &gamma, sizeof(double), cudaMemcpyHostToDevice);

    if (cudaStat != cudaSuccess) {
        printf("host to device memory copy failed\n");
        return EXIT_FAILURE;
    }
    for (int k = 0; k < A_cols; k++) {
        printf("inside loop\n");
        printf("before gpu_house\n");
        //gpu_house<<<1,1>>>(handle, k, R_rows, R_cols, dev_R, dev_v, &beta, k);
        cudaDeviceSynchronize();
        gpu_house<<<1,1>>>(handle, k, R_rows, R_cols, dev_R, dev_v, dev_beta);
        cudaDeviceSynchronize();

        //gpu_house
        //norm_x = 0.0;
        //beta = 0.0;
        ////printf("rows -k = %d\n", rows -k);
        //for (int i = 0; i < R_rows - k; i++) {
        //    //printf("d_x[%d] = %f\n", i*R_cols, d_x[i*R_cols]);
        //    printf("here\n");
        //    (dev_v + k)[i] = 0.0;
        //    printf("here2\n");
        //    (dev_v + k)[i] = (dev_R + k + k * R_cols)[i*R_cols];
        //    printf("here3\n");
        //    norm_x += (dev_R + k + k * R_cols)[i*R_cols] * (dev_R + k + k * R_cols)[i*R_cols];
        //    printf("here4\n");
        //    beta += (dev_v + k)[i] * (dev_v + k)[i];
        //    printf("here5\n");
        //}

        ////for(int i = row_beg; i < rows; i++) {
        ////    printf("in loop\n");
        ////    printf("i = %d\n", i);
        ////    d_v[i] = d_x[i*cols + col];
        ////    norm_x += d_x[i*cols + col] * d_x[i*cols + col];
        ////    *beta += d_v[i] * d_v[i];
        ////}

        //norm_x = sqrt(norm_x);
        //if (norm_x == 0.0) {
        //    beta = 0.0;
        //} else {
        //    (dev_v+k)[0] = (dev_v+k)[0] - norm_x;
        //    beta = -2.0 / beta;
        //}

        printf("before cublasDgemv\n");
        // Gets dev_Rv = R[k:m, k:n] @ v
        stat = cublasDgemv(handle, CUBLAS_OP_N, 
                           R_rows - k, R_cols - k, 
                           dev_alpha, 
                           dev_R + k + k * R_cols, R_rows,
                           dev_v + k, 1,
                           dev_gamma,
                           dev_Rv + k, 1);
        cudaDeviceSynchronize();
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            //printf("R update failed, stat = %d\n", stat);
            cudaFree(dev_A);
            cudaFree(dev_Q);
            cudaFree(dev_R);
            cudaFree(dev_x);
            cudaFree(dev_v);
            cudaFree(dev_Rv);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        printf("before cublasDger\n");
        // Sets R[k:m, k:n] = R[k:m, k:n] - beta * v @ dev_Rv.T
        stat = cublasDger(handle, 
                          R_rows - k, R_cols - k,
                          dev_beta, 
                          dev_v + k, 1,
                          dev_Rv + k, 1,
                          dev_R + k + k * R_cols, R_rows);
        cudaDeviceSynchronize();

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            //printf("R update failed, stat = %d\n", stat);
            cudaFree(dev_A);
            cudaFree(dev_Q);
            cudaFree(dev_R);
            cudaFree(dev_x);
            cudaFree(dev_v);
            cudaFree(dev_Rv);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

    }

    cudaStat = cudaMemcpy(A, dev_A, A_rows * A_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(Q, dev_Q, Q_rows * Q_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(R, dev_R, R_rows * R_cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_Q);
    cudaFree(dev_R);
    cudaFree(dev_x);
    cudaFree(dev_v);
    cudaFree(dev_Rv);
    cublasDestroy(handle);
    return status;
}
