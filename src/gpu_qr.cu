#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void print_ptr(double *A, int A_rows, int A_cols) {
    // Data stored in column-major order
    //
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            printf("%f ", A[j*A_rows + i]);
        }
        printf("\n");
    }
}

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
    if (d_v == NULL || d_x == NULL || beta == NULL) {
        return;
    }
    double norm_x = 0.0;
    *beta = 0.0;
    // Data stored in column-major order
    // d_v gets first column
    for (int i = 0; i < rows - k; i++) {
        (d_v + k)[i] = (d_x + k + k * rows)[i];
        norm_x += (d_x + k + k * rows)[i] * (d_x + k + k * rows)[i];
    }

    norm_x = sqrt(norm_x);
    if (norm_x == 0.0) {
        *beta = 0.0;
    } else {
        (d_v + k)[0] = (d_v + k)[0] - norm_x;

        for (int i = 0; i < rows - k; i++) {
            *beta += (d_v + k)[i] * (d_v + k)[i];
        }
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

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("stat = %s\n", _cudaGetErrorEnum(stat));
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    double* dev_A = NULL;
    double* dev_Q = NULL;
    double* dev_R = NULL;
    double* dev_x = NULL;
    double* dev_v = NULL;
    double* dev_Rv = NULL;
    double* dev_Qv = NULL;
    double* dev_beta = NULL;
    double* dev_alpha = NULL;
    double* dev_gamma = NULL;

    double beta = 0.0;
    double alpha = 1.0;
    double gamma = 0.0;

    cudaStat = cudaMalloc(&dev_A, A_rows * A_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_Q, Q_rows * Q_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_R, R_rows * R_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_x, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_v, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_Rv, R_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_Qv, Q_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_beta, sizeof(double));
    cudaStat = cudaMalloc(&dev_alpha, sizeof(double));
    cudaStat = cudaMalloc(&dev_gamma, sizeof(double));

    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed\n");
        return EXIT_FAILURE;
    }

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
        gpu_house<<<1,1>>>(handle, k, R_rows, R_cols, dev_R, dev_v, dev_beta);
        cudaStat = cudaMemcpy(&beta, dev_beta, sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Gets dev_Rv = R[k:m, k:n] @ v
        stat = cublasDgemv(handle, CUBLAS_OP_T, 
                           R_rows - k, R_cols - k, 
                           &alpha, 
                           (dev_R + k + k * R_rows), R_rows,
                           (dev_v + k), 1,
                           &gamma,
                           (dev_Rv + k), 1);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            cudaFree(dev_A);
            cudaFree(dev_Q);
            cudaFree(dev_R);
            cudaFree(dev_x);
            cudaFree(dev_v);
            cudaFree(dev_Rv);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        // Sets R[k:m, k:n] = R[k:m, k:n] - beta * v @ dev_Rv.T
        stat = cublasDger(handle, 
                          R_rows - k, R_cols - k,
                          &beta,
                          (dev_v + k), 1,
                          (dev_Rv + k), 1,
                          (dev_R + k + k * R_rows), R_rows);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            cudaFree(dev_A);
            cudaFree(dev_Q);
            cudaFree(dev_R);
            cudaFree(dev_x);
            cudaFree(dev_v);
            cudaFree(dev_Rv);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }

        // Gets dev_Qv = Q[:, k:m] @ v
        stat = cublasDgemv(handle, CUBLAS_OP_N, 
                           Q_rows, Q_cols - k, 
                           &alpha, 
                           (dev_Q + k * Q_rows), Q_rows,
                           (dev_v + k), 1,
                           &gamma,
                           (dev_Qv + k), 1);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            cudaFree(dev_A);
            cudaFree(dev_Q);
            cudaFree(dev_R);
            cudaFree(dev_x);
            cudaFree(dev_v);
            cudaFree(dev_Rv);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        printf("beta = %f\n", beta);

        // Sets Q[:, k:m] = Q[:, k:m] - beta * (Q[:, k:m] @ v) @ v.T
        stat = cublasDger(handle, 
                          Q_rows, Q_cols - k,
                          &beta,
                          (dev_Qv + k), 1,
                          (dev_v + k), 1,
                          (dev_Q + k * Q_rows), Q_rows);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
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


extern "C" int gpu_block_qr(double *A, double *Q, double *R, int A_rows, int A_cols, int r) {
    int status = EXIT_SUCCESS;
    if (A == NULL || Q == NULL || R == NULL) {
        status = EXIT_FAILURE;
    } 

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

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("stat = %s\n", _cudaGetErrorEnum(stat));
        cublasDestroy(handle);
        return EXIT_FAILURE;
    } 

    double* dev_A = NULL;
    double* dev_Q = NULL;
    double* dev_R = NULL;
    double* dev_x = NULL;
    //double* dev_v = NULL;
    double* dev_Rv = NULL;
    double* dev_Qv = NULL;
    double* dev_beta = NULL;
    double* dev_alpha = NULL;
    double* dev_gamma = NULL;

    double* dev_B = NULL;
    double* dev_Vmat = NULL;
    double* B = NULL;

    double beta = 0.0;
    double alpha = 1.0;
    double gamma = 0.0;
    int s = 0;
    int u = 0;

    cudaStat = cudaMalloc(&dev_A, A_rows * A_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_Q, Q_rows * Q_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_R, R_rows * R_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_x, R_rows * sizeof(double));
    //cudaStat = cudaMalloc(&dev_v, R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_Rv, R_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_Qv, Q_cols * sizeof(double));
    cudaStat = cudaMalloc(&dev_beta, sizeof(double));
    cudaStat = cudaMalloc(&dev_alpha, sizeof(double));
    cudaStat = cudaMalloc(&dev_gamma, sizeof(double));

    cudaStat = cudaMalloc(&dev_B, r * sizeof(double));
    cudaStat = cudaMalloc(&dev_Vmat, A_rows * r * sizeof(double));

    B = (double *)malloc(r * sizeof(double));

    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed\n");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMemcpy(dev_A, A, A_rows * A_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_Q, Q, Q_rows * Q_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_R, R, R_rows * R_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(dev_gamma, &gamma, sizeof(double), cudaMemcpyHostToDevice);

    if (cudaStat != cudaSuccess) {
        printf("host to device memory copy failed\n");
        return EXIT_FAILURE;
    }

    for (int k = 0; k < (A_cols / r); k++) {
        printf("k = %d\n", k);
        s = k * r;
        //TODO: zero out dev_Vmat?
        for (int j = 0; j < r; j++) {
            printf("j = %d\n", j);
            u = s + j;
            gpu_house<<<1,1>>>(handle, u, R_rows, R_cols, dev_R, dev_Vmat + u * R_rows, dev_beta);
            cudaStat = cudaMemcpy((B + j), dev_beta, sizeof(double), cudaMemcpyDeviceToHost);
            // R[u:m, u:(s + r)].T @ v
            stat = cublasDgemv(handle, CUBLAS_OP_T, 
                               R_rows - u, (s + r) - u, 
                               &alpha, 
                               (dev_R + u + u * R_rows), R_rows,
                               (dev_Vmat + u + u * R_rows), 1,
                               &gamma,
                               (dev_Rv + u), 1);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf("stat = %s\n", _cudaGetErrorEnum(stat));
                cudaFree(dev_A);
                cudaFree(dev_Q);
                cudaFree(dev_R);
                cudaFree(dev_x);
                //cudaFree(dev_v);
                cudaFree(dev_Rv);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }

            // Sets R[u:m, u:(s+r)] = R[u:m, u:(s+r)] - beta * v @ dev_Rv.T
            stat = cublasDger(handle, 
                              R_rows - u, (s + r) - u,
                              &beta,
                              (dev_Vmat + u + u * R_rows), 1,
                              (dev_Rv + u), 1,
                              (dev_R + u + u * R_rows), R_rows);

            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf("stat = %s\n", _cudaGetErrorEnum(stat));
                cudaFree(dev_A);
                cudaFree(dev_Q);
                cudaFree(dev_R);
                cudaFree(dev_x);
                //cudaFree(dev_v);
                cudaFree(dev_Rv);
                cublasDestroy(handle);
                return EXIT_FAILURE;
            }
            printf("R[u:m, u:(s + r)] = \n");
            print_ptr<<<1,1>>>(dev_R + u + u * R_rows, R_rows - u, r - u);
            cudaDeviceSynchronize();

            printf("dev_Vmat[:, j] = \n");
            print_ptr<<<1,1>>>(dev_Vmat + u, R_rows - u, 1);
            cudaDeviceSynchronize();

            printf("dev_B[j] = \n");
            print_ptr<<<1,1>>>((dev_B + j), 1, 1);
            cudaDeviceSynchronize();
        }
    }


    //for (int k = 0; k < A_cols; k++) {
    //    printf("k = %d\n", k);
    //    gpu_house<<<1,1>>>(handle, k, R_rows, R_cols, dev_R, dev_v, dev_beta);
    //    cudaStat = cudaMemcpy(&beta, dev_beta, sizeof(double), cudaMemcpyDeviceToHost);

    //    printf("dev_v = \n");
    //    print_ptr<<<1,1>>>(dev_v, R_rows, 1);
    //    cudaDeviceSynchronize();

    //    // Gets dev_Rv = R[k:m, k:n] @ v
    //    stat = cublasDgemv(handle, CUBLAS_OP_T, 
    //                       R_rows - k, R_cols - k, 
    //                       &alpha, 
    //                       (dev_R + k + k * R_rows), R_rows,
    //                       (dev_v + k), 1,
    //                       &gamma,
    //                       (dev_Rv + k), 1);

    //    if (stat != CUBLAS_STATUS_SUCCESS) {
    //        printf("stat = %s\n", _cudaGetErrorEnum(stat));
    //        cudaFree(dev_A);
    //        cudaFree(dev_Q);
    //        cudaFree(dev_R);
    //        cudaFree(dev_x);
    //        cudaFree(dev_v);
    //        cudaFree(dev_Rv);
    //        cublasDestroy(handle);
    //        return EXIT_FAILURE;
    //    }
    //    printf("beta = %f\n", beta);

    //    // Sets R[k:m, k:n] = R[k:m, k:n] - beta * v @ dev_Rv.T
    //    stat = cublasDger(handle, 
    //                      R_rows - k, R_cols - k,
    //                      &beta,
    //                      (dev_v + k), 1,
    //                      (dev_Rv + k), 1,
    //                      (dev_R + k + k * R_rows), R_rows);

    //    if (stat != CUBLAS_STATUS_SUCCESS) {
    //        printf("stat = %s\n", _cudaGetErrorEnum(stat));
    //        cudaFree(dev_A);
    //        cudaFree(dev_Q);
    //        cudaFree(dev_R);
    //        cudaFree(dev_x);
    //        cudaFree(dev_v);
    //        cudaFree(dev_Rv);
    //        cublasDestroy(handle);
    //        return EXIT_FAILURE;
    //    }

    //    // Gets dev_Qv = Q[:, k:m] @ v
    //    stat = cublasDgemv(handle, CUBLAS_OP_N, 
    //                       Q_rows, Q_cols - k, 
    //                       &alpha, 
    //                       (dev_Q + k * Q_rows), Q_rows,
    //                       (dev_v + k), 1,
    //                       &gamma,
    //                       (dev_Qv + k), 1);

    //    if (stat != CUBLAS_STATUS_SUCCESS) {
    //        printf("stat = %s\n", _cudaGetErrorEnum(stat));
    //        cudaFree(dev_A);
    //        cudaFree(dev_Q);
    //        cudaFree(dev_R);
    //        cudaFree(dev_x);
    //        cudaFree(dev_v);
    //        cudaFree(dev_Rv);
    //        cublasDestroy(handle);
    //        return EXIT_FAILURE;
    //    }
    //    printf("beta = %f\n", beta);

    //    // Sets Q[:, k:m] = Q[:, k:m] - beta * (Q[:, k:m] @ v) @ v.T
    //    stat = cublasDger(handle, 
    //                      Q_rows, Q_cols - k,
    //                      &beta,
    //                      (dev_Qv + k), 1,
    //                      (dev_v + k), 1,
    //                      (dev_Q + k * Q_rows), Q_rows);

    //    if (stat != CUBLAS_STATUS_SUCCESS) {
    //        printf("stat = %s\n", _cudaGetErrorEnum(stat));
    //        cudaFree(dev_A);
    //        cudaFree(dev_Q);
    //        cudaFree(dev_R);
    //        cudaFree(dev_x);
    //        cudaFree(dev_v);
    //        cudaFree(dev_Rv);
    //        cublasDestroy(handle);
    //        return EXIT_FAILURE;
    //    }
    //}

    cudaStat = cudaMemcpy(A, dev_A, A_rows * A_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(Q, dev_Q, Q_rows * Q_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaStat = cudaMemcpy(R, dev_R, R_rows * R_cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_A);
    cudaFree(dev_Q);
    cudaFree(dev_R);
    cudaFree(dev_x);
    //cudaFree(dev_v);
    cudaFree(dev_Rv);
    cublasDestroy(handle);
    return status;
}
