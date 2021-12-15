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

__global__ void print_ptr_num(double *A) {
    // Data stored in column-major order
    printf("%f\n", *A);
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
//TODO get rid of cols parameter
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

    double* dev_Y = NULL;
    double* dev_W = NULL;

    double* dev_B = NULL;
    double* dev_Vmat = NULL;
    double* B = NULL;

    double* dev_Yt_v = NULL;
    double* dev_WYt_v = NULL;

    // intermediate -- used for updating R
    double* dev_WTR = NULL;  // "W transpose times R" 
    double* dev_YWTR = NULL; // "Y times W transpose times R" 

    // intermediate -- used for updating Q
    double* dev_WYT = NULL;  // "W times Y transpose"
    double* dev_QWYT = NULL; // "Q times W times Y transpose"

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

    cudaStat = cudaMalloc(&dev_Y, A_rows * r * sizeof(double));
    cudaStat = cudaMalloc(&dev_W, A_rows * r * sizeof(double));

    cudaStat = cudaMalloc(&dev_Yt_v, r * sizeof(double));
    cudaStat = cudaMalloc(&dev_WYt_v, A_rows * sizeof(double));

    cudaStat = cudaMalloc(&dev_WTR, r * (R_cols - r) * sizeof(double));
    cudaStat = cudaMalloc(&dev_YWTR, R_rows * (R_cols - r) * sizeof(double));

    cudaStat = cudaMalloc(&dev_WYT, R_rows * R_rows * sizeof(double));
    cudaStat = cudaMalloc(&dev_QWYT, Q_rows * R_rows * sizeof(double));

        // R[s:, s + r:] = R[s:, s + r:] + Y @ (W.T @ R[s:, s + r:])
        // gemm  dev_WTR <-- W.T @ (R + s + (s + r) * R_rows)
        // gemm  dev_YWTR <-- Y @ dev_WTR
        // axpy  R + s + (s + r) * R_rows <-- dev_YWTR + (R + s + (s + r) * R_rows)
        //
        // Update Q
        // Q[:, s:] = Q[:, s:] + Q[:, s:] @ W  @ Y.T
        // gemm dev_WYT <-- W @ Y.T
        // gemm dev_QWYT <-- (Q + s * Q_rows) @ dev_WYT
        // axpy (Q + s * Q_rows) <-- dev_QWYT + (Q + s * Q_rows)

        // R[s:, s + r:] is,  at max, R_rows x R_cols - r
        // W is R_rows x r, so W.T @ R[s:, s + r:] = dev_WTR is r x (R_cols - r)
        // Y is R_rows x r, so Y @ dev_WTR = dev_YWTR is R_rows x (R_cols - r)

        // W is R_rows x r, Y is R_rows x r, so W @ Y.T = dev_WYT is R_rows x R_rows
        // at max, Q[:, s:] is Q_rows x Q_rows, so Q[:, s:] @ (W @ Y.T) = dev_QWYT is (Q_rows x R_rows)

    // V is m x r, v is m x 1
    // Y is m x r, Y.T @ v = Yt_v is r x 1
    // W is m x r, W @ Yt_v = WYt_v is m x 1
    // z is m x 1

    // V is (m - s) x r, 

        // gemv Yt_v <-- Y.T @ v
        // gemm WYt_v <-- -B[j] * W @ Yt_v
        // gemm z <-- -B[j]v + WYt_v

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

            //gpu_house<<<1,1>>>(handle, k, R_rows, R_cols, dev_R, dev_v, dev_beta);
            //cudaStat = cudaMemcpy(&beta, dev_beta, sizeof(double), cudaMemcpyDeviceToHost);
            //cudaDeviceSynchronize();

            //// Gets dev_Rv = R[k:m, k:n] @ v
            //stat = cublasDgemv(handle, CUBLAS_OP_T, 
            //                   R_rows - k, R_cols - k, 
            //                   &alpha, 
            //                   (dev_R + k + k * R_rows), R_rows,
            //                   (dev_v + k), 1,
            //                   &gamma,
            //                   (dev_Rv + k), 1);

            printf("j = %d\n", j);
            u = s + j;
            //gpu_house<<<1,1>>>(handle, u, R_rows, R_cols, dev_R, dev_Vmat + u * R_rows, dev_beta);
            printf("\n");
            printf("===== gpu_house =====\n");

            printf("R[u:, u]\n");
            print_ptr<<<1,1>>>(dev_R + u + u * R_rows, R_rows - u, 1);
            cudaDeviceSynchronize();

            printf("u = %d\n", u);

            gpu_house<<<1,1>>>(handle, u, R_rows, R_cols, dev_R, (dev_Vmat + j * R_rows + j - u), dev_beta);
            cudaStat = cudaMemcpy((B + j), dev_beta, sizeof(double), cudaMemcpyDeviceToHost);

            printf("v = \n");
            print_ptr<<<1,1>>>(dev_Vmat + j * R_rows + j, R_rows - j, 1);
            cudaDeviceSynchronize();
            printf("beta = %f\n", *(B + j));
            //cudaStat = cudaMemcpy(&beta, dev_beta, sizeof(double), cudaMemcpyDeviceToHost);
            //B[j] = beta;
            // R[u:m, u:(s + r)].T @ v

            printf("\n");
            printf("===== cublasDgemv =====\n");

            printf("R = \n");
            print_ptr<<<1,1>>>(dev_R, R_rows, R_cols);
            cudaDeviceSynchronize();

            printf("R[u:m, u:(s + r)] = \n");
            print_ptr<<<1,1>>>(dev_R + u + u * R_rows, R_rows - u, (s + r) - u);
            cudaDeviceSynchronize();

            printf("dev_Vmat[:, j] = \n");
            print_ptr<<<1,1>>>(dev_Vmat + j + j * R_rows, R_rows - j, 1);
            cudaDeviceSynchronize();

            stat = cublasDgemv(handle, CUBLAS_OP_T, 
                               R_rows - u, (s + r) - u, 
                               &alpha, 
                               (dev_R + u + u * R_rows), R_rows,
                               //(dev_Vmat + u + u * R_rows), 1,
                               (dev_Vmat + j + j * R_rows), 1,
                               &gamma,
                               (dev_Rv + u), 1);

            printf("R[u:m, u:(s + r)].T @ v\n");
            print_ptr<<<1,1>>>((dev_Rv + u), (s + r) - u, 1);
            cudaDeviceSynchronize();

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
            printf("\n");
            printf("===== cublasDger =====\n");

            printf("dev_Vmat[:, j] = \n");
            print_ptr<<<1,1>>>(dev_Vmat + j + j * R_rows, R_rows - j, 1);
            cudaDeviceSynchronize();

            printf("dev_Rv + u\n");
            print_ptr<<<1,1>>>((dev_Rv + u), (s + r) - u, 1);
            cudaDeviceSynchronize();

            printf("B[j] = %f\n", *(B + j));
            cudaDeviceSynchronize();

            printf("rows = %d, cols = %d\n", R_rows - u, (s + r) - u);

            stat = cublasDger(handle, 
                              R_rows - u, (s + r) - u,
                              (B + j),
                              //(dev_Vmat + u + u * R_rows), 1,
                              (dev_Vmat + j + j * R_rows), 1,
                              (dev_Rv + u), 1,
                              (dev_R + u + u * R_rows), R_rows);

            printf("R[u:m, u:(s + r)] = \n");
            print_ptr<<<1,1>>>(dev_R + u + u * R_rows, R_rows - u, (s + r) - u);
            cudaDeviceSynchronize();

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

            //printf("dev_B[j] = \n");
            //print_ptr_num<<<1,1>>>((dev_B + j));
            //cudaDeviceSynchronize();

            //printf("beta = %f\n", beta);
            //cudaDeviceSynchronize();
            printf("===== R at end of j loop =====\n");
            printf("R = \n");
            print_ptr<<<1,1>>>(dev_R, R_rows, R_cols);
            cudaDeviceSynchronize();
        }

        // copy W, Y vectors

        //Y = np.atleast_2d(V[:,0]).T
        // copy first column of dev_Vmat into first column of Y
        printf("\n");
        printf("===== GENERATING W, Y =====\n");
        cudaStat = cudaMemcpy(dev_Y, dev_Vmat, A_rows * sizeof(double), cudaMemcpyDeviceToDevice); // Y = V[:, 0]
        cudaStat = cudaMemcpy(dev_W, dev_Vmat, A_rows * sizeof(double), cudaMemcpyDeviceToDevice); // W = V[:, 0]
        cublasDscal(handle, A_rows, B, dev_W, 1); // W = -B[0] * V[:, 0]

        for (int i = 1; i < r; i++) {
            // gemv Yt_v <-- Y.T @ v
            printf("=====  i = %d  =====\n", i);
            printf("Y = \n");
            print_ptr<<<1,1>>>(dev_Y, A_rows, r);
            cudaDeviceSynchronize();
            printf("v = \n");
            print_ptr<<<1,1>>>((dev_Vmat + i * A_rows), A_rows, 1);
            cudaDeviceSynchronize();

            stat = cublasDgemv(handle, CUBLAS_OP_T, 
                               //A_rows, r,
                               A_rows, i,
                               &alpha, 
                               dev_Y, A_rows,
                               //(dev_Vmat + i * A_rows), A_rows,
                               (dev_Vmat + i * A_rows), 1,
                               &gamma,
                               dev_Yt_v, 1);

            printf("Yt_v = \n");
            print_ptr<<<1,1>>>(dev_Yt_v , r, 1);
            cudaDeviceSynchronize();

            printf("W = \n");
            print_ptr<<<1,1>>>(dev_W, A_rows, r);
            cudaDeviceSynchronize();

            // gemm WYt_v <-- -B[j] * W @ Yt_v
            stat = cublasDgemv(handle, CUBLAS_OP_N, 
                               A_rows, i,
                               (B + i), 
                               dev_W, A_rows,
                               dev_Yt_v, 1,
                               &gamma,
                               dev_WYt_v, 1);

            printf("WYt_v = \n");
            print_ptr<<<1,1>>>(dev_WYt_v, A_rows, 1);
            cudaDeviceSynchronize();

            // axpy WYt_v <-- -B[j]v + WYt_v
            stat = cublasDaxpy(handle, A_rows, 
                               (B + i),
                               dev_Vmat + i * A_rows, 1,
                               dev_WYt_v, 1);

            // scal WYt_v <-- -B[j] * WYt_v
            //cublasDscal(handle, A_rows, (B + i), dev_WYt_v, 1);

            printf("z = \n");
            print_ptr<<<1,1>>>(dev_WYt_v, A_rows, 1);
            cudaDeviceSynchronize();

            // memcpy Y + r*A_rows <-- v
            cudaStat = cudaMemcpy(dev_Y + i * A_rows, dev_Vmat + i * A_rows, A_rows * sizeof(double), cudaMemcpyDeviceToDevice);
            // memcpy W + r*A_rows <-- z
            cudaStat = cudaMemcpy(dev_W + i * A_rows, dev_WYt_v, A_rows * sizeof(double), cudaMemcpyDeviceToDevice);

            printf("Y = \n");
            print_ptr<<<1,1>>>(dev_Y, A_rows, r);
            cudaDeviceSynchronize();

            printf("W = \n");
            print_ptr<<<1,1>>>(dev_W, A_rows, r);
            cudaDeviceSynchronize();
        }

        printf("===== COMPLETED W, Y UPDATE =====\n");

        printf("W = \n");
        print_ptr<<<1,1>>>(dev_W, A_rows, r);
        cudaDeviceSynchronize();
        printf("Y = \n");
        print_ptr<<<1,1>>>(dev_Y, A_rows, r);
        cudaDeviceSynchronize();

        printf("===== UPDATING Q,R =====\n");

        // Update Q, R
        // Update R
        // R[s:, s + r:] = R[s:, s + r:] + Y @ (W.T @ R[s:, s + r:])
        // W.T is (r x R_rows). R[s: s + r:] is (R_rows - s) x (R_cols - (s + r))
        // gemm  dev_WTR <-- W.T @ (R + s + (s + r) * R_rows)
        stat = cublasDgemm(handle, 
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           r, R_cols - (s + r), R_rows - s,
                           &alpha,
                           dev_W, R_rows,
                           (dev_R + s + (s + r) * R_rows), R_rows,
                           &gamma,
                           dev_WTR, r);
        // gemm  dev_YWTR <-- Y @ dev_WTR
        // Y is A_rows x r
        // dev_WTR is r x (R_cols - (s + r))
        stat = cublasDgemm(handle, 
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           A_rows, R_cols - (s + r), r,
                           &alpha, 
                           dev_Y, A_rows,
                           dev_WTR, r,
                           &gamma,
                           dev_YWTR, A_rows);

        // axpy  R + s + (s + r) * R_rows <-- dev_YWTR + (R + s + (s + r) * R_rows)
        stat = cublasDaxpy(handle, A_rows * (R_cols - (s + r)), &alpha, dev_YWTR, 1, (dev_R + s + (s + r) * R_rows), 1);

        printf("===== FINAL R AFTER BLOCK k = %d =====\n", k);
        print_ptr<<<1,1>>>(dev_R, R_rows, R_cols);
        cudaDeviceSynchronize();

        // Update Q
        // Q[:, s:] = Q[:, s:] + Q[:, s:] @ W  @ Y.T
        // gemm dev_WYT <-- W @ Y.T
        // W is (A_rows - s ) x r, Y.T is r x (A_rows - s)
        printf("first dgemm\n");
        stat = cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_T,
                           R_rows - s, R_rows - s, r,
                           &alpha,
                           dev_W, R_rows,
                           dev_Y, R_rows,
                           &gamma,
                           dev_WYT, R_rows);

        printf("R_rows - s = %d\n", R_rows - s);
        printf("r = %d\n", r);
        printf("dev_W = %p\n", dev_W);
        printf("R_rows= %d\n", R_rows);
        printf("dev_Y = %p\n", dev_Y);
        printf("r = %d\n", r);
        printf("gamma = %f\n", gamma);
        printf("dev_WYT = %p\n", dev_WYT);
        printf("R_rows = %d\n", R_rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("stat = %s\n", _cudaGetErrorEnum(stat));
            cublasDestroy(handle);
            return EXIT_FAILURE;
        } 
        printf("dev_WYT\n");
        print_ptr<<<1,1>>>(dev_WYT, R_rows - s, R_rows - s);
        cudaDeviceSynchronize();

        // gemm dev_QWYT <-- (Q + s * Q_rows) @ dev_WYT
        // Q[:, s:] is A_rows x (A_rows - s)
        // dev_WYT is (A_rows - s) x (A_rows - s)
        printf("second dgemm\n");
        stat = cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           A_rows, A_rows - s, A_rows - s,
                           &alpha,
                           dev_Q + s * Q_rows, Q_rows,
                           dev_WYT, A_rows,
                           &gamma,
                           dev_QWYT, A_rows);
        // axpy (Q + s * Q_rows) <-- dev_QWYT + (Q + s * Q_rows)
        stat = cublasDaxpy(handle, Q_rows * (Q_cols - s), &alpha, dev_QWYT, 1, (dev_Q + s * Q_rows), 1);

        printf("===== FINAL Q AFTER BLOCK k = %d =====\n", k);
        print_ptr<<<1,1>>>(dev_Q, Q_rows, Q_cols);
        cudaDeviceSynchronize();

        // R[s:, s + r:] is,  at max, R_rows x R_cols - r
    }

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
