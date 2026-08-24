/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#ifndef GEMM_C_API_GPU_H
#define GEMM_C_API_GPU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include "../offload/offload_runtime.h"
#if defined(__OFFLOAD_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#else
/* Keep the host-only build type-correct; cuBLAS calls are disabled below. */
typedef void *cublasHandle_t;
typedef int cublasOperation_t;
typedef void *offloadStream_t;
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#endif

/*******************************************************************************
 * Opaque GPU GEMM context
 ******************************************************************************/
typedef struct gemm_ctx_gpu gemm_ctx_gpu_t;

/*******************************************************************************
 * Create a GPU GEMM context with a CUDA stream.
 *
 * \param[in] stream  CUDA stream for asynchronous operations
 * \return            Context handle, or NULL on failure
 *
 * Notes:
 *   - Requires CUDA device with compute capability >= 3.5
 *   - Tensor Cores enabled automatically for Volta+ (CC >= 7.0)
 *   - The stream is used for all operations and must remain valid
 *     for the lifetime of the context
 ******************************************************************************/
gemm_ctx_gpu_t *gemm_ctx_gpu_create(offloadStream_t stream);

/*******************************************************************************
 * Destroy a GPU GEMM context and release resources.
 *
 * \param[in,out] ctx  Context to destroy (may be NULL)
 ******************************************************************************/
void gemm_ctx_gpu_destroy(gemm_ctx_gpu_t *ctx);

/*******************************************************************************
 * Set the operation-size threshold below which GPU is bypassed.
 *
 * Small matrices may be faster on CPU due to kernel launch overhead.
 *
 * \param[in,out] ctx       Context
 * \param[in]     threshold Product m*n*k below which to use CPU fallback
 ******************************************************************************/
void gemm_ctx_gpu_set_threshold(gemm_ctx_gpu_t *ctx, size_t threshold);

/*******************************************************************************
 * Enable or disable Tensor Cores (Volta+ only).
 *
 * \param[in,out] ctx    Context
 * \param[in]     enable true to enable Tensor Cores, false to disable
 ******************************************************************************/
void gemm_ctx_gpu_set_tensor_cores(gemm_ctx_gpu_t *ctx, bool enable);

/*******************************************************************************
 * Set cuBLAS math mode.
 *
 * \param[in,out] ctx   Context
 * \param[in]     mode  cuBLAS math mode (CUBLAS_DEFAULT_MATH,
 *                      CUBLAS_TENSOR_OP_MATH, CUBLAS_FAST_MATH)
 ******************************************************************************/
void gemm_ctx_gpu_set_math_mode(gemm_ctx_gpu_t *ctx, unsigned int mode);

/*******************************************************************************
 * Double-precision real GEMM on device pointers: C = alpha*op(A)*op(B) + beta*C
 *
 * All pointers must be device pointers (allocated via offloadMalloc).
 * No host-device copies are performed - matrices must already reside on GPU.
 *
 * \param[in,out] ctx    Context
 * \param[in]     transa 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * \param[in]     transb 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * \param[in]     m      Number of rows of op(A) and C
 * \param[in]     n      Number of columns of op(B) and C
 * \param[in]     k      Number of columns of op(A) and rows of op(B)
 * \param[in]     alpha  Scalar multiplier for A*B
 * \param[in]     A      Device pointer to matrix A (lda x k or lda x m)
 * \param[in]     lda    Leading dimension of A
 * \param[in]     B      Device pointer to matrix B (ldb x n or ldb x k)
 * \param[in]     ldb    Leading dimension of B
 * \param[in]     beta   Scalar multiplier for C
 * \param[in,out] C      Device pointer to output matrix C (ldc x n or ldc x m)
 * \param[in]     ldc    Leading dimension of C
 *
 * Memory layout: Column-major (Fortran order).
 * All pointers are expected to be valid device pointers.
 ******************************************************************************/
void gemm_ctx_gpu_dgemm(gemm_ctx_gpu_t *ctx, char transa, char transb,
                        int m, int n, int k,
                        double alpha, const double *A, int lda,
                        const double *B, int ldb,
                        double beta, double *C, int ldc);

/*******************************************************************************
 * Single-precision real GEMM on device pointers.
 *
 * \param[in,out] ctx    Context
 * \param[in]     transa 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * \param[in]     transb 'N' or 'n' = no-transpose, 'T' or 't' = transpose
 * \param[in]     m      Number of rows of op(A) and C
 * \param[in]     n      Number of columns of op(B) and C
 * \param[in]     k      Number of columns of op(A) and rows of op(B)
 * \param[in]     alpha  Scalar multiplier for A*B
 * \param[in]     A      Device pointer to matrix A
 * \param[in]     lda    Leading dimension of A
 * \param[in]     B      Device pointer to matrix B
 * \param[in]     ldb    Leading dimension of B
 * \param[in]     beta   Scalar multiplier for C
 * \param[in,out] C      Device pointer to output matrix C
 * \param[in]     ldc    Leading dimension of C
 ******************************************************************************/
void gemm_ctx_gpu_sgemm(gemm_ctx_gpu_t *ctx, char transa, char transb,
                        int m, int n, int k,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta, float *C, int ldc);

/*******************************************************************************
 * Get the CUDA stream associated with the context.
 *
 * \param[in] ctx  Context
 * \return         CUDA stream handle
 ******************************************************************************/
offloadStream_t gemm_ctx_gpu_get_stream(gemm_ctx_gpu_t *ctx);

/*******************************************************************************
 * Query whether Tensor Cores are enabled.
 *
 * \param[in] ctx  Context
 * \return         true if Tensor Cores are enabled
 ******************************************************************************/
bool gemm_ctx_gpu_tensor_cores_enabled(gemm_ctx_gpu_t *ctx);

/*******************************************************************************
 * Get a human-readable backend description.
 *
 * \param[in] ctx  Context
 * \return         Static string (do not free)
 ******************************************************************************/
const char *gemm_ctx_gpu_get_info(gemm_ctx_gpu_t *ctx);


#endif /* GEMM_C_API_GPU_H */