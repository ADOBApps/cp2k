/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "gemm_c_api_gpu.h"

/*******************************************************************************
 * Error-checking macros
 ******************************************************************************/

#if defined(__OFFLOAD_CUDA)

#define CUBLAS_CHECK(cmd)                                                      \
  do {                                                                         \
    cublasStatus_t status__ = (cmd);                                           \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "CUBLAS_ERROR: %s:%d ", __FILE__, __LINE__);             \
      switch (status__) {                                                      \
      case CUBLAS_STATUS_NOT_INITIALIZED:                                      \
        fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED\n");                    \
        break;                                                                 \
      case CUBLAS_STATUS_ALLOC_FAILED:                                         \
        fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED\n");                       \
        break;                                                                 \
      case CUBLAS_STATUS_INVALID_VALUE:                                        \
        fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_ARCH_MISMATCH:                                        \
        fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_MAPPING_ERROR:                                        \
        fprintf(stderr, "CUBLAS_STATUS_MAPPING_ERROR\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_EXECUTION_FAILED:                                     \
        fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED\n");                   \
        break;                                                                 \
      case CUBLAS_STATUS_INTERNAL_ERROR:                                       \
        fprintf(stderr, "CUBLAS_STATUS_INTERNAL_ERROR\n");                     \
        break;                                                                 \
      case CUBLAS_STATUS_NOT_SUPPORTED:                                        \
        fprintf(stderr, "CUBLAS_STATUS_NOT_SUPPORTED\n");                      \
        break;                                                                 \
      case CUBLAS_STATUS_LICENSE_ERROR:                                        \
        fprintf(stderr, "CUBLAS_STATUS_LICENSE_ERROR\n");                      \
        break;                                                                 \
      default:                                                                 \
        fprintf(stderr, "Unknown CUBLAS error %d\n", status__);                \
      }                                                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t status__ = (cmd);                                              \
    if (status__ != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA_ERROR: %s %s:%d\n", cudaGetErrorString(status__),  \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#else
#define CUBLAS_CHECK(cmd) ((void)0)
#define CUDA_CHECK(cmd) ((void)0)
#endif

/*******************************************************************************
 * Internal context structure
 ******************************************************************************/
struct gemm_ctx_gpu {
    cublasHandle_t handle;      /** cuBLAS handle */
    offloadStream_t stream;     /** CUDA stream for async ops */
    size_t threshold;           /** m*n*k threshold for GPU dispatch */
    int tensor_cores_enabled;   /** 1 if Tensor Cores are enabled */
    int compute_capability_major; /** CUDA compute capability major version */
    int compute_capability_minor; /** CUDA compute capability minor version */
};

/*******************************************************************************
 * Helper: Detect compute capability
 ******************************************************************************/
static void detect_compute_capability(int *major, int *minor) {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, device_id));
}

/*******************************************************************************
 * Helper: Get cuBLAS operation type from character
 ******************************************************************************/
static inline cublasOperation_t get_cublas_op(char trans) {
    return (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
}

/*******************************************************************************
 * Public API
 ******************************************************************************/

gemm_ctx_gpu_t *gemm_ctx_gpu_create(offloadStream_t stream) {
    gemm_ctx_gpu_t *ctx = (gemm_ctx_gpu_t *)calloc(1, sizeof(gemm_ctx_gpu_t));
    if (!ctx) {
        fprintf(stderr, "gemm_ctx_gpu_create: calloc failed\n");
        return NULL;
    }

    ctx->stream = stream;
    ctx->threshold = 0; // Default: always use GPU
    ctx->tensor_cores_enabled = 0;

    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&ctx->handle));
    CUBLAS_CHECK(cublasSetStream(ctx->handle, stream));

    // Detect compute capability
    detect_compute_capability(
      &ctx->compute_capability_major,
      &ctx->compute_capability_minor
    );

    ctx->tensor_cores_enabled = 0;

    return ctx;
}

void gemm_ctx_gpu_destroy(gemm_ctx_gpu_t *ctx) {
    if (!ctx) return;
    if (ctx->handle) {
        CUBLAS_CHECK(cublasDestroy(ctx->handle));
    }
    free(ctx);
}

void gemm_ctx_gpu_set_threshold(gemm_ctx_gpu_t *ctx, size_t threshold) {
    if (ctx) ctx->threshold = threshold;
}

void gemm_ctx_gpu_set_tensor_cores(gemm_ctx_gpu_t *ctx, bool enable) {
    if (!ctx) return;
    
    if (ctx->compute_capability_major < 7) {
        fprintf(stderr, "WARNING: Tensor Cores require Volta+ (CC >= 7.0)\n");
        return;
    }

    ctx->tensor_cores_enabled = enable ? 1 : 0;
  #if defined(__OFFLOAD_CUDA)
    cublasMath_t mode = enable ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    CUBLAS_CHECK(cublasSetMathMode(ctx->handle, mode));
  #endif
}

void gemm_ctx_gpu_set_math_mode(gemm_ctx_gpu_t *ctx, unsigned int mode) {
    if (ctx) {
  #if defined(__OFFLOAD_CUDA)
        CUBLAS_CHECK(cublasSetMathMode(ctx->handle, (cublasMath_t)mode));
  #endif
    }
}

void gemm_ctx_gpu_dgemm(
  gemm_ctx_gpu_t *ctx, char transa, char transb,
  int m, int n, int k,
  double alpha, const double *A, int lda,
  const double *B, int ldb,
  double beta, double *C, int ldc) {
    if (!ctx) {
        fprintf(stderr, "gemm_ctx_gpu_dgemm: NULL context\n");
        abort();
    }

    // Unlike gemm_c_api.c's host-pointer API, this device-only API has no meaningful CPU fallback.
    size_t ops = (size_t)m * n * k;
    if (ctx->threshold > 0 && ops < ctx->threshold) {
        fprintf(stderr,
                "FATAL: gemm_ctx_gpu_dgemm threshold=%zu ops=%zu would "
                "silently skip this GEMM call (C left unmodified). This "
                "device-only API has no CPU fallback -- do not call "
                "gemm_ctx_gpu_set_threshold() on a gemm_ctx_gpu_t context.\n",
                ctx->threshold, ops);
        abort();
    }

    cublasOperation_t opA = get_cublas_op(transa);
    cublasOperation_t opB = get_cublas_op(transb);

    // All pointers are expected to be device pointers
    // No host-device copies - matrices must already reside on GPU

    CUBLAS_CHECK(cublasDgemm(ctx->handle, opA, opB,
                             m, n, k,
                             &alpha, A, lda,
                             B, ldb,
                             &beta, C, ldc));
}

void gemm_ctx_gpu_sgemm(
  gemm_ctx_gpu_t *ctx, char transa, char transb,
  int m, int n, int k,
  float alpha, const float *A, int lda,
  const float *B, int ldb,
  float beta, float *C, int ldc) {
    if (!ctx) {
        fprintf(stderr, "gemm_ctx_gpu_sgemm: NULL context\n");
        abort();
    }

    size_t ops = (size_t)m * n * k;
    if (ctx->threshold > 0 && ops < ctx->threshold) {
        fprintf(stderr,
                "FATAL: gemm_ctx_gpu_sgemm threshold=%zu ops=%zu would "
                "silently skip this GEMM call (C left unmodified). This "
                "device-only API has no CPU fallback -- do not call "
                "gemm_ctx_gpu_set_threshold() on a gemm_ctx_gpu_t context.\n",
                ctx->threshold, ops);
        abort();
    }

    cublasOperation_t opA = get_cublas_op(transa);
    cublasOperation_t opB = get_cublas_op(transb);

    CUBLAS_CHECK(cublasSgemm(ctx->handle, opA, opB,
                             m, n, k,
                             &alpha, A, lda,
                             B, ldb,
                             &beta, C, ldc));
}

offloadStream_t gemm_ctx_gpu_get_stream(gemm_ctx_gpu_t *ctx) {
    return ctx ? ctx->stream : 0;
}

bool gemm_ctx_gpu_tensor_cores_enabled(gemm_ctx_gpu_t *ctx) {
    return ctx ? (ctx->tensor_cores_enabled != 0) : false;
}

const char *gemm_ctx_gpu_get_info(gemm_ctx_gpu_t *ctx) {
    if (!ctx) return "NULL context";
    
    static char info[256];
    snprintf(info, sizeof(info),
             "GPU GEMM Context | CC: %d.%d | Tensor Cores: %s | Stream: %p",
             ctx->compute_capability_major,
             ctx->compute_capability_minor,
             ctx->tensor_cores_enabled ? "Enabled" : "Disabled",
             (void*)ctx->stream);
    return info;
}