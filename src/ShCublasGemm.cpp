#include "ShCublasGemm.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

ShCublasGemm::ShCublasGemm(float *denseMatrix, int nRows, int nCols):
    ShGemm(nRows, nCols)
{
    m_pDenseMatrix = denseMatrix;
    cublasCreate(&m_cublasHandle);
}

// C = A * self[Sparse]
// Since cusparse only support C=self[sparse]^T*A[dense]^T, we need to transpose both A and self
double ShCublasGemm::mutiplyBy(float *A, int nRows, int nCols, float *C) 
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasStatus_t status = cublasSgemm(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_nCols, nRows, nCols, &alpha, m_pDenseMatrix, m_nCols, A, nCols, &beta, C, m_nCols);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Fail by cublasSgemm, status: %d\n", status);
        exit(1);
    }
    return 0.0;
}

double ShCublasGemm::mutiply(float *B, int nRows, int nCols, float *C)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasStatus_t status = cublasSgemm(m_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, nCols, m_nRows, m_nCols, &alpha, B, nCols, m_pDenseMatrix, m_nCols, &beta, C, nCols);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Fail by cublasSgemm, status: %d\n", status);
        exit(1);
    }
    return 0.0;

}

ShCublasGemm::~ShCublasGemm() {
    cublasDestroy(m_cublasHandle);
}
