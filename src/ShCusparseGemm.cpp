#include "ShCusparseGemm.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <chrono>
#include "utils.h"

ShCusparseGemm::ShCusparseGemm(float *denseMatrix, int nRows, int nCols):
    ShGemm(nRows, nCols)
{
    m_pDenseMatrix = denseMatrix;
    cusparseCreate(&m_cusparseHandle);
    cublasCreate(&m_cublasHandle);

    cusparseCreateMatDescr(&m_descrB);
    cusparseSetMatType(m_descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(m_descrB, CUSPARSE_INDEX_BASE_ZERO);
}

// C = A * self[Sparse]
// Since cusparse only support C=self[sparse]^T*A[dense]^T, and matrices are column major, we don't need to transpose both A and self
double ShCusparseGemm::mutiplyBy(float *A, int nRows, int nCols, float *C) 
{
    //1. Convert self dense matrix to CSR format
    //1.1 Calculate nnz per row
    int *pNnzPerRow;
    int *nnz_u;
    int nnz;
    auto start = std::chrono::steady_clock::now();
    checkCudaErrors(cudaMalloc((void **)&(pNnzPerRow), sizeof(int) * (m_nCols + 1)));
    checkCudaErrors(cudaMalloc((void **)&(nnz_u), sizeof(int) * 1));
    int *nnz_h = (int *)malloc(sizeof(int) * 1);

    cusparseStatus_t status;
    status = cusparseSnnz(m_cusparseHandle, 
            CUSPARSE_DIRECTION_ROW, 
            m_nCols,
            m_nRows,
            m_descrB,
            m_pDenseMatrix,
            m_nCols,
            pNnzPerRow,
            nnz_u);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(nnz_h, nnz_u, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    nnz = nnz_h[0];
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSnnz, status: %d\n", status);
        exit(1);
    }
    //printf("nnz: %d\n", nnz);

    float *pCsrVal;
    int *pCsrRow;
    int *pCsrInd;

    checkCudaErrors(cudaMalloc((void **)&(pCsrVal), sizeof(float) * nnz));
    checkCudaErrors(cudaMalloc((void **)&(pCsrRow), sizeof(int) * (m_nCols+1)));
    checkCudaErrors(cudaMalloc((void **)&(pCsrInd), sizeof(int) * nnz));

    status = cusparseSdense2csr(m_cusparseHandle, 
            m_nCols, 
            m_nRows, 
            m_descrB,
            m_pDenseMatrix,
            m_nCols, //leading dimension
            pNnzPerRow,
            pCsrVal,
            pCsrRow,
            pCsrInd);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSdense2csr, status: %d\n", status);
        exit(1);
    }

    auto end = std::chrono::steady_clock::now();
    int timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    printf("EO:%d\n", timeUsed);

    float alpha = 1.0;
    float beta = 0.0;
    start = std::chrono::steady_clock::now();
    status = cusparseScsrmm2(m_cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            m_nCols,
            nRows,
            m_nRows,
            nnz,
            &alpha,
            m_descrB,
            pCsrVal,
            pCsrRow,
            pCsrInd,
            A,
            m_nRows,
            &beta,
            C,
            m_nCols);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseScsrmm, status: %d\n", status);
        exit(1);
    }
    end = std::chrono::steady_clock::now();
    timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    printf("KC:%d\n", timeUsed);

    cudaFree(pNnzPerRow);
    cudaFree(pCsrVal);
    cudaFree(pCsrRow);
    cudaFree(pCsrInd);
    cudaDeviceSynchronize();
    free(nnz_h);
    return 0.0;
}

// C = self[Sparse] * B[Dense]
double ShCusparseGemm::mutiply(float *B, int nRows, int nCols, float *C) 
{
    //1. Transpose A[Self] first
    float alpha = 1.0;
    float beta = 0.0;
    float *AT;
    float *CT;
    if (DEBUG) {
        checkCudaErrors(cudaMallocManaged((void **)&(AT), sizeof(float) * m_nCols * m_nRows));
        checkCudaErrors(cudaMallocManaged((void **)&(CT), sizeof(float) * m_nRows * nCols));
    } else {
        checkCudaErrors(cudaMalloc((void **)&(AT), sizeof(float) * m_nCols * m_nRows));
        checkCudaErrors(cudaMalloc((void **)&(CT), sizeof(float) * m_nRows * nCols));
    }
	cublasSgeam(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m_nRows, m_nCols, &alpha, m_pDenseMatrix, m_nCols, &beta, NULL, m_nCols, AT, m_nRows); 
    cudaDeviceSynchronize();
    if (DEBUG) {
        printf("AT: \n");
        print_array(AT, m_nRows, m_nCols);
    }

    //2. Convert self dense matrix to CSR format
    //2.1 Calculate nnz per row
    int *pNnzPerRow;
    int *nnz_u;
    int nnz;
    if (DEBUG) {
        checkCudaErrors(cudaMallocManaged((void **)&(pNnzPerRow), sizeof(int) * (m_nCols + 1)));
        checkCudaErrors(cudaMallocManaged((void **)&(nnz_u), sizeof(int) * 1));
    } else {
        checkCudaErrors(cudaMalloc((void **)&(pNnzPerRow), sizeof(int) * (m_nCols + 1)));
        checkCudaErrors(cudaMalloc((void **)&(nnz_u), sizeof(int) * 1));
    }
    int *nnz_h = (int *)malloc(sizeof(int) * 1);

    cusparseStatus_t status;
    status = cusparseSnnz(m_cusparseHandle, 
            CUSPARSE_DIRECTION_ROW, 
            m_nCols,
            m_nRows,
            m_descrB,
            AT,
            m_nCols,
            pNnzPerRow,
            nnz_u);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(nnz_h, nnz_u, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    nnz = nnz_h[0];
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSnnz, status: %d\n", status);
        exit(1);
    }
    if (DEBUG) {
        printf("gpu nnz: %d\n", nnz);
    }

    float *pCsrVal;
    int *pCsrRow;
    int *pCsrInd;

    if (DEBUG) {
        checkCudaErrors(cudaMallocManaged((void **)&(pCsrVal), sizeof(float) * nnz));
        checkCudaErrors(cudaMallocManaged((void **)&(pCsrRow), sizeof(int) * (m_nCols+1)));
        checkCudaErrors(cudaMallocManaged((void **)&(pCsrInd), sizeof(int) * nnz));
    } else {
        checkCudaErrors(cudaMalloc((void **)&(pCsrVal), sizeof(float) * nnz));
        checkCudaErrors(cudaMalloc((void **)&(pCsrRow), sizeof(int) * (m_nCols+1)));
        checkCudaErrors(cudaMalloc((void **)&(pCsrInd), sizeof(int) * nnz));
    }

    status = cusparseSdense2csr(m_cusparseHandle, 
            m_nCols, 
            m_nRows, 
            m_descrB,
            AT,
            m_nCols, //leading dimension
            pNnzPerRow,
            pCsrVal,
            pCsrRow,
            pCsrInd);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSdense2csc, status: %d\n", status);
        exit(1);
    }
    if (DEBUG) {
        printf("pCsrVal: \n");
        print_array(pCsrVal, nnz, 1);
        printf("pCsrRow: \n");
        print_array(pCsrRow, m_nCols+1, 1);
        printf("pCsrInd: \n");
        print_array(pCsrInd, nnz, 1);
    }

    status = cusparseScsrmm2(m_cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE,
            m_nCols,
            nRows,
            m_nRows,
            nnz,
            &alpha,
            m_descrB,
            pCsrVal,
            pCsrRow,
            pCsrInd,
            B,
            m_nRows,
            &beta,
            CT,
            m_nCols);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseScsrmm2, status: %d\n", status);
        exit(1);
    }
	cublasSgeam(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, nCols, m_nRows, &alpha, CT, m_nRows, &beta, NULL, m_nRows, C, nCols); 
    cudaDeviceSynchronize();

    cudaFree(AT);
    cudaFree(CT);
    cudaFree(pNnzPerRow);
    cudaFree(pCsrVal);
    cudaFree(pCsrRow);
    cudaFree(pCsrInd);
    cudaDeviceSynchronize();
    free(nnz_h);
    return 0.0;
}

void ShCusparseGemm::build()
{
    //1. Convert self dense matrix to CSR format
    //1.1 Calculate nnz per row
    cusparseStatus_t status;
    checkCudaErrors(cudaMalloc((void **)&(m_pNnzPerRow), sizeof(int) * (m_nCols + 1)));
    checkCudaErrors(cudaMalloc((void **)&(m_nnz_u), sizeof(int) * 1));
    int *nnz_h = (int *)malloc(sizeof(int) * 1);

    status = cusparseSnnz(m_cusparseHandle, 
            CUSPARSE_DIRECTION_ROW, 
            m_nCols,
            m_nRows,
            m_descrB,
            m_pDenseMatrix,
            m_nCols,
            m_pNnzPerRow,
            m_nnz_u);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(nnz_h, m_nnz_u, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    m_nnz = nnz_h[0];
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSnnz, status: %d\n", status);
        exit(1);
    }
    //printf("nnz: %d\n", nnz);

    checkCudaErrors(cudaMalloc((void **)&(m_pCsrVal), sizeof(float) * m_nnz));
    checkCudaErrors(cudaMalloc((void **)&(m_pCsrRow), sizeof(int) * (m_nCols+1)));
    checkCudaErrors(cudaMalloc((void **)&(m_pCsrInd), sizeof(int) * m_nnz));

    status = cusparseSdense2csr(m_cusparseHandle, 
            m_nCols, 
            m_nRows, 
            m_descrB,
            m_pDenseMatrix,
            m_nCols, //leading dimension
            m_pNnzPerRow,
            m_pCsrVal,
            m_pCsrRow,
            m_pCsrInd);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseSdense2csr, status: %d\n", status);
        exit(1);
    }
    free(nnz_h);
    cudaFree(m_pDenseMatrix);
    m_pDenseMatrix = NULL;
}

double ShCusparseGemm::runKernelTest(float *A, int nRows, int nCols, float *C)
{

    float alpha = 1.0;
    float beta = 0.0;
    cusparseStatus_t status;
    status = cusparseScsrmm2(m_cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            m_nCols,
            nRows,
            m_nRows,
            m_nnz,
            &alpha,
            m_descrB,
            m_pCsrVal,
            m_pCsrRow,
            m_pCsrInd,
            A,
            m_nRows,
            &beta,
            C,
            m_nCols);
    cudaDeviceSynchronize();
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Fail by cusparseScsrmm, status: %d\n", status);
        exit(1);
    }
    return 0.0;
}


ShCusparseGemm::~ShCusparseGemm() {
    cusparseDestroy(m_cusparseHandle);
    cusparseDestroyMatDescr(m_descrB);
    cublasDestroy(m_cublasHandle);
    if (m_pCsrVal) {
        cudaFree(m_nnz_u);
        cudaFree(m_pNnzPerRow);
        cudaFree(m_pCsrVal);
        cudaFree(m_pCsrRow);
        cudaFree(m_pCsrInd);
    }

}
