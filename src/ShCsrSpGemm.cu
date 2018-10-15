#include "ShCsrSpGemm.h"
#include "constants.h"

#include <helper_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


template<int BLOCK_SIZE>
__global__ void column_major_dense_sparse_csc_mat_mul_kernel_v2(float *A, int wA, int hA, float *Bcsc, int *BJ, int *BI, int wB, int hB, float *C)
{
    int Ci = blockIdx.x;
    int Cj = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    int valIdx = BI[Ci];
    int valEnd = BI[Ci+1];
    int rowB;
    float bVal;
    float c = 0.0;
    __shared__ int sBJ[BLOCK_SIZE];
    __shared__ float sBcsc[BLOCK_SIZE];
    for (int k = valIdx; k < valEnd; k += BLOCK_SIZE){
	    int inputIdx = threadIdx.x + k;
	    int actualEnd = k + BLOCK_SIZE > valEnd ? (valEnd - k) : BLOCK_SIZE;
	    if (inputIdx < valEnd) {
		    sBJ[threadIdx.x] = BJ[inputIdx];
		    sBcsc[threadIdx.x] = Bcsc[inputIdx];
        }
        __syncthreads();

	    if (Ci < wB && Cj < wA) {
		    for (int i = 0; i < actualEnd; i++) {
			    rowB = sBJ[i];
			    bVal = sBcsc[i];
			    c += A[rowB * wA + Cj] * bVal;
		    }
            C[Ci * wA + Cj] = c;
	    }
        __syncthreads();
    }
}


ShCsrSpGemm::ShCsrSpGemm(float *denseMatrix, int nRows, int nCols):
    ShGemm(nRows, nCols)
{
    m_pDenseMatrix = denseMatrix;

    cusparseCreate(&m_cusparseHandle);
    cublasCreate(&m_cublasHandle);

    cusparseCreateMatDescr(&m_descrB);
    cusparseSetMatType(m_descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(m_descrB, CUSPARSE_INDEX_BASE_ZERO);
}


double ShCsrSpGemm::mutiplyBy(float *A, int nRows, int nCols, float *C) 
{

    //1.1 Calculate nnz per row
    int *pNnzPerRow;
    int *nnz_u;
    int nnz;
    checkCudaErrors(cudaMallocManaged((void **)&(pNnzPerRow), sizeof(int) * (m_nCols + 1)));
    checkCudaErrors(cudaMallocManaged((void **)&(nnz_u), sizeof(int) * 1));
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

    checkCudaErrors(cudaMallocManaged((void **)&(pCsrVal), sizeof(float) * nnz));
    checkCudaErrors(cudaMallocManaged((void **)&(pCsrRow), sizeof(int) * (m_nCols+1)));
    checkCudaErrors(cudaMallocManaged((void **)&(pCsrInd), sizeof(int) * nnz));

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

    float *AT;
    float alpha = 1.0;
    float beta = 0.0;

    cublasStatus_t status_b;
    checkCudaErrors(cudaMallocManaged((void **) &AT, sizeof(float) * nRows * nCols));
	status_b = cublasSgeam(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, nRows, nCols, &alpha, A, nCols, &beta, NULL, nCols, AT, nRows); 
    if (status_b != CUBLAS_STATUS_SUCCESS) {
        printf("Fail by cublasSgemm in transpose A, status: %d\n", status);
        exit(1);
    }
    cudaDeviceSynchronize();

    float *CT;
    checkCudaErrors(cudaMallocManaged((void **) &CT, nRows * m_nCols * sizeof(float)));

	dim3 grid(m_nCols, (nRows+BS-1)/BS);
    dim3 threads(BS);
    column_major_dense_sparse_csc_mat_mul_kernel_v2<BS><<<grid, threads>>>(AT, nRows, nCols, pCsrVal, pCsrRow, pCsrInd, m_nCols, nCols, CT);

    status_b = cublasSgeam(m_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m_nCols, nRows, &alpha, CT, nRows, &beta, NULL, nRows, C, m_nCols);
    if (status_b != CUBLAS_STATUS_SUCCESS) {
        printf("Fail by cublasSgemm in transpose C, status: %d\n", status);
        exit(1);
    }

    cudaFree(AT);
    cudaFree(CT);
    cudaFree(pNnzPerRow);
    cudaFree(pCsrVal);
    cudaFree(pCsrRow);
    cudaFree(pCsrInd);

    return 0.0;
}

ShCsrSpGemm::~ShCsrSpGemm(void)
{
    cusparseDestroy(m_cusparseHandle);
    cusparseDestroyMatDescr(m_descrB);
    cublasDestroy(m_cublasHandle);
}
