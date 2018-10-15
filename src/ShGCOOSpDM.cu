#include "ShGCOOSpDM.h" 
#include "constants.h"
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "utils.h"
#include <cub/cub.cuh>
#include <chrono>
#include "gcoo_spdm_kernels.cu"
using namespace std;


ShGCOOSpDM::ShGCOOSpDM(float *denseMatrix, int nRows, int nCols):
    ShGemm(nRows, nCols)
{
    m_pDenseMatrix = denseMatrix;
}

void ShGCOOSpDM::convertToGroupCOOFormat(float *denseMatrix, int nRows, int nCols, 
                float* &pVals,
                int* &pCols,
                int* &pRows,
                int* &pGroupIndex,
                int* &pNnzPerGroup,
                int nGroup)
{
    //printf("nGroup: %d\n", nGroup);
    auto start = std::chrono::steady_clock::now();
    if (DEBUG) {
        checkCudaErrors(cudaMallocManaged((void **) &pGroupIndex, sizeof(int) * (nGroup+1)));
        checkCudaErrors(cudaMallocManaged((void **) &pNnzPerGroup, sizeof(int) * (nGroup+1)));
    } else {
        checkCudaErrors(cudaMalloc((void **) &pGroupIndex, sizeof(int) * (nGroup+1)*2));
        //checkCudaErrors(cudaMalloc((void **) &pNnzPerGroup, sizeof(int) * (nGroup+1)));
        pNnzPerGroup = pGroupIndex + (nGroup+1);
    }
    cudaMemset(pNnzPerGroup, 0, (nGroup+1) * sizeof(int));
    auto end = std::chrono::steady_clock::now();
    int timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    printf("MC:%d\n", timeUsed);

    start = std::chrono::steady_clock::now();

    const int BS_OF_CAL = 512;
    dim3 gridCal(nGroup);
    dim3 tbCal(BS_OF_CAL);
    cal_group_coo_format_nnz_kernel_cm<BS_OF_CAL, COLUMN_PER_GROUP><<<gridCal, tbCal>>>(
            denseMatrix,
            nRows, 
            nCols,
            pNnzPerGroup);

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    printf("D2S:%d\n", timeUsed);


    start = std::chrono::steady_clock::now();
    if (DEBUG) {
        printf("gpu pNnzPerGroup:\n");
        print_array(pNnzPerGroup, nGroup, 1);
    }
    void     *d_temp_storage = NULL;
    prefix_sum_kernel2<<<1,1>>>(pNnzPerGroup, pGroupIndex, nGroup+1);
    //size_t   temp_storage_bytes = 0;
    //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, pNnzPerGroup, pGroupIndex, nGroup+1);
    //// Allocate temporary storage
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //// Run exclusive prefix sum
    //cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, pNnzPerGroup, pGroupIndex, nGroup+1);
    cudaDeviceSynchronize();
    if (DEBUG) {
        printf("gpu pGroupIndex:\n");
        print_array(pGroupIndex, nGroup, 1);
    }
    end = std::chrono::steady_clock::now();
    timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    //printf("ExclusiveSum:%d\n", timeUsed);

    start = std::chrono::steady_clock::now();

    int *nnz_h = (int *)malloc(sizeof(int) * 1);
    checkCudaErrors(cudaMemcpy(nnz_h, pGroupIndex+nGroup, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    int nnz = nnz_h[0];
    if (DEBUG)
        printf("nnz: %d\n", nnz);

    if (DEBUG) {
        checkCudaErrors(cudaMallocManaged((void **) &pVals, sizeof(float) * nnz));
        checkCudaErrors(cudaMallocManaged((void **) &pRows, sizeof(int) * nnz));
        checkCudaErrors(cudaMallocManaged((void **) &pCols, sizeof(int) * nnz));
    } else {
        checkCudaErrors(cudaMalloc((void **) &pVals, sizeof(float) * nnz));
        checkCudaErrors(cudaMalloc((void **) &pRows, sizeof(int) * nnz*2));
        //checkCudaErrors(cudaMalloc((void **) &pCols, sizeof(int) * nnz));
        pCols = pRows + nnz;
    }
    end = std::chrono::steady_clock::now();
    timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    //printf("cudaMalloc2:%d\n", timeUsed);

    convert_to_group_coo_format_kernel_cm<BS_OF_CAL, COLUMN_PER_GROUP><<<gridCal, tbCal>>>(
            denseMatrix,
            nRows,
            nCols,
            pVals,
            pRows,
            pCols,
            pGroupIndex,
            pNnzPerGroup);

    cudaDeviceSynchronize();
    if (DEBUG) {
        printf("vals:\n");
        print_array(pVals, nnz, 1);
        printf("cols:\n");
        print_array(pCols, nnz, 1);
        printf("rows:\n");
        print_array(pRows, nnz, 1);
    }

    free(nnz_h);
    if (d_temp_storage)
        cudaFree(d_temp_storage);
}

/*
* Do sparse A multiplies dense B, consider A and B are column majar matrices, which results in column majar C. Similar to cusparse.
*/
double ShGCOOSpDM::mutiplyBy(float *A, int nRows, int nCols, float *C) 
{
    float *pVals;
    int *pCols;
    int *pRows;
    int *pGroupIndex;
    int *pNnzPerGroup;
    int nGroup = (m_nRows + COLUMN_PER_GROUP - 1) / COLUMN_PER_GROUP; // The last group may includes less `COLUMN_PER_GROUP`
    auto start = std::chrono::steady_clock::now();
    convertToGroupCOOFormat(m_pDenseMatrix, m_nRows, m_nCols,
            pVals,
            pCols,
            pRows,
            pGroupIndex,
            pNnzPerGroup,
            nGroup);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    int timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    //printf("EO:%d\n", timeUsed);
    
    start = std::chrono::steady_clock::now();
    dim3 grid(nGroup, (nCols+BS-1)/BS);
    dim3 threadBlock(BS);
    sparse_dense_groupcoo_mat_mul_kernel<BS, COLUMN_PER_GROUP><<<grid, threadBlock>>>(
            pVals,
            pCols,
            pRows,
            pGroupIndex,
            pNnzPerGroup,
            m_nCols,
            m_nRows,
            A,
            nCols,
            nRows,
            C 
            );
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    //printf("KC:%d\n", timeUsed);


    cudaFree(pVals);
    cudaFree(pCols);
    //cudaFree(pRows);
    //cudaFree(pNnzPerGroup);
    cudaFree(pGroupIndex);
    return 0.0;
}

void ShGCOOSpDM::build() 
{
    m_nGroup = (m_nRows + COLUMN_PER_GROUP - 1) / COLUMN_PER_GROUP; // The last group may includes less `COLUMN_PER_GROUP`
    convertToGroupCOOFormat(m_pDenseMatrix, m_nRows, m_nCols,
            m_pVals,
            m_pCols,
            m_pRows,
            m_pGroupIndex,
            m_pNnzPerGroup,
            m_nGroup);
    cudaDeviceSynchronize();
}

double ShGCOOSpDM::runKernelTest(float *A, int nRows, int nCols, float *C) 
{
    dim3 grid(m_nGroup, (nCols+BS-1)/BS);
    dim3 threadBlock(BS);
    sparse_dense_groupcoo_mat_mul_kernel<BS, COLUMN_PER_GROUP><<<grid, threadBlock>>>(
            m_pVals,
            m_pCols,
            m_pRows,
            m_pGroupIndex,
            m_pNnzPerGroup,
            m_nCols,
            m_nRows,
            A,
            nCols,
            nRows,
            C 
            );
    cudaDeviceSynchronize();
    return 0.0;
}

ShGCOOSpDM::~ShGCOOSpDM(void)
{
}
