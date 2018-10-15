#include <cub/cub.cuh>
using namespace cub;

template<int BLOCK_SIZE, int CPG>
__global__ void cal_group_coo_format_nnz_kernel_cm(float *A, int nRows, int nCols, int *pNnzPerGroup) 
{
    int startIdx = blockIdx.x * CPG;
    int nnz = 0;
    int nColPerThread = (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int colOffset = threadIdx.x * nColPerThread;
    for (int i = threadIdx.x; i < nCols; i+=BLOCK_SIZE) {
        for (int j = 0; j < CPG; j++) {
            int row = j + startIdx;
            if (row >= nRows) 
                break;
            float v = A[row * nCols + i];
            if (v != 0.0) {
                nnz++;
            }
        }
    }
    typedef BlockReduce<int, BLOCK_SIZE> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    int aggregate = BlockReduceT(temp_storage).Sum(nnz);
    if (threadIdx.x == 0) {
        pNnzPerGroup[blockIdx.x] = aggregate;
    }
}

template<int BLOCK_SIZE, int CPG>
__global__ void convert_to_group_coo_format_kernel_cm(float *A, int nRows, int nCols,
        float *pVals, int *pRows, int *pCols, int *pGroupIndex, int *pNnzPerGroup)
{

    int startIdx = blockIdx.x * CPG;
    int currGroupOffset = pGroupIndex[blockIdx.x];
    int cooIndex = currGroupOffset;
    float *currVals = pVals + cooIndex;
    int *currCols = pCols + cooIndex;
    int *currRows = pRows + cooIndex;

    __shared__ float sA[BLOCK_SIZE * CPG];
    typedef BlockScan<int, BLOCK_SIZE> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    __shared__ int sNNz;
    sNNz = 0;
    __syncthreads();

    int end = (nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE * BLOCK_SIZE;
    for (int i = threadIdx.x; i < end; i+=BLOCK_SIZE) {
        int nnz = 0;
        int nnz_i = 0;
        for (int j = 0; j < CPG; j++) {
            int row = j + startIdx;
            if (row < nRows && i < nCols) {
                float v = A[row * nCols + i];
                sA[j * BLOCK_SIZE + threadIdx.x] = v;
                if (v != 0.0) 
                    nnz++;
            } 
        }
        BlockScanT(temp_storage).InclusiveSum(nnz, nnz_i);
        __syncthreads();
        BlockScanT(temp_storage).ExclusiveSum(nnz, nnz);

        float *vals = currVals + nnz;
        int *cols = currCols + nnz;
        int *rows = currRows + nnz;

        for (int j = 0; j < CPG; j++) {
            int row = j + startIdx;
            if (row >= nRows || i >= nCols) 
                break;
            float v = sA[j * BLOCK_SIZE + threadIdx.x];
            if (v != 0.0)  {
                *(vals++) = v;
                *(rows++) = row;
                *(cols++) = i;
            }
        }
        if (threadIdx.x == BLOCK_SIZE - 1) {
            sNNz = nnz_i;
        }
        __syncthreads();
        currVals += sNNz;
        currCols += sNNz;
        currRows += sNNz;
    }
}

template<int BLOCK_SIZE, int CPG>
__global__ void sparse_dense_groupcoo_mat_mul_kernel(float *vals_A, int *cols_A, int *rows_A, int *groupIndex_A, int *nnzPerGroup_A, int wA, int hA, float *B, int wB, int hB, float *C)
{
    int Cj = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int Ci0 = blockIdx.x * CPG;
    float c0 =0.0; float c1 =0.0; float c2 =0.0; float c3 =0.0; 
    int groupIdxOfCurrentBlock = groupIndex_A[blockIdx.x];
    int nnz = nnzPerGroup_A[blockIdx.x];
    float *currValsA = vals_A + groupIdxOfCurrentBlock;
    int *currColsA = cols_A + groupIdxOfCurrentBlock;
    int *currRowsA = rows_A + groupIdxOfCurrentBlock;

    __shared__ float sValsA[BLOCK_SIZE]; 
    __shared__ int sRowsA[BLOCK_SIZE]; 
    __shared__ int sColsA[BLOCK_SIZE]; 
    __shared__ int sNNz[1];

    int nIter = (BLOCK_SIZE + nnz - 1) / BLOCK_SIZE;
    int extra = nnz & (BLOCK_SIZE - 1);
    for (int i = 0; i < nIter; i++) {
        sColsA[threadIdx.x] = -1;
        sValsA[threadIdx.x] = 0.0;
        sNNz[0] = BLOCK_SIZE;
        __syncthreads();

        int valIdxStart = i * BLOCK_SIZE;
        int valIdx = valIdxStart + threadIdx.x;
        if (valIdx < nnz) {
            sValsA[threadIdx.x] = currValsA[valIdx];
            sRowsA[threadIdx.x] = currRowsA[valIdx];
            sColsA[threadIdx.x] = currColsA[valIdx];
        } else {
            sNNz[0] = extra;
        }
        __syncthreads();

        if (Cj < wB) {
            int k = 1;
            int rNNz = sNNz[0];
            int precol = -1;
            float b;
            for (int j = 0; j < rNNz;) {
                int col = sColsA[j];
                if (col != precol) {
                    b = B[col * wB + Cj];
                    precol = col;
                }
                float a = sValsA[j];
                int currRow = sRowsA[j];
                int index = currRow & (CPG-1);
                if (index == 0) c0 = fmaf(a,b,c0); else if (index == 1) c1=fmaf(a,b,c1); else if (index == 2) c2=fmaf(a,b,c2); else if (index == 3) c3=fmaf(a,b,c3);
                j++;
            }
        }
        __syncthreads();
    }
    if (Cj < wB) {
        if (Ci0 < hA) 
            C[Cj + Ci0 * wB] = c0; 
        if (Ci0+1 < hA) 
            C[Cj  + (Ci0 + 1)*wB] = c1; 
        if (Ci0+2 < hA) 
            C[Cj  + (Ci0 + 2)*wB] = c2; 
        if (Ci0+3 < hA) 
            C[Cj  + (Ci0 + 3)*wB] = c3; 
    }
}

__global__ void prefix_sum_kernel2(int *src, int *dst, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = 0;
        for (int i = 1; i < n; i++) {
            dst[i] = dst[i-1] + src[i-1];
        }
    }
}


