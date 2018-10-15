#ifndef __SHCUSPARSEGEMM_H__
#define __SHCUSPARSEGEMM_H__

#include "ShGemm.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>

class ShCusparseGemm: public ShGemm
{
    public:
        ShCusparseGemm(float *denseMatrix, int nRows, int nCols);
        ~ShCusparseGemm(void);
        double mutiplyBy(float *A, int nRows, int nCols, float *C);
        double mutiply(float *B, int nRows, int nCols, float *C);
        double runKernelTest(float *A, int nRows, int nCols, float *C);
        void build();
    public:
        cusparseHandle_t m_cusparseHandle;
        cusparseMatDescr_t m_descrB;
        cublasHandle_t m_cublasHandle;

        float *m_pCsrVal;
        int *m_pCsrRow;
        int *m_pCsrInd;
        int *m_pNnzPerRow;
        int *m_nnz_u;
        int m_nnz;
};

#endif
