#ifndef __SHCSRSPGEMM_H__
#define __SHCSRSPGEMM_H__

#include "ShGemm.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>


class ShCsrSpGemm: public ShGemm
{
    public:
        ShCsrSpGemm(float *denseMatrix, int nRows, int nCols);
        ~ShCsrSpGemm(void);

        double mutiplyBy(float *A, int nRows, int nCols, float *C);
        double mutiply(float *B, int nRows, int nCols, float *C){}
        void build() {}
        double runKernelTest(float *A, int nRows, int nCols, float *C) {}
    public:
        float *m_pDenseMatrix;

        cublasHandle_t m_cublasHandle;
        cusparseHandle_t m_cusparseHandle;
        cusparseMatDescr_t m_descrB;
};

#endif
