#ifndef __SHCUBLASGEMM_H__
#define __SHCUBLASGEMM_H__

#include "ShGemm.h"
#include <cublas_v2.h>

class ShCublasGemm: public ShGemm
{
    public:
        ShCublasGemm(float *denseMatrix, int nRows, int nCols);
        ~ShCublasGemm(void);
        double mutiplyBy(float *A, int nRows, int nCols, float *C);
        double mutiply(float *B, int nRows, int nCols, float *C);
        void build() {}
        double runKernelTest(float *A, int nRows, int nCols, float *C) {}
    public:
        cublasHandle_t m_cublasHandle;
};

#endif
