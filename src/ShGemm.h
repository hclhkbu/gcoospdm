#ifndef __SHGEMM_H__
#define __SHGEMM_H__ 

#include <iostream>
#include "constants.h"

class ShGemm
{
    public:
        ShGemm() {}
        ShGemm(int nRows, int nCols) : m_nRows(nRows), m_nCols(nCols) {}
        ~ShGemm(void) {}
        // C = A * self, self is the matrix that defined in this class, it should be a sparse format for spgemm.
        virtual double mutiplyBy(float *A, int nRows, int nCols, float *C) = 0;
        // C = self * B, self is a sparse matrix
        virtual double mutiply(float *B, int nRows, int nCols, float *C) = 0;
        virtual void build() = 0;
        virtual double runKernelTest(float *A, int nRows, int nCols, float *C) = 0;

    public:
        int m_nRows;
        int m_nCols;
        float *m_pDenseMatrix;
};

#endif
