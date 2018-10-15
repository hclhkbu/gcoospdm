#ifndef __SHGCOOSPDM_H__
#define __SHGCOOSPDM_H__

#include "ShGemm.h"
#include <cublas_v2.h>

class ShGCOOSpDM: public ShGemm
{
    public:
        ShGCOOSpDM(float *denseMatrix, int nRows, int nCols);
        ~ShGCOOSpDM(void);
        void convertToGroupCOOFormat(float *denseMatrix, int nRows, int nCols, 
                float* &pVals,
                int* &pCols,
                int* &pRows,
                int* &pGroupIndex,
                int* &pNnzPerGroup,
                int nGroup);

        double mutiplyBy(float *A, int nRows, int nCols, float *C);
        double mutiply(float *B, int nRows, int nCols, float *C){}
        void build();
        double runKernelTest(float *A, int nRows, int nCols, float *C);
    public:
        float *m_pDenseMatrix;

        float *m_pVals;
        int *m_pCols;
        int *m_pRows;
        int *m_pGroupIndex;
        int *m_pNnzPerGroup;
        int m_nGroup;
};

#endif
