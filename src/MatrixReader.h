#ifndef __MATRIXREADER_H__
#define __MATRIXREADER_H__

class MatrixReader
{
    public:
        MatrixReader() {};
        MatrixReader(const char *filename);
        void testGroupNNz();
        ~MatrixReader(void);
    public:
        float *m_pDenseMatrix;
        int m_nRows;
        int m_nCols;
        int m_nnz;
};

#endif
