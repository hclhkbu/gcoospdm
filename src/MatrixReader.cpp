#include "MatrixReader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils.h"
#include "constants.h"

#define LINE_LENGTH_MAX 256

MatrixReader::MatrixReader(const char *filename):
    m_pDenseMatrix(NULL), m_nRows(0), m_nCols(0), m_nnz(0)
{
    FILE *fp;
    fp = fopen(filename, "r");
    if(fp == NULL) {
        printf("Cannot find file: %s\n", filename);
        exit(1);
    }
    printf("Read mtx file: %s\n", filename);
    char *line, *ch;
    line = (char *)malloc(sizeof(char) * LINE_LENGTH_MAX);
    fgets(line, LINE_LENGTH_MAX, fp);
    int isUnsy = 0;
    if (strstr(line, "general")) {
        isUnsy = 1;
    }
    do {
        fgets(line, LINE_LENGTH_MAX, fp);
    } while(line[0] == '%');
  
    /* Get size info */
    int nRows, nCols, nnz;
    sscanf(line, "%d %d %d", &nRows, &nCols, &nnz);
    m_nRows = nRows;
    m_nCols = nCols;

    // Allocate memory
    m_pDenseMatrix = (float *)malloc(sizeof(float) * nRows * nCols);

    float *pMatrix = m_pDenseMatrix;
    for (int i = 0; i < nRows * nCols; i++) {
        *(pMatrix++) = 0.0;
    }
    int num = 0;

    while (fgets(line, LINE_LENGTH_MAX, fp)) {
        ch = line;
        /* Read first word (row id)*/
        int row = (int)(atoi(ch) - 1);
        ch = strchr(ch, ' ');
        ch++;
        /* Read second word (column id)*/
        int col = (int)(atoi(ch) - 1);
        ch = strchr(ch, ' ');
        float val = rand();
        if (ch != NULL) {
            ch++;
            /* Read third word (value data)*/
            val = (float)atof(ch);
            m_pDenseMatrix[row * nCols + col] = val;
            m_nnz++;
            ch = strchr(ch, ' ');
        } else {
            m_pDenseMatrix[row * nCols + col] = val; 
            m_nnz++;
        }
        if (!isUnsy) {
            m_pDenseMatrix[col * nCols + row] = val; 
            if (col != row) {
                m_nnz++;
            }
        }
        num++;
    }
    fclose(fp);
    free(line);
    if (DEBUG) {
        testGroupNNz();
    }

    //printf("nRows: %d, nCols: %d\n", m_nRows, m_nCols);
    //print_array(m_pDenseMatrix, m_nCols, m_nRows);
}

void MatrixReader::testGroupNNz()
{
    int nGroup = (m_nCols + COLUMN_PER_GROUP - 1) / COLUMN_PER_GROUP;
    int *pNnzPerGroup = (int *)calloc(nGroup, sizeof(int));
    for (int i = 0; i < m_nCols; i++) {
        int idx = i / COLUMN_PER_GROUP;
        for (int j = 0; j < m_nRows; j++) {
            if (m_pDenseMatrix[j * m_nCols + i] != 0) 
                pNnzPerGroup[idx]++;
        }
    }
    if (DEBUG) {
        printf("pNnzPerGroup: %d groups\n", nGroup);
        print_array(pNnzPerGroup, nGroup, 1);
    }
    free(pNnzPerGroup);
}

MatrixReader::~MatrixReader(void) {
    if (m_pDenseMatrix) {
        free(m_pDenseMatrix);
    }
}
