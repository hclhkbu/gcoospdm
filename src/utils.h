#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdlib.h>
#include <stdio.h>

int randomInit(float *data, size_t size, float sparsity=0.0);
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB);
void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol);
void print_array(float *data, size_t w, size_t h);
void print_array(int *data, size_t w, size_t h);

#endif
