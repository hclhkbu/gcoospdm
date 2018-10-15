#include "utils.h"
#include <math.h>

// Allocates a matrix with random float entries.
int randomInit(float *data, size_t size, float sparsity/*=0.0*/)
{
    const int num = 1000000;
    int nnz = 0;
    for (int i = 0; i < size; ++i) {
        int r = rand();
        if (r % num < sparsity * num) {
            data[i] = 0.0;
        } else {
            data[i] = r / (float)RAND_MAX;
            nnz++;
        }
    }
    return nnz;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void print_array(float *data, size_t w, size_t h)
{
    printf("[");
    for (int i = 0; i < h; i++) {
        printf("[");
        for (int j = 0; j < w; j++) {
            if (j == w - 1) {
                printf("%.3f", data[j + i * w]);
            } else {
                printf("%.3f,", data[j + i * w]);
            }
        }
        if (i == h - 1) {
            printf("]");
        } else {
            printf("],\n");
        }
    }
    printf("]\n\n");
}

void print_array(int *data, size_t w, size_t h)
{
    printf("[");
    for (int i = 0; i < h; i++) {
        printf("[");
        for (int j = 0; j < w; j++) {
            if (j == w - 1) {
                printf("%d", data[j + i * w]);
            } else {
                printf("%d,", data[j + i * w]);
            }
        }
        if (i == h - 1) {
            printf("]");
        } else {
            printf("],\n");
        }
    }
    printf("]\n\n");
}
