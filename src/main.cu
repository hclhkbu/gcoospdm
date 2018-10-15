#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "utils.h"
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "SpGemmBuilder.h"
#include "MatrixReader.h"
#include "constants.h"

using namespace std;

bool g_bCheckResult = false;

void setArgumentInt(int argc, char **argv, const char *string_ref, size_t &target) 
{
    if (checkCmdLineFlag(argc, (const char **)argv, string_ref)) {
        target = getCmdLineArgumentInt(argc, (const char **)argv, string_ref);
    }
}
void setArgumentFloat(int argc, char **argv, const char *string_ref, float &target) 
{
    if (checkCmdLineFlag(argc, (const char **)argv, string_ref)) {
        target = getCmdLineArgumentFloat(argc, (const char **)argv, string_ref);
    }
}

void benchSpGemm(float *A, size_t nRowsA, size_t nColsA, float *B, size_t nRowsB, size_t nColsB, float *C, string algo, bool kernelonly)
{
    printf("A[%d,%d] * B[%d,%d]\n", nRowsA, nColsA, nRowsB, nColsB);
    printf("Algorithm: %s\n", algo.c_str());
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&(d_A), (size_t)nRowsA * nColsA * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&(d_B), (size_t)nRowsB * nColsB * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&(d_C), (size_t)nRowsA * nColsB * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_A, A, (size_t)nRowsA * nColsA * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, (size_t)nRowsB * nColsB * sizeof(float), cudaMemcpyHostToDevice));

    //ShGemm *pGemmAlgo = SpGemmBuilder::getShGemmAlgo(d_B, nRowsB, nColsB, "cusparse");
    ShGemm *pGemmAlgo = SpGemmBuilder::getShGemmAlgo(d_B, nRowsB, nColsB, algo);
    if (DEBUG)
        print_array(B, nColsB, nRowsB);

    //auto start = std::chrono::steady_clock::now();
    int numRepeats = 100;
    if (nRowsA > 8000) {
        numRepeats = 20;
    }
    if (DEBUG) {
        numRepeats = 0;
    }
    if (kernelonly) {
        pGemmAlgo->build();
    }
    // warmup
    if (kernelonly) {
        pGemmAlgo->runKernelTest(d_A, nRowsA, nColsA, d_C);
    } else {
        pGemmAlgo->mutiplyBy(d_A, nRowsA, nColsA, d_C);
    }
    //start = std::chrono::steady_clock::now();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < numRepeats; ++i) {
        if (kernelonly) {
            pGemmAlgo->runKernelTest(d_A, nRowsA, nColsA, d_C);
        } else {
            pGemmAlgo->mutiplyBy(d_A, nRowsA, nColsA, d_C);
        //pGemmAlgo->mutiply(d_A, nRowsA, nColsA, d_C);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    msecTotal /= numRepeats;
    auto end = std::chrono::steady_clock::now();
    //int timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

    checkCudaErrors(cudaMemcpy(C, d_C, nRowsA * nColsB * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    //printf("Time:%f\n", msecTotal);
    printf("Time:%f\n", msecTotal*1000);
    delete pGemmAlgo;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


void timeGemm(size_t nRowsA, size_t nColsA,
        size_t nRowsB, size_t nColsB, float sparsity,
        std::string algo, bool kernelonly)
{

    float *A = (float *)malloc(nRowsA * nColsA * sizeof(float)); // construct A 
    float *B = (float *)malloc(nRowsB * nColsB * sizeof(float)); // construct B 
    float *C = (float *)malloc(nRowsA * nColsB* sizeof(float)); // construct C
    float *reference = (float *)malloc(nRowsA * nColsB* sizeof(float)); // construct C

    randomInit(A, nRowsA * nColsA);
    if (DEBUG) {
        printf("A: \n");
        print_array(A, nColsA, nRowsA);
    }
    int nnz = randomInit(B, nRowsB * nColsB, sparsity);
    if (DEBUG) {
        printf("B: \n");
        print_array(B, nColsB, nRowsB);
    }
    printf("nnz cpu: %d, density: %.4f\n", nnz, 
            nnz*1.0/(nRowsB*nColsB));

    benchSpGemm(A, nRowsA, nColsA, B, nRowsB, nColsB, C, algo, kernelonly);
    if (DEBUG) {
        printf("C: \n");
        print_array(C, nColsB, nRowsA);
    }


    bool bCheckResult = g_bCheckResult;
    if (bCheckResult) {
        //matrixMulCPU(reference, A, B, nRowsA, nColsA, nColsB);
        matrixMulCPU(reference, B, A, nRowsA, nColsA, nColsB);
        if (DEBUG) {
            printf("Reference: \n");
            print_array(reference, nColsB, nRowsA);
        }
        bool resGPU = sdkCompareL2fe(reference, C, nRowsA * nColsB, 1.0e-5f);
        if (resGPU != true) {
            printDiff(reference, C, nColsB, nRowsA, 100, 1.0e-5f);
        } else {
            printf("Test passed\n");
        }
    }
    free(A);
    free(B);
    free(C);
    free(reference);
}

void timeFromFile(const char *filename, string algo, bool kernelonly)
{
    MatrixReader *pMatrixReader = new MatrixReader(filename);
    printf("nnz cpu: %d, density: %.4f\n", pMatrixReader->m_nnz, 
            pMatrixReader->m_nnz*1.0/(pMatrixReader->m_nRows*pMatrixReader->m_nCols));
    size_t nRowsA = pMatrixReader->m_nRows;
    size_t nColsA = pMatrixReader->m_nCols;
    size_t nRowsB = nRowsA;
    size_t nColsB = nColsA;
    float *A = pMatrixReader->m_pDenseMatrix;
    float *B = pMatrixReader->m_pDenseMatrix;
    float *C = (float *)malloc(nRowsA * nColsB* sizeof(float)); // construct C
    float *reference = (float *)malloc(nRowsA * nColsB* sizeof(float)); // construct C

    benchSpGemm(A, nRowsA, nColsA, B, nRowsB, nColsB, C, algo, kernelonly);

    bool bCheckResult = g_bCheckResult;
    if (bCheckResult) {
        matrixMulCPU(reference, B, A, nRowsA, nColsA, nColsB);
        bool resGPU = sdkCompareL2fe(reference, C, nRowsA * nColsB, 1.0e-5f);
        if (resGPU != true) {
            printDiff(reference, C, nColsB, nRowsA, 100, 1.0e-5f);
        } else {
            printf("Test passed\n");
        }
    }

    delete pMatrixReader;
    free(C);
}

int main(int argc, char **argv) {
    printf("[Benchmark sparse gemm on GPU (Dense A multiply sparse B] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage -nRowsA=nRowsA\n");
        printf("      -nColsA=nColsA\n");
        printf("      -nRowsB=nRowsB\n");
        printf("      -nColsB=nColsB\n");
        printf("      -nColsB=nColsB\n");
        printf("      -file=file (Sparse matrix, if specified, then C=A * A)\n");
        printf("      -sparsity=sparsity (Sparsity of matrix B\n");
        printf("      -algo=algo (Current supports: cusparse, groupspgemm)'\n");
        printf("      -kernelonly=0 (Benchmark kernel only)'\n");
        printf("nColsA should equal to nRowsB    \n");
        exit(EXIT_SUCCESS);
    }
    size_t nRowsA = 1024; 
    size_t nColsA = 1024; 
    size_t nRowsB = 1024; 
    size_t nColsB = 1024; 
    float sparsity = 0.5;
    size_t kernelonly = 0;
    setArgumentInt(argc, argv, "nRowsA", nRowsA);
    setArgumentInt(argc, argv, "nColsA", nColsA);
    setArgumentInt(argc, argv, "nRowsB", nRowsB);
    setArgumentInt(argc, argv, "nColsB", nColsB);
    setArgumentInt(argc, argv, "kernelonly", kernelonly);
    setArgumentFloat(argc, argv, "sparsity", sparsity);
    char *value = 0; 
    getCmdLineArgumentString(argc, (const char **)argv, "algo", &value);
    std::string algo(value);

    value = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &value);
    if (value) {
        std::string filename(value);
        timeFromFile(filename.c_str(), algo, kernelonly==1);
    } else {
        timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, algo, kernelonly==1);
    }
    return 0;
}
