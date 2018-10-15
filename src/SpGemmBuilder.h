#include "ShGemm.h"
#include "ShCusparseGemm.h"
#include "ShGCOOSpDM.h"
#include "ShCublasGemm.h"
#include "ShCsrSpGemm.h"

using namespace std;

class SpGemmBuilder
{
    public:
        static ShGemm* getShGemmAlgo(float *denseMatrix, int nRows, int nCols, const std::string &desc) {
            if (desc == "cusparse") {
                return new ShCusparseGemm(denseMatrix, nRows, nCols);
            } else if (desc == "gcoospdm") {
                return new ShGCOOSpDM(denseMatrix, nRows, nCols);
            } else if (desc == "cublas") {
                return new ShCublasGemm(denseMatrix, nRows, nCols);
            } else if (desc == "csr") {
                return new ShCsrSpGemm(denseMatrix, nRows, nCols);
            }
            return NULL;
        }
};
