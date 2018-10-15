# GCOOSpDM
Efficient sparse-dense matrix-matrix multiplication on GPUs using the customized sparse storage format

## Quick Install
```
git clone https://github.com/hclhkbu/gcoospdm.git 
cd gcoospdm
git submodule update --init --recursive
cd src
make
```

## Example 
After sucessfully compiled, one can try the following example:
```
./main -sparsity=0.900000 -nRowsA=800 -nRowsB=800 -nColsA=800 -nColsB=800 -algo=gcoospdm
```
