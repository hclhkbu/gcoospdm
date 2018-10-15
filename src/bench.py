import numpy as np
import os
import fnmatch
from os import listdir
from os.path import isdir
from perf_compare import execute, execute_breakdown

CONTINUE=False
KERNEL_ONLY=False
DEVICE='titanx'

def _get_cmd(n, sparsity, algo):
    """
    n: the width and height of matrix
    sparsity: sparsity of matrix
    """
    cmd='./main -sparsity=%f -nRowsA=%d -nRowsB=%d -nColsA=%d -nColsB=%d -algo=%s' % (sparsity, n, n ,n, n, algo)
    if KERNEL_ONLY:
        cmd = '%s -kernelonly=1' % cmd
    return cmd

def _get_cmd_with_file(fn, algo, kernelonly=False):
    """
    fn: file name of matrix
    """
    cmd='./main -file=%s -algo=%s' % (fn, algo)
    if kernelonly:
        cmd = '%s -kernelonly=1'%cmd
    if algo == 'mergepath' or algo == 'fixedrow4' or algo == 'fixedrow3':
        cmd = '/home/shshi/downloads/merge-spmm/bin/gbspmm --iter 100 --mode %s %s' % (algo, fn)
    return cmd


def _get_matrix_size(fn):
    with open(fn, 'r') as f:
        for line in f.readlines():
            if line.find('%') < 0:
                items = line.split(' ')
                vals = [int(i) for i in items]
                return vals


def _get_all_files(path):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.mtx'):
            matches.append(os.path.join(root, filename))
    print len(matches)
    return matches


def bench():
    result_file_name='result.syth.%s.log' % DEVICE
    if KERNEL_ONLY:
        result_file_name = 'kernel.%s'%result_file_name
    start_n = 8000
    start_sparsity = 0.9
    if CONTINUE:
        with open(result_file_name, 'r') as f:
            content = f.readlines()
            line = content[-1]
            items = line.split('\t')
            print items
            start_n = int(items[0])
            start_sparsity = float(items[1])
    algos = ['gcoospdm', 'cusparse', 'cublas']
    sparsities = np.arange(start_sparsity, 0.9999999, 0.005)
    sparsities = np.concatenate((sparsities, np.arange(0.9955, 0.99999, 0.0005)))
    original_sparsities = np.arange(0.8, 0.9999999, 0.005)
    original_sparsities = np.concatenate((original_sparsities, np.arange(0.9955, 0.99999, 0.0005)))
    print sparsities 
    ns = range(start_n,30000,100)
    bufsize = 20
    f = open(result_file_name, 'a', bufsize)
    if not CONTINUE:
        f.write('N\tsparsity\tgcoospdm\tcusparse\tcublas\n')
    for n in ns:
        for s in sparsities:
            speeds = []
            for algo in algos:
                cmd = _get_cmd(n, s, algo)
                try:
                    ms = execute(cmd)
                except:
                    ms = 10000000.0
                speeds.append(ms)
            f.write('%d\t%.4f\t%.6f\t%.6f\t%.6f\n'%(n,s,speeds[0],speeds[1],speeds[2]))
        sparsities = original_sparsities
    f.close()


def bench_public_dataset():
    OUTPUT_PATH='/home/shshi/gpuhome/data/gemm/output'
    result_file_name='result.real.%s.raw' % DEVICE
    algos = ['gcoospdm', 'cusparse', 'cublas']
    bufsize = 20
    rf = open(result_file_name, 'a', bufsize)
    files = _get_all_files(OUTPUT_PATH)
    rf.write('N\tsparsity\tfilename\t%s\n' % '\t'.join(algos))
    if not CONTINUE:
        candicate_files = files
    else:
        processed = []
        candicate_files = []
        with open(result_file_name, 'r') as ff:
            for line in ff.readlines()[1:]:
                processed.append(line.split('\t')[2])
        processed = str(processed)
        for f in files:
            matrix_name = f.split('/')[-1]
            if matrix_name not in processed:
                candicate_files.append(f)
    print len(files), len(candicate_files)
    for f in candicate_files:
        matrix_name = f.split('/')[-1]
        print f
        size = _get_matrix_size(f)
        if size[0] != size[1] or size[0] > 50000 or size[0] == 1:
            continue
        speeds = []
        for algo in algos:
            cmd = _get_cmd_with_file(f, algo, KERNEL_ONLY)
            print cmd
            try:
                ms = execute(cmd)
            except:
                ms = 10000000.0
            speeds.append(str(ms))
        print speeds
        print size 
        rf.write('%d\t%.9f\t%s\t%s\n'%(size[0],float(size[2])/(size[0]*size[1]),matrix_name, '\t'.join(speeds)))

    rf.close()

def bench_breakdown():
    result_file_name='breakdown.result.syth.titanx.log'
    start_n = [4000, 14000]
    sparsities = [0.95, 0.96, 0.97, 0.98, 0.99]
    algos = ['gcoospdm', 'cusparse', 'cublas']
    ns = start_n 
    bufsize = 20
    f = open(result_file_name, 'a', bufsize)
    for n in ns:
        for s in sparsities:
            speeds = []
            for algo in algos:
                cmd = _get_cmd(n, s, algo)
                print cmd
                try:
                    ms = execute_breakdown(cmd)
                except:
                    ms = '-' 
                speeds.append(ms)
            f.write('%d\t%.4f\t%s\t%s\t%s\n'%(n,s,speeds[0],speeds[1],speeds[2]))
    f.close()

if __name__ == '__main__':
    bench()
    #bench_breakdown()
    #bench_public_dataset()
