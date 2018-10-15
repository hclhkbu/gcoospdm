from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import os
import fnmatch
import csv
OUTPUT_PATH = '/media/sf_Shared_Data/tmp/spgemm'
from pandas import Series, DataFrame
from matplotlib.ticker import ScalarFormatter
import pandas as pd
#OUTPUT_PATH = '/media/sf_Shared_Data/tmp/spgemm'
#OUTPUT_PATH = './'

def get_syth_data(device):
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.%s.raw'%device
    f = open(logfile)
    matrix_dict = {}
    for line in f.readlines():
        items = line.split('\t')
        if line[0] == 'N' or float(items[3]) > 1000000.000000 or float(items[2]) > 1000000.000000:
            continue
        #if name.split('.')[0] not in filters:
        #    continue
        name = items[0]+items[1]
        matrix_dict[name] = [items[0], 1-float(items[1]), name, items[2], items[3], items[4].strip('\n')] 
        #print(matrix_dict[name])
    return matrix_dict


def plot_sparse_vs_dense():
    N = 8192.0
    b = 16.0
    sparse = N/b**2 + N
    dense = 2.0*N/b
    print(sparse)
    print(dense)
    s = np.arange(0.1, 1, step=0.005)
    s_y = sparse * (1-s)
    d_y = [dense for i in s]
    plt.plot(s, s_y)
    plt.plot(s, d_y)
    plt.show()


def plot_real_data_logs():
    device='980'
    ##device='titanx'
    #device='p100'
    def _get_all_files(path):
        matches = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*.mtx'):
                matches.append(os.path.join(root, filename))
        return matches
    def _find_kind_of_matrix(name, files):
        try:
            index = files.index(name)
        except:
            return None
        fn = files[index]
        f = open(fn)
        for line in f.readlines():
            if line.find('kind') > 0:
                return line.split(':')[-1]
        return None
    def _get_filters():
        with open('filters.txt') as f:
            filters = []
            for line in f.readlines():
                filters.append(line.split('&')[0].strip(' \t'))
            return filters

    def plot_bar_of_filters(matrix_dict):
        keys = []
        gcoo = []
        cusparse = []
        for key in matrix_dict:
            items = matrix_dict[key]
            name = key.split('.')[0][0:8]+'.'
            keys.append(name)
            N = int(items[0])
            flop = 2 * N**3 * float(items[1]) / 1e9
            #flop = 2 * N**3  / 1e9
            gcoo.append(flop/float(items[3]))
            cusparse.append(flop/float(items[4]))
        count = len(cusparse)
        ind = np.arange(count)
        width = 0.25
        s = -int(count/2)
        print('s: ', s)
        margin = 0.01
        xticklabels = keys
        #s = (1 - (width*count+(count-1) *margin))/2+width
        ind = np.array([i for i in range(count)])
        newind = ind-width/2
        p1 = plt.bar(newind, cusparse, width, color='r',hatch='x', label='cuSPARSE')
        newind = ind+width/2
        p2 = plt.bar(newind, gcoo, width, color='g', label='GCOOSpDM')
        plt.legend()
        plt.ylabel('Effective GFLOPS')
        plt.xticks(np.arange(count),xticklabels, rotation=90)
        plt.subplots_adjust(bottom=0.19)
        plt.xlim(left=-0.6)
        #plt.savefig('%s/realbar%s.pdf' % (OUTPUT_PATH, device))
        plt.show()

    def get_real_data():
        logfile = '/media/sf_Shared_Data/tmp/spgemm/result.real.%s.raw'%device
        f = open(logfile)
        matrix_dict = {}
        filters=_get_filters()
        print(filters)
        for line in f.readlines():
            items = line.split('\t')
            if line[0] == 'N' or float(items[3]) > 1000000.000000 or float(items[4]) > 1000000.000000:
                continue
            name = items[2]
            #if name.split('.')[0] not in filters:
            #    continue
            matrix_dict[items[2]] = items
        return matrix_dict

    def plot_total_comp(matrix_dict):
        sparsities = []
        Ns = []
        ratios = []
        colors = []
        names = []
        for key in matrix_dict:
            items = matrix_dict[key]
            if float(items[1]) > 0.01:
                continue
            names.append(key)
            r = float(items[4])/float(items[3])
            ratios.append(r)
            c = 'g' if r > 1 else 'r'
            colors.append(c)
            sparsities.append(1-float(items[1]))
            Ns.append(int(items[0]))
        ratios = np.array(ratios); colors = np.array(colors)
        x = np.array(Ns)
        y = np.array(sparsities)
        larger_than = ratios > 1 
        smaller_than = ratios < 1 
        smallernames = np.array(names)[np.where(smaller_than)]
        print('smaller_than: ', np.sort(smallernames))
        print('#of larger: ', len(ratios[larger_than]))
        print('Larger avg: ', np.mean(ratios[larger_than]))
        print('Smaller avg: ', np.mean(1./ratios[smaller_than]))
        print('#of total: ', len(ratios))
        print('Percent: ', len(ratios[larger_than])*1./len(ratios))
        print('ratios: ', ratios)
        print('x: ', x)
        print('N: ', np.min(x), np.max(x))
        print('Sparisity: ', np.min(y), np.max(y))
        print('y: ', y)
        s1 = plt.scatter(x[smaller_than], y[smaller_than], c=colors[smaller_than], marker='o', s=12, label=r'$T_{GCOOSpDM} > T_{cuSPARSE}$', alpha=1)
        s2 = plt.scatter(x[larger_than], y[larger_than], c=colors[larger_than], marker='s', s=12, label=r'$T_{GCOOSpDM} < T_{cuSPARSE}$', alpha=1)
        #plt.yscale("log", basey=2)
        #plt.xscale("log", basex=2)
        plt.legend(ncol=2, loc=9)
        #plt.ylim(top=1.04, bottom=0.78)
        plt.xlim(left=0, right=15000)
        plt.xlabel(r'$N$')
        plt.ylabel('Sparsity')
        #plt.savefig('%s/syth%s.pdf' % (OUTPUT_PATH, device))
        plt.savefig('%s/real%s.pdf' % (OUTPUT_PATH, device))
        #plt.show()
    #plot_bar_of_filters(matrix_dict)
    matrix_dict = get_real_data()
    #matrix_dict = get_syth_data()
    plot_total_comp(matrix_dict)

def plot_logs():
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.p100.2.raw'
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.titanx.1.raw'
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.980.1.raw'
    #logfile = '../src/result.syth.k80.raw'
    #logfile = '../src/result.syth.980.raw'
    #logfile = '../src/result.syth.titanx.raw'
    f = open(logfile)
    cublases = []
    Ns = []
    dicts = {}
    for line in f.readlines():
        if line[0] == 'N':
            continue
        items = line.split('\t')
        cublas = float(items[-1])/1000.0 # in seconds
        N = int(items[0])
        flops = 2*N*N*N*1e-9/cublas # GFLOPS
        if N in dicts:
            dicts[N].append(flops)
        else:
            dicts[N] = [flops]
    for N in dicts:
        flopslist = dicts[N]
        cublases.append(np.mean(flopslist))
        Ns.append(N)
    plt.ylabel('GFLOPS')
    plt.xlabel('N')
    plt.scatter(Ns, cublases)
    plt.show()

gpus = ['980', 'titanx', 'p100']
specs = [(224, 4.981*1000), (433, 10.97*1000), (732, 9.5*1000)]
fig, ax = plt.subplots()
def predict_perf(r, gpu):
    i = gpus.index(gpu)
    p = specs[i]
    flops = p[1]
    mem = p[0]
    rc = flops/mem
    #print('gpu: ', gpu, ', rc: ', rc, ', r: ', r)
    if r >= rc:
        return flops * 1e9 * 0.85
    return mem * r * 1e9 

def plot_flops():
    linestyles = ['-', '--', '-.', ':']
    device='titanx'
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.%s.raw'%device
    print(logfile)
    f = open(logfile, 'r')
    content = f.readlines()
    header = content[0].split()
    print(header)
    content = content[1:]
    data = [line.split() for line in content]
    
    df = DataFrame(data, columns=header, dtype='float')
    sparsity = 0.995
    #sparsity = 0.98
    df = df[(df["sparsity"] == sparsity) & (df["N"] <= 14000)]

    flops_gcoo = df["N"] ** 3 * 2 * (1-sparsity) / (df["groupcoospgemm"] / 1000) / 1e9
    flops_cusp = df["N"] ** 3 * 2 * (1-sparsity) / (df["cusparse"] / 1000) / 1e9
    flops_cubl = df["N"] ** 3 * 2 * (1-sparsity) / (df["cublas"] / 1000) / 1e9 
    #flops_gcoo = 4*df["N"] ** 2 * 3 * (1-sparsity) / (df["groupcoospgemm"] / 1000) / 1e9
    #flops_cusp = 4*df["N"] ** 2 * 3 * (1-sparsity) / (df["cusparse"] / 1000) / 1e9
    #flops_cubl = 4*df["N"] ** 2 * 3 * (1-sparsity) / (df["cublas"] / 1000) / 1e9 

    N_range = df["N"]

    ax.plot(N_range, flops_gcoo, label="GCOOSpDM", linestyle=linestyles[0], linewidth=1, marker='s', color='g')
    ax.plot(N_range, flops_cusp, label="cuSPARSE", linestyle=linestyles[1], linewidth=1, marker='d', color='r')
    ax.plot(N_range, flops_cubl, label="cuBLAS", linestyle=linestyles[0], linewidth=1, marker='^', color='b')
    ax.set_yscale("log", basey=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.set_xscale("log", basex=2)
    ax.legend(loc=4)
    ax.set_xlabel('Matrix size '+r'$N$,'+' s=%.3f' % sparsity )
    ax.set_ylabel('Throughput [GFLOPS]')
    u.update_fontsize(ax, 14)
    fig.subplots_adjust(bottom=0.15, top=0.94)
    #plt.show()
    plt.savefig('%s/%s%s%s.pdf' % (OUTPUT_PATH, 'effective_flops', device, str(sparsity)))
    

def fit_roofline(specs, rawf):
    linestyles = ['-', '--', '-.', ':']
    f = open(rawf, "r")
    content = f.readlines()
    f.close()
    content = [line.split() for line in content[1:]]
    real_flops = [float(line[0]) ** 3 * 2 / (float(line[4]) / 1000) / 10 ** 9 for line in content]
    N_range = range(400, 10100, 100)
    N_search = range(400, 4000)
    flops = specs[1]
    mem = specs[0]
    rc = flops/mem
    best_error = 1e10
    best_n = 1000
    for n in N_search:
	alpha = rc / n
	rs = [N * alpha for N in N_range]
        th_flops = [flops if r >= rc else mem * r for r in rs]
        index = [1 if th_flops[i] < real_flops[i] else 0 for i in range(len(th_flops))]
	if sum(index) > 0:
	    continue
	aver_error = np.mean([(abs(th_flops[i] - real_flops[i]) / real_flops[i]) for i in range(len(th_flops))])
	if aver_error < best_error:
	    best_error = aver_error
	    best_n = n
	# print(th_flops, real_flops)
    alpha = rc / best_n
    #alpha = 0.0225
    rs = [N * alpha for N in N_range]
    th_flops = [flops if r >= rc else mem * r for r in rs]
    print(best_n, alpha)
    
    ax.plot(rs, th_flops, label="theoretical", linestyle=linestyles[0], linewidth=2)
    ax.plot(rs, real_flops, label="real", linestyle=linestyles[1], linewidth=2)
    ax.set_yscale("log", basey=2)
    ax.set_xscale("log", basex=2)
    ax.legend(loc=4)
    ax.set_xlabel('Operational intensity (FLOPS/byte)')
    ax.set_ylabel('Throughput (GFLOPS)')
    u.update_fontsize(ax, 14)
    fig.subplots_adjust(bottom=0.15, top=0.94)
    plt.show()

def cublas_roofline():
    linestyles = ['-', '--', '-.', ':']
    N_range = range(400, 10100, 100)

    f = open("gtx980_raw.txt", "r")
    gtx980 = f.readlines()
    f.close()
    gtx980 = [line.split() for line in gtx980[1:]]
    gtx980_flops = [float(line[0]) ** 3 * 2 / (float(line[4]) / 1000) / 10 ** 9 for line in gtx980]

    f = open("titanX_raw.txt", "r")
    titanX = f.readlines()
    f.close()
    titanX = [line.split() for line in titanX[1:]]
    titanX_flops = [float(line[0]) ** 3 * 2 / (float(line[4]) / 1000) / 10 ** 9 for line in titanX]

    alpha = 0.0225
    rs = [N * alpha for N in N_range]

    flops = specs[0][1]
    mem = specs[0][0]
    rc = flops/mem
    gtx980_th_flops = [flops if r >= rc else mem * r for r in rs]
    
    flops = specs[1][1]
    mem = specs[1][0]
    rc = flops/mem
    titanX_th_flops = [flops if r >= rc else mem * r for r in rs]

    #rs = list(N_range)
    ax.plot(rs, gtx980_th_flops, label="GTX 980 (theoretical)", linestyle=linestyles[0], linewidth=2)
    ax.scatter(rs, gtx980_flops, label="GTX 980 (CUBLAS)", linestyle=linestyles[1], linewidth=2)
    ax.plot(rs, titanX_th_flops, label="Titan X (theoretical)", linestyle=linestyles[0], linewidth=2)
    ax.scatter(rs, titanX_flops, label="GTX 980 (CUBLAS)", linestyle=linestyles[3], linewidth=2)
    ax.set_yscale("log", basey=2)
    #ax.set_xscale("log", basex=2)
    ax.legend(loc=4)
    ax.set_xlabel('Operational intensity (FLOPS/byte)')
    ax.set_ylabel('Throughput (GFLOPS)')
    u.update_fontsize(ax, 14)
    fig.subplots_adjust(bottom=0.15, top=0.94)
    #plt.show()
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'cublas_model'))
	


def rooflines():

    rs = np.arange(2, 8, step=0.05)
    rs = np.power(2,rs)#np.logspace(2, 8, base=2)
    linestyles = ['-', '--', '-.', ':']
    for i, gpu in enumerate(gpus):
        perfs = [predict_perf(r, gpu) for r in rs]
        ax.plot(rs, perfs, label=gpu, linestyle=linestyles[i], linewidth=2)
    ax.set_yscale("log", basey=2)
    ax.set_xscale("log", basex=2)
    ax.legend(loc=4)
    ax.set_xlabel('Operational intensity (FLOPS/byte)')
    ax.set_ylabel('Throughput (GFLOPS)')
    u.update_fontsize(ax, 14)
    fig.subplots_adjust(bottom=0.15, top=0.94)
    plt.show()
    #plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'rooflinemodel'))

def time_of_s(s, N, gpu):
    #r = 4* N* (1 - s) / ((1-s)*N+4)
    r = 2* N* (1 - s)/4.
    perf = predict_perf(r, gpu)
    return 2*(1-s)*N*N*N/perf, r
def time_of_s_cusparse(s, N, gpu):
    r = 2*N* (1 - s) / ((1-s)*N)
    perf = predict_perf(r, gpu)
    return 2*(1-s)*N*N*N/perf, r

def time_of_cublas(N, gpu):
    r = N/6.
    perf = predict_perf(r, gpu)
    return 2*N*N*N/perf, r

def gcoo_model():
    gpu = gpus[0]

    ss = np.arange(0.9, 1, step=0.005)
    N = 8000 
    ts = []
    rs = []
    ts_cusparse = []
    rs_cusparse = []
    ts_cb = []
    rs_cb = []
    for s in ss:
        t, r = time_of_s(s, N, gpu)
        t_cs, r_cs = time_of_s_cusparse(s, N, gpu)
        t_cb, r_cb = time_of_cublas(N, gpu)
        ts.append(t)
        rs.append(r)
        ts_cusparse.append(t_cs)
        rs_cusparse.append(r_cs)
        ts_cb.append(t_cb)
        rs_cb.append(r_cb)
    plt.plot(ss, ts, label='Time (GCOOSpDM)')
    #plt.plot(ss, rs, label='r')
    plt.plot(ss, ts_cusparse, label='Time (cuSPARSE)')
    #plt.plot(ss, rs_cusparse, label='r (cuSPARSE)')
    plt.plot(ss, ts_cb, label='Time (cuBLAS)')
    #plt.plot(ss, rs_cb, label='r (cuBLAS)')
    plt.legend()
    plt.show()

def plot_special_Ns(Ns=[500, 4000, 12000]):
    device = '980'
    gcooarrs = {}; cusparrs={};cublarrs={};sparsities={}
    for N in Ns:
        gcooarrs[N] = [] 
        cusparrs[N] = []
        cublarrs[N] = []
        sparsities[N] = []
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.%s.raw'%device
    f = open(logfile)
    for line in f.readlines():
        items = line.split('\t')
        if line[0] == 'N' or float(items[3]) > 1000000.000000 or float(items[2]) > 1000000.000000 or float(items[1]) < 0.95:
            continue
        N  = int(items[0])
        if N in Ns:
            sparsity = float(items[1])
            flop = (1-sparsity) *2 * N**3/1e9
            gcooarrs[N].append(float(items[2]))
            cusparrs[N].append(float(items[3]))
            cublarrs[N].append(float(items[4]))
            sparsities[N].append(float(items[1]))
    for N in Ns:
        plt.plot(sparsities[N], gcooarrs[N], label='GCOOSpDM', c='g', marker='s')
        plt.plot(sparsities[N], cusparrs[N], label='cuSPARSE', c='r', marker='d')
        plt.plot(sparsities[N], cublarrs[N], label='cuBLAS', c='b', marker='^')
    plt.legend()
    plt.xlabel('Sparsity')
    plt.ylabel('Time [ms]')
    plt.xlim(left=0.949, right=1.001)
    plt.savefig('%s/perfvssparsity%s_%d.pdf' % (OUTPUT_PATH, device, Ns[0]))
    #plt.show()

def compare_predict_with_real():
    device = '980'
    logfile = '/media/sf_Shared_Data/tmp/spgemm/result.syth.%s.raw'%device
    f = open(logfile)
    meas = []
    pred = []
    Ns = []
    for line in f.readlines():
        items = line.split('\t')
        if line[0] == 'N' or float(items[3]) > 1000000.000000 or float(items[2]) > 1000000.000000 or float(items[1]) < 0.95:
            continue
        N  = int(items[0])
        if N in Ns:
            continue
        Ns.append(N)
        flop = 2 * N**3/1e9
        meas.append(flop/(float(items[4])*1e-3))
        r = N/6.
        perf = predict_perf(r, device)
        pred.append(perf/1e9)
    plt.scatter(Ns, meas, label='Measured', marker='s', c='r')
    plt.scatter(Ns, pred, label='Predicted', marker='^', c='g')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #plot_sparse_vs_dense()
    #plot_logs()
    plot_real_data_logs()
    #rooflines()
    #gcoo_model()
    #compare_predict_with_real()
    #fit_roofline(specs[0], "gtx980_raw.txt")
    #fit_roofline(specs[1], "titanX_raw.txt")
    #fit_roofline(specs[2], "p100_raw.txt")
    #cublas_roofline()
    #plot_flops()
    #plot_special_Ns(Ns=[14000])
    #plot_special_Ns(Ns=[4000])
