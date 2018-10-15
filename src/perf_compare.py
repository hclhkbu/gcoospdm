import subprocess
import argparse
import numpy as np

DEBUG = False
DEBUG = True

NUM_REPEAT = 1

def _gen_args(**kwargs):
    return kwargs

def _gen_deep_bench_format(args):
        #Vector saves w, h, c, n, k, r, s, pad_w, pad_h, wstride, std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems = 
        #std::make_tuple(56, 56, 96, 1, 4, 5, 5, 0, 0, 1, 1),           
    command = 'std::make_tuple(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d),' % (args['w'], args['h'], args['c'], 1, args['nout'], args['rk'], args['sk'], args['pad'], args['pad'], args['stride'], args['stride'])
    return command 

def _gen_command(args, is_sparse=1, gpu=-1):
    command = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  ./cnnBench2 -w=%d -h=%d -c=%d -rk=%d -sk=%d -pad=%d -stride=%d -nout=%d -debug=0 -sparsity=%.3f -gpu=%d -isSparse=%d ' % (args['w'], args['h'], args['c'], args['rk'], args['sk'], args['pad'], args['stride'], args['nout'], args['sparsity'], gpu, is_sparse)
    return command 

def _gen_gpu_command(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride, is_sparse=0):
    command = './cnnBench2 -w=%d -h=%d -c=%d -rk=%d -sk=%d -pad=%d -stride=%d -nout=%d -debug=0 -isSparse=%d -sparsity=%.3f -n=%d -gpu=0' % (w, h, c, r, s, pad_w, wstride, k, is_sparse, 0.0, n)
    return command


def read_cudnn_log(filename):
    f = open(filename)
    content = f.readlines()
    times = {}
    for index, line in enumerate(content):
        network = line.strip().split(' ')[0]
        t = float(line.strip().split(' ')[-1]) / 1000
        times[network] = t
    f.close()
    return times


def execute(cmd):
    if DEBUG:
        print cmd
    #os.system(cmd)
    out = subprocess.check_output(cmd, shell=True)
    #ridx = out.find('memratio')
    #if ridx >= 0:
    #    print out[ridx:ridx+16]
    #ridx = out.find('cpuratio')
    #if ridx >= 0:
    #    print out[ridx:ridx+16]
    tidx = out.find('Time')
    out = out[tidx:]
    ms = float(out.split(':')[1])/1000
    return ms

def execute_breakdown(cmd):
    out = subprocess.check_output(cmd, shell=True)
    lines = out.split('\n')
    results = {'EO': [], 'KC': []}
    for line in lines:
        if line.find('EO') >= 0 or line.find('KC') >= 0:
            items = line.split(':')
            key = items[0]
            ms = float(items[1])/1000
            results[key].append(ms)

    tidx = out.find('Time')
    out = out[tidx:]
    total = float(out.split(':')[1])/1000

    eo = np.mean(results['EO'][1:])
    kc = np.mean(results['KC'][1:])
    ret =  '%.6f,%.6f,%.6f' % (eo, kc, total)
    return ret


def benchmark(cmd):
    avg = 0.0
    for i in range(NUM_REPEAT):
        ms = execute(cmd)
        avg += ms
    return avg / NUM_REPEAT

layers = [
        'LeNet-Conv2',
        'AlexNetC-Conv3',
        'AlexNetI-Conv2',
        'AlexNetI-Conv3',
        'AlexNetI-Conv4',
        'AlexNetI-Conv5',
        #'VGG-Conv3.1',
        #'VGG-Conv3.2',
        #'VGG-Conv4.1',
        #'VGG-Conv4.2',
        #'VGG-Conv5.1',
        #'VGG-Conv5.2',

        'GoogLeNet-Inception3a.2',
        'GoogLeNet-Inception3a.5',

        'GoogLeNet-Inception3b.1',
        'GoogLeNet-Inception3b.2',
        'GoogLeNet-Inception3b.3',
        'GoogLeNet-Inception3b.4',
        'GoogLeNet-Inception3b.5',
        'GoogLeNet-Inception3b.6',

        'GoogLeNet-Inception4a.1',
        'GoogLeNet-Inception4a.2',
        'GoogLeNet-Inception4a.3',
        'GoogLeNet-Inception4a.4',
        'GoogLeNet-Inception4a.5',
        'GoogLeNet-Inception4a.6',

        'GoogLeNet-Inception4b.1',
        'GoogLeNet-Inception4b.2',
        'GoogLeNet-Inception4b.3',
        'GoogLeNet-Inception4b.4',
        'GoogLeNet-Inception4b.5',
        'GoogLeNet-Inception4b.6',

        'GoogLeNet-Inception4c.1',
        'GoogLeNet-Inception4c.2',
        'GoogLeNet-Inception4c.3',
        'GoogLeNet-Inception4c.4',
        'GoogLeNet-Inception4c.5',
        'GoogLeNet-Inception4c.6',

        'GoogLeNet-Inception4d.1',
        'GoogLeNet-Inception4d.2',
        'GoogLeNet-Inception4d.3',
        'GoogLeNet-Inception4d.4',
        'GoogLeNet-Inception4d.5',
        'GoogLeNet-Inception4d.6',

        'GoogLeNet-Inception4e.1',
        'GoogLeNet-Inception4e.2',
        'GoogLeNet-Inception4e.3',
        'GoogLeNet-Inception4e.4',
        'GoogLeNet-Inception4e.5',
        'GoogLeNet-Inception4e.6',

        'GoogLeNet-Inception5a.1',
        'GoogLeNet-Inception5a.2',
        'GoogLeNet-Inception5a.3',
        'GoogLeNet-Inception5a.4',
        'GoogLeNet-Inception5a.5',
        'GoogLeNet-Inception5a.6',
        
        'GoogLeNet-Inception5b.1',
        'GoogLeNet-Inception5b.2',
        'GoogLeNet-Inception5b.3',
        'GoogLeNet-Inception5b.4',
        'GoogLeNet-Inception5b.5',
        'GoogLeNet-Inception5b.6',

        ]
#layers = [
#        'AlexNetI-Conv2',
#        ]


configs = {
        #Layer: {w, h, c, rk, sk, pad, stride, nout, sparsity}
        'LeNet-Conv2': _gen_args(w=11,h=11,c=20,rk=5,sk=5,pad=1,stride=1,nout=64,sparsity=0.95),
        'AlexNetC-Conv3': _gen_args(w=6,h=6,c=32,rk=5,sk=5,pad=2,stride=1,nout=64,sparsity=0.9),
        'AlexNetI-Conv2': _gen_args(w=26,h=26,c=96,rk=5,sk=5,pad=2,stride=2,nout=256,sparsity=0.6),
        'AlexNetI-Conv3': _gen_args(w=5,h=5,c=256,rk=3,sk=3,pad=1,stride=1,nout=384,sparsity=0.7),
        'AlexNetI-Conv4': _gen_args(w=5,h=5,c=384,rk=3,sk=3,pad=1,stride=1,nout=384,sparsity=0.9),
        'AlexNetI-Conv5': _gen_args(w=5,h=5,c=384,rk=3,sk=3,pad=1,stride=1,nout=256,sparsity=0.8),
        'VGG-Conv3.1': _gen_args(w=55,h=55,c=128,rk=3,sk=3,pad=1,stride=1,nout=256,sparsity=0.75),
        'VGG-Conv3.2': _gen_args(w=55,h=55,c=256,rk=3,sk=3,pad=1,stride=1,nout=256,sparsity=0.8),
        'VGG-Conv4.1': _gen_args(w=27,h=27,c=256,rk=3,sk=3,pad=1,stride=1,nout=512,sparsity=0.9),
        'VGG-Conv4.2': _gen_args(w=27,h=27,c=512,rk=3,sk=3,pad=1,stride=1,nout=512,sparsity=0.85),
        'VGG-Conv5.1': _gen_args(w=13,h=13,c=512,rk=3,sk=3,pad=1,stride=1,nout=512,sparsity=0.95),
        'VGG-Conv5.2': _gen_args(w=13,h=13,c=512,rk=3,sk=3,pad=1,stride=1,nout=512,sparsity=0.85),

        'GoogLeNet-Inception3a.2': _gen_args(w=28,h=28,c=192,rk=1,sk=1,pad=0,stride=1,nout=64,sparsity=0.84),
        'GoogLeNet-Inception3a.5': _gen_args(w=28,h=28,c=128,rk=5,sk=5,pad=2,stride=1,nout=16,sparsity=0.7),

        'GoogLeNet-Inception3b.1': _gen_args(w=28,h=28,c=256,rk=1,sk=1,pad=0,stride=1,nout=128,sparsity=0.8),
        'GoogLeNet-Inception3b.2': _gen_args(w=28,h=28,c=128,rk=1,sk=1,pad=0,stride=1,nout=128,sparsity=0.75),
        'GoogLeNet-Inception3b.3': _gen_args(w=28,h=28,c=128,rk=3,sk=3,pad=1,stride=1,nout=192,sparsity=0.85),
        'GoogLeNet-Inception3b.4': _gen_args(w=28,h=28,c=192,rk=1,sk=1,pad=0,stride=1,nout=32,sparsity=0.65),
        'GoogLeNet-Inception3b.5': _gen_args(w=28,h=28,c=32,rk=5,sk=5,pad=2,stride=1,nout=96,sparsity=0.65),
        'GoogLeNet-Inception3b.6': _gen_args(w=28,h=28,c=96,rk=1,sk=1,pad=1,stride=1,nout=64,sparsity=0.6),

        'GoogLeNet-Inception4a.1': _gen_args(w=14,h=14,c=480,rk=1,sk=1,pad=0,stride=1,nout=192,sparsity=0.9),
        'GoogLeNet-Inception4a.2': _gen_args(w=14,h=14,c=192,rk=1,sk=1,pad=0,stride=1,nout=96,sparsity=0.9),
        'GoogLeNet-Inception4a.3': _gen_args(w=14,h=14,c=96,rk=3,sk=3,pad=1,stride=1,nout=208,sparsity=0.7),
        'GoogLeNet-Inception4a.4': _gen_args(w=14,h=14,c=208,rk=1,sk=1,pad=0,stride=1,nout=16,sparsity=0.6),
        'GoogLeNet-Inception4a.5': _gen_args(w=14,h=14,c=16,rk=5,sk=5,pad=2,stride=1,nout=48,sparsity=0.75),
        'GoogLeNet-Inception4a.6': _gen_args(w=14,h=14,c=48,rk=1,sk=1,pad=1,stride=1,nout=64,sparsity=0.4),

        'GoogLeNet-Inception4b.1': _gen_args(w=14,h=14,c=512,rk=1,sk=1,pad=0,stride=1,nout=160,sparsity=0.75),
        'GoogLeNet-Inception4b.2': _gen_args(w=14,h=14,c=160,rk=1,sk=1,pad=0,stride=1,nout=112,sparsity=0.8),
        'GoogLeNet-Inception4b.3': _gen_args(w=14,h=14,c=112,rk=3,sk=3,pad=1,stride=1,nout=224,sparsity=0.7),
        'GoogLeNet-Inception4b.4': _gen_args(w=14,h=14,c=224,rk=1,sk=1,pad=0,stride=1,nout=24,sparsity=0.6),
        'GoogLeNet-Inception4b.5': _gen_args(w=14,h=14,c=24,rk=5,sk=5,pad=2,stride=1,nout=64,sparsity=0.7),
        'GoogLeNet-Inception4b.6': _gen_args(w=14,h=14,c=64,rk=1,sk=1,pad=1,stride=1,nout=64,sparsity=0.45),

        'GoogLeNet-Inception4c.1': _gen_args(w=14,h=14,c=512,rk=1,sk=1,pad=0,stride=1,nout=128,sparsity=0.8),
        'GoogLeNet-Inception4c.2': _gen_args(w=14,h=14,c=128,rk=1,sk=1,pad=0,stride=1,nout=128,sparsity=0.8),
        'GoogLeNet-Inception4c.3': _gen_args(w=14,h=14,c=128,rk=3,sk=3,pad=1,stride=1,nout=256,sparsity=0.7),
        'GoogLeNet-Inception4c.4': _gen_args(w=14,h=14,c=256,rk=1,sk=1,pad=0,stride=1,nout=24,sparsity=0.65),
        'GoogLeNet-Inception4c.5': _gen_args(w=14,h=14,c=24,rk=5,sk=5,pad=2,stride=1,nout=64,sparsity=0.7),
        'GoogLeNet-Inception4c.6': _gen_args(w=14,h=14,c=64,rk=1,sk=1,pad=1,stride=1,nout=64,sparsity=0.5),

        'GoogLeNet-Inception4d.1': _gen_args(w=14,h=14,c=512,rk=1,sk=1,pad=0,stride=1,nout=112,sparsity=0.75),
        'GoogLeNet-Inception4d.2': _gen_args(w=14,h=14,c=112,rk=1,sk=1,pad=0,stride=1,nout=144,sparsity=0.8),
        'GoogLeNet-Inception4d.3': _gen_args(w=14,h=14,c=144,rk=3,sk=3,pad=1,stride=1,nout=288,sparsity=0.7),
        'GoogLeNet-Inception4d.4': _gen_args(w=14,h=14,c=288,rk=1,sk=1,pad=0,stride=1,nout=32,sparsity=0.7),
        'GoogLeNet-Inception4d.5': _gen_args(w=14,h=14,c=32,rk=5,sk=5,pad=2,stride=1,nout=64,sparsity=0.75),
        'GoogLeNet-Inception4d.6': _gen_args(w=14,h=14,c=64,rk=1,sk=1,pad=1,stride=1,nout=64,sparsity=0.5),

        'GoogLeNet-Inception4e.1': _gen_args(w=14,h=14,c=528,rk=1,sk=1,pad=0,stride=1,nout=256,sparsity=0.8),
        'GoogLeNet-Inception4e.2': _gen_args(w=14,h=14,c=256,rk=1,sk=1,pad=0,stride=1,nout=160,sparsity=0.8),
        'GoogLeNet-Inception4e.3': _gen_args(w=14,h=14,c=160,rk=3,sk=3,pad=1,stride=1,nout=320,sparsity=0.9),
        'GoogLeNet-Inception4e.4': _gen_args(w=14,h=14,c=320,rk=1,sk=1,pad=0,stride=1,nout=32,sparsity=0.85),
        'GoogLeNet-Inception4e.5': _gen_args(w=14,h=14,c=32,rk=5,sk=5,pad=2,stride=1,nout=128,sparsity=0.6),
        'GoogLeNet-Inception4e.6': _gen_args(w=14,h=14,c=128,rk=1,sk=1,pad=1,stride=1,nout=128,sparsity=0.6),

        'GoogLeNet-Inception5a.1': _gen_args(w=7,h=7,c=832,rk=1,sk=1,pad=0,stride=1,nout=256,sparsity=0.95),
        'GoogLeNet-Inception5a.2': _gen_args(w=7,h=7,c=256,rk=1,sk=1,pad=0,stride=1,nout=160,sparsity=0.9),
        'GoogLeNet-Inception5a.3': _gen_args(w=7,h=7,c=160,rk=3,sk=3,pad=1,stride=1,nout=320,sparsity=0.7),
        'GoogLeNet-Inception5a.4': _gen_args(w=7,h=7,c=320,rk=1,sk=1,pad=0,stride=1,nout=32,sparsity=0.65),
        'GoogLeNet-Inception5a.5': _gen_args(w=7,h=7,c=32,rk=5,sk=5,pad=2,stride=1,nout=128,sparsity=0.75),
        'GoogLeNet-Inception5a.6': _gen_args(w=7,h=7,c=128,rk=1,sk=1,pad=1,stride=1,nout=128,sparsity=0.6),
        
        'GoogLeNet-Inception5b.1': _gen_args(w=7,h=7,c=832,rk=1,sk=1,pad=0,stride=1,nout=384,sparsity=0.8),
        'GoogLeNet-Inception5b.2': _gen_args(w=7,h=7,c=384,rk=1,sk=1,pad=0,stride=1,nout=192,sparsity=0.8),
        'GoogLeNet-Inception5b.3': _gen_args(w=7,h=7,c=192,rk=3,sk=3,pad=1,stride=1,nout=384,sparsity=0.95),
        'GoogLeNet-Inception5b.4': _gen_args(w=7,h=7,c=384,rk=1,sk=1,pad=0,stride=1,nout=48,sparsity=0.75),
        'GoogLeNet-Inception5b.5': _gen_args(w=7,h=7,c=48,rk=5,sk=5,pad=2,stride=1,nout=128,sparsity=0.95),
        'GoogLeNet-Inception5b.6': _gen_args(w=7,h=7,c=128,rk=1,sk=1,pad=1,stride=1,nout=128,sparsity=0.65),

        }



def perf():
    print('\tLayer\t&Sparsity & GEMM\t&ISC\t&Speedup \\\\\\cline{1-4}')
    for k in layers:
        v = configs[k]
        #if float(v['sparsity']) < 0.9:
        #    continue
        sc = _gen_command(v)
        dc = _gen_command(v, is_sparse=0)
        sparse_ms = benchmark(sc)
        dense_ms = benchmark(dc)
        print('\t%s&\t%s\t&%.3f\t&%.3f\t&%.2fX \\\\\\cline{1-4}' % (k, v['sparsity'], dense_ms, sparse_ms, dense_ms/sparse_ms))

def perf_gpu():
    dense_times = read_cudnn_log('cudnn_perf_titanx.txt')
    #dense_times = read_cudnn_log('cudnn_perf_k80.txt')
    for k in layers:
        v = configs[k]
        c = _gen_command(v, is_sparse=1, gpu=0)
        #if int(v['nout']) > 64 or int(v['w']) > 15:
        #    continue
        try:
            sparse_ms = benchmark(c)
        except:
            sparse_ms = 10000
        dense_ms = dense_times[k]
        print('w:%s, h:%s, sparsity: %s'%(v['w'],v['h'],v['sparsity']))
        print('\t%s\t&%.3f\t&%.3f\t&%.2fX \\\\\\cline{1-4}' % (k, dense_ms, sparse_ms, dense_ms/sparse_ms))

def print_for_deep_bench():
    for k in layers:
        #Vector saves w, h, c, n, k, r, s, pad_w, pad_h, wstride, std::vector<std::tuple<int, int, int, int, int, int, int, int, int, int, int>> problems = 
        #std::make_tuple(56, 56, 96, 1, 4, 5, 5, 0, 0, 1, 1),           
        v = configs[k]
        c = _gen_deep_bench_format(v)
        print c



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark script')
    parser.add_argument('-D', '--debug', help='Debug mode', default='0')
    p = parser.parse_args()
    DEBUG = p.debug == '1'
    if DEBUG:
        NUM_REPEAT = 1
    #perf()
    perf_gpu()
    #print_for_deep_bench()
    #read_cudnn_log('cudnn_perf.txt')
    
