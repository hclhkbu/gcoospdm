import os
from os import listdir
from os.path import isdir


DATA_PATH='/home/ubuntu/GEMM_dataset/MM/'
OUTPUT_PATH='/data/gemm/output'

def untar(path):
    dirs = listdir(path)
    for dir in dirs:
        d = '%s/%s' % (path, dir)
        if isdir(d):
            files = listdir(d)
            for f in files:
                ff = '%s/%s' % (d, f)
                op = '%s/%s/' % (OUTPUT_PATH, dir)
                if not os.path.exists(op):
                    os.makedirs(op)
                name = f.split('.')[0]
                if os.path.exists('%s/%s'%(op, name)):
                    continue
                cmd = 'tar xzvf %s -C %s' % (ff, op)
                os.system(cmd)


if __name__ == '__main__':
    untar(DATA_PATH)
