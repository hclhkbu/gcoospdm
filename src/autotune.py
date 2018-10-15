import os
import itertools
import logging as L
import numpy as np
from perf_compare import execute

L.basicConfig(format='%(levelname)s:%(message)s', level=L.DEBUG)

class Autotune():

    def __init__(self, template_list, key_values, cmd):
        """
        template_list: ['GroupCOOSparseMatrix.h.t', 'cnnBench2.cu.t'] 
        key_values: {'$COLUMN_PER_GROUP$': [2, 4, 8],
                     '$BS$': [32, 64]}
        """
        self.template_list = template_list
        self.key_values = key_values
        self.cmd = cmd

    def _compile(self):
        L.info('Compiling ...')
        os.system('./make.sh')

    def _gen_unrolling_src(self, cpg):
        template = {}
        c = ['float c%d =0.0; '%i for i in range(cpg)]
        c_definition = ''.join(c)
        template['$c_definition$'] = c_definition
        c_unroll_write = []
        for i in range(cpg):
            if i == 0:
                s = 'if (index == 0) c0 += a*b; '
            else:
                s = 'else if (index == %d) c%d += a*b; '%(i, i)
            c_unroll_write.append(s)
        template['$c_unroll_write$'] = ''.join(c_unroll_write);
        c_unroll_write_to_C = []
        for i in range(cpg):
            s = 'if (Cj0+%d < wB) C[Ci * wB + Cj0 + %d] = c%d; ' % (i, i, i)
            c_unroll_write_to_C.append(s)
        template['$c_unroll_write_to_C$'] = ''.join(c_unroll_write_to_C);
        return template

    def _replace_src(self, kv):
        L.info('Generate source codes with configured values ...')
        for template in self.template_list:
            with open(template, 'r') as f:
                content = f.read()
                #print content
                cpg = int(kv['$COLUMN_PER_GROUP$'])
                unrolling_src = self._gen_unrolling_src(cpg)
                kv.update(unrolling_src)
                for k in kv:
                    v = kv[k]
                    content = content.replace(k, str(v))
                new_filename = template[0:-2]
                with open(new_filename, 'w') as newf:
                    newf.write(content)

    def run(self):
        keys = self.key_values.keys()
        all_values = [self.key_values[k] for k in keys]
        experiments = list(itertools.product(*all_values))
        exps = []
        for e in experiments:
            ed = {}
            for i, v in enumerate(e):
                ed[keys[i]] = v
            exps.append(ed)
        results = []
        for ed in exps:
            self._replace_src(ed)
            self._compile()
            #os.system(self.cmd)
            try:
                ms = execute(self.cmd)
            except:
                ms = 10000000.0
            results.append(ms)
        min = np.min(np.array(results))
        minidx = np.argmin(np.array(results))
        L.info('exps: {}'.format(exps))
        L.info('results: {}'.format(results))
        with open('result.log', 'a+') as f:
            f.write('%s\n%s: %f\n'%(self.cmd, exps[minidx], min))


if __name__ == '__main__':
    template_list = ['constants.h.t', 'group_spgemm_kernels.cu.t']
    #key_values = {'$COLUMN_PER_GROUP$': [1, 2, 4, 8, 16, 32], 
    #        '$BS$': [32, 64, 128, 256, 512]} 
    #key_values = {'$COLUMN_PER_GROUP$': [4], 
    #        '$BS$': [32, 64, 128, 256, 512, 1024]} 
    key_values = {'$COLUMN_PER_GROUP$': [4], 
            '$BS$': [128]} 
    with open('bc.conf', 'r') as f:
        ls = f.readlines()
        for l in ls:
            cmd = l[0:-1]
            print cmd
            at = Autotune(template_list, key_values, cmd)
            at.run()

