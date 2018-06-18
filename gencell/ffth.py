import os
import numpy as np



class _wdir:
    def __init__(self, newpath):
        self.path = newpath

    def __enter__(self):
        self.oldpath = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.oldpath)


def run(filename, nx, ny, nz, materials, phases, loading, temperature=0, nthreads=2):

    import subprocess

    with open('mat.inp', 'w') as f:
        f.write('%d\n' % len(materials))
        for C, alpha in materials:
            for i in range(6):
                for j in range(6):
                    f.write('%e ' % C[i][j])
                f.write('\n')
            f.write('\n')
            for i in range(6):
                f.write('%e ' % alpha[i])
            f.write('\n')
            f.write('\n')


    with open('phases.inp', 'w') as f:
        f.write('%d\n' % len(phases))
        for x in phases:
            f.write('%d %e %e %e\n' % x)

    with open('load.inp', 'w') as f:
        for x in loading:
            f.write('%e\n' % x)


    os.environ['OMP_NUM_THREADS'] = str(nthreads)
    cmd = 'ffth --input %s --nx %d --ny %d --nz %d --material %s --phase %s --loading %s --temperature %e' % (filename, nx, ny, nz, 'mat.inp', 'phases.inp', 'load.inp', temperature)

    print cmd
    subprocess.call(cmd, shell=True)

    sig = np.loadtxt('res.out')

    return sig
