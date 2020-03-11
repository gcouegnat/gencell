from .meshutils import MeshBase, read_mesh
from subprocess import call
from tempfile import mktemp

def yams(mesh, option=1, hmin=None, hmax=None, gradation=None, metric=None, smoothing=True, verbose=False, memory=1000):

    filename=mktemp()
    print(filename)
    mesh.save(filename + '.mesh')

    cmdline = ['yams']
    cmdline.append('-f')
    cmdline.extend(['-O','%d' % option])
    cmdline.extend(['-m','%d' % memory])

    if hmin:
        cmdline.extend(['-hmin','%f' % hmin])

    if hmax:
        cmdline.extend(['-hmax','%f' % hmax])

    if gradation:
        cmdline.extend(['-hgrad','%f' % gradation])

    if metric is not None:
        assert len(metric) == len(mesh.vertices), "wrong metric size"
        # write .sol file
        sol = open(filename + '.sol', 'w')
        sol.write('MeshVersionFormatted\n1\nDimension\n2\nSolAtVertices\n%d\n1 1\n' % len(mesh.vertices))
        for h in metric:
            sol.write('%e\n' % h)
        sol.close()

    if not smoothing:
        cmdline.extend(['-ns'])

    if not verbose:
        cmdline.extend(['-v','0'])

    cmdline.append(filename)
    call(cmdline)
    outmesh = read_mesh(filename + '.d.mesh')

    return outmesh



