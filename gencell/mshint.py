from gencell.meshutils import read_mesh
from subprocess import call
from tempfile import mktemp


def write_sol(filename, u):
    ''' 
    Write a scalar field to *.sol format (Medit) 
    <http://www.math.u-bordeaux1.fr/~dobrzyns/logiciels/RT-422/node58.html>
    '''
    sol = open(filename, 'w')
    sol.write('MeshVersionFormatted 1\n')
    sol.write('Dimension\n2\n')
    sol.write('SolAtVertices\n%d\n' % len(u))
    sol.write('1 1\n')
    for val in u:
        sol.write('%e\n' % val)
    sol.write('End')
    sol.close()


def read_sol(filename):
    sol = []
    reading_sol = False
    lines = open(filename, 'r').readlines()
    for line in lines:
        line = line.strip()
        if line and line.startswith('1 1'):
            reading_sol = True
        elif line and reading_sol and not line.startswith('End'):
            sol.append(float(line))
    return sol


def interpolate(m1, u, m2):
    ''' 
    Interpolate a scalar field `u` from mesh1 to mesh2 using mshint
    '''
    filename1 = mktemp()  # filename for mesh m1
    filename2 = mktemp()  # filename for mesh m2
    # Write temp mesh and sol files
    m1.save(filename1 + '.mesh')
    write_sol(filename1 + '.sol', u)
    m2.save(filename2 + '.mesh')
    # Call mshint, must be in ${PATH}
    cmdline = ['mshint', filename1 + '.mesh', filename2 + '.mesh']
    call(cmdline)
    # Read interpolated solution
    v = read_sol(filename2 + '.sol')
    return v
