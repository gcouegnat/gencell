from meshutils import MeshBase
from subprocess import call
from tempfile import mktemp

def triangle(vertices, edges, quality=False, max_area=None, min_angle=None, verbose=False):

    filename=mktemp()

    f = open(filename+'.poly','w')
    f.write('%d 2 0 0\n' % len(vertices))
    for i, v in enumerate(vertices):
        f.write('%d %f %f\n' % (i, v[0], v[1]))
    f.write('%d 0\n' % len(edges))
    for i, e in enumerate(edges):
        f.write('%d %d %d\n' % (i, e[0], e[1]))
    f.write('0\n')
    f.write('0\n')
    f.close()

    opts = "-pzjA"

    if min_angle:
        opts += "q%d" % min_angle

    if not min_angle and quality:
        opts += "q"
    
    if max_area:
        opts += "a%.16f" % max_area

    if verbose:
        opts += "VV"
    else:
        opts += "Q"

    cmdline = ["triangle", opts, filename + ".poly"]

    print cmdline
    call(cmdline)

    f=open(filename+'.1.node','r')
    num_vertices = int(f.readline().split()[0])
    vertices = []
    for i in xrange(num_vertices):
        vertices.append([float(coord) for coord in f.readline().split()[1:3]])
    f.close()

    f=open(filename+'.1.ele','r')
    num_cells = int(f.readline().split()[0])
    cells = []
    for i in xrange(num_cells):
        cells.append([int(vertice) for vertice in f.readline().split()[1:]])
    f.close()

    mesh = MeshBase()
    mesh.set_vertices(vertices)
    mesh.set_cells(cells)

    return mesh
    

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    v = np.array([(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)])
    e = [(0,1),(1,2),(2,3),(3,0)]

    mesh = triangle(v, e, min_angle=30, max_area=0.0001)
    mesh.save("square.mesh")
    #mesh.save('random_mesh.pdf')


