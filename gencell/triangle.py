from .meshutils import MeshBase
from subprocess import call
from tempfile import mktemp

def triangle(vertices, edges, quality=False, max_area=None, min_angle=None, verbose=False, refine_edges=True, user_size=None):

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

    if not refine_edges:
        opts += "Y"

    cmdline = ["triangle", opts, filename + ".poly"]

    #print cmdline
    call(cmdline)

    f=open(filename+'.1.node','r')
    num_vertices = int(f.readline().split()[0])
    vertices = []
    for i in range(num_vertices):
        vertices.append([float(coord) for coord in f.readline().split()[1:3]])
    f.close()

    f=open(filename+'.1.ele','r')
    num_cells = int(f.readline().split()[0])
    cells = []
    for i in range(num_cells):
        cells.append([int(vertice) for vertice in f.readline().split()[1:]])
    f.close()

    mesh = MeshBase()
    mesh.set_vertices(vertices)
    mesh.set_cells(cells)

    if user_size:

        opts += 'ra'
        old_len = len(mesh.cells)

        for loop in range(1,10):

            print(('Adapting mesh: iteration %d' % loop))
            area = user_size(mesh)

            f=open('%s.%d.area' % (filename, loop),'w')
            f.write('%d\n' % len(area))
            for i, a in enumerate(area):
                f.write('%d %f\n' % (i, a))
            f.close()

            cmdline = ["triangle", opts, "%s.%d.poly" % (filename, loop)]

            #print cmdline
            call(cmdline)

            f=open('%s.%d.node' % (filename, loop+1),'r')
            num_vertices = int(f.readline().split()[0])
            vertices = []
            for i in range(num_vertices):
                vertices.append([float(coord) for coord in f.readline().split()[1:3]])
            f.close()

            f=open('%s.%d.ele' % (filename, loop+1), 'r')
            num_cells = int(f.readline().split()[0])
            cells = []
            for i in range(num_cells):
                cells.append([int(vertice) for vertice in f.readline().split()[1:]])
            f.close()

            mesh = MeshBase()
            mesh.set_vertices(vertices)
            mesh.set_cells(cells)

            print('   %d cells' % len(mesh.cells))

            ratio = 1.0*abs(old_len - len(mesh.cells))/old_len
            print(ratio)

            if ratio < 0.01:
                break

            old_len = len(mesh.cells)

    return mesh
