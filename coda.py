import os

class DirichletBC():
    def __init__(self, vertices=None, ux=None, uy=None):
        self.vertices = vertices
        self.ux = ux
        self.uy = uy

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_ux(self, ux):
        self.ux = ux

    def set_uy(self, uy):
        self.uy = uy

class _wdir:

    def __init__(self, newpath):
        self.path = newpath
    
    def __enter__(self):
        self.oldpath = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.oldpath)

def _load_data(filename):
    f = open(filename, 'r')
    data = f.read()
    return [float(value) for value in data.split()]

def run(mesh, bcs, materials=None, temperature=0.0, output=None):

    from tempfile import gettempdir

    tmpdir = gettempdir()
    print 'Working in %s' % tmpdir

    with _wdir(tmpdir):
        # mesh
        filename = 'mesh.txt'
        f = open(filename, 'w')
        f.write('%d\n' % mesh.vertices.shape[0])
        for i in range(mesh.vertices.shape[0]):
            f.write('%f %f\n' % (mesh.vertices[i,0], mesh.vertices[i,1]))
        f.write('%d\n' % mesh.cells.shape[0])
        for i in range(mesh.cells.shape[0]):
            f.write('%d %d %d %d\n' % (mesh.cells[i,0], mesh.cells[i,1], mesh.cells[i,2], mesh.cell_markers[i]))
        f.write('%d\n' % len(bcs))
        for bc in bcs:
            vertices = bc.vertices[0]
            f.write('%d\n' % len(vertices))
            for vertex in vertices:
                f.write('%d\n' % vertex)
        f.close()

        # boundary conditions
        nbc = 0
        for bc in bcs:
            if bc.ux is not None: nbc += 1
            if bc.uy is not None: nbc += 1

        f = open('bc.txt', 'w')
        f.write('%d\n' % nbc)
        for i, bc in enumerate(bcs):
            if bc.ux is not None:
                f.write('%d 0 %f\n' % (i, bc.ux))
            if bc.uy is not None:
                f.write('%d 1 %f\n' % (i, bc.uy))
        f.close()

        f = open('temperature.txt', 'w')
        f.write('%d\n' % temperature)
        f.close()

        # materials
        filename = 'materials.txt'
        f = open(filename, 'w')
        f.write('%d\n' % len(materials))
        for C, alpha in materials:
            for i in range(6):
                for j in range(i,6):
                   f.write('%f ' % C[i][j])
            for i in range(6):
                f.write('%f ' % alpha[i])
            f.write('\n')
        f.close()

        # run coda
        from subprocess import call
        cmdline=["/Users/couegnat/dev/fea-thermal-strain/fea2d"]
        print cmdline
        call(cmdline)

        point_data={}
        point_data['U'] = _load_data('u.txt')

        cell_data={}
        cell_data['S11'] = _load_data('sig11.txt')
        cell_data['S22'] = _load_data('sig22.txt')
        cell_data['S12'] = _load_data('sig12.txt')
        cell_data['E11'] = _load_data('eto11.txt')
        cell_data['E22'] = _load_data('eto22.txt')
        cell_data['E12'] = _load_data('eto12.txt')

    if output is not None:
        if not output.endswith('.vtk'):
            output += '.vtk'

        from gencell.meshutils import vtk_write
        vtk_write(output, mesh.vertices, mesh.cells,
                                     mesh.cell_markers, cell_data = cell_data,
                                     point_data = point_data)

    return cell_data
