from subprocess import call
from tempfile import mktemp

from .meshutils import MeshBase


def tetgen(
    smesh,
    quality=False,
    max_volume=None,
    min_ratio=None,
    verbose=False,
    refine_faces=True,
    user_size=None,
):
    filename = mktemp()
    smesh.save(filename + ".mesh")

    opts = "-pzA"

    if min_ratio:
        opts += "q%d" % min_ratio

    if not min_ratio and quality:
        opts += "q"

    if max_volume:
        opts += "a%.16f" % max_volume

    if verbose:
        opts += "VV"
    else:
        opts += "Q"

    if not refine_faces:
        opts += "YY"

    cmdline = ["tetgen", opts, filename + ".mesh"]

    print(cmdline)
    call(cmdline)

    f = open(filename + ".1.node", "r")
    num_vertices = int(f.readline().split()[0])
    vertices = []
    for i in range(num_vertices):
        vertices.append([float(coord) for coord in f.readline().split()[1:4]])
    f.close()

    f = open(filename + ".1.ele", "r")
    num_cells = int(f.readline().split()[0])
    cells = []
    for i in range(num_cells):
        cells.append([int(vertice) for vertice in f.readline().split()[1:5]])
    f.close()

    mesh = MeshBase()
    mesh.set_vertices(vertices)
    mesh.set_cells(cells)

    if user_size:
        opts = "-ra"
        if quality:
            opts += "q"
        if not refine_faces:
            opts += "YY"
        if verbose:
            opts += "VV"
        else:
            opts += "Q"

        old_len = len(mesh.cells)
        for loop in range(1, 10):
            area = user_size(mesh)
            f = open("%s.%d.vol" % (filename, loop), "w")
            f.write("%d\n" % len(area))
            for i, a in enumerate(area):
                f.write("%d %f\n" % (i, a))
            f.close()

            cmdline = ["tetgen", opts, "%s.%d" % (filename, loop)]
            print(cmdline)
            call(cmdline)

            f = open("%s.%d.node" % (filename, loop + 1), "r")
            num_vertices = int(f.readline().split()[0])
            vertices = []
            for i in range(num_vertices):
                vertices.append([float(coord) for coord in f.readline().split()[1:4]])
            f.close()

            f = open("%s.%d.ele" % (filename, loop + 1), "r")
            num_cells = int(f.readline().split()[0])
            cells = []
            for i in range(num_cells):
                cells.append([int(vertice) for vertice in f.readline().split()[1:5]])
            f.close()

            mesh = MeshBase()
            mesh.set_vertices(vertices)
            mesh.set_cells(cells)

            if old_len == len(mesh.cells):
                break

            old_len = len(mesh.cells)

    return mesh
