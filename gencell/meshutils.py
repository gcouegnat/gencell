import numpy as np


class MeshBase:
    def __init__(
        self, vertices=None, cells=None, cell_markers=None, vertex_markers=None
    ):
        self.vertices = np.asarray(vertices)
        self.cells = np.asarray(cells)
        if vertex_markers:
            self.vertex_markers = np.asarray(vertex_markers)
        else:
            self.vertex_markers = None
        if cell_markers:
            self.cell_markers = np.asarray(cell_markers)
        else:
            self.cell_markers = None

    def set_vertices(self, vertices):
        self.vertices = np.array(vertices, dtype="float")
        for i, vertex in enumerate(vertices):
            self.vertices[i] = vertex

    def set_cells(self, cells, markers=None):
        if markers is not None:
            assert len(markers) == len(cells)

        self.cells = np.array(cells, dtype="int")
        # for i, cell in enumerate(cells):
        # self.cells[i] = cell

        if markers is not None:
            self.set_cell_markers(markers)

    def set_vertex_markers(self, markers):
        assert len(markers) == len(self.vertices)
        self.vertex_markers = np.array(markers, dtype="int")

    def set_cell_markers(self, markers):
        assert len(markers) == len(self.cells)
        self.cell_markers = np.array(markers, dtype="int")
        # for i, mark in enumerate(markers):
        # self.cell_markers[i] = mark

    def set_all_cell_markers(self, marker):
        self.cell_markers = int(marker) * np.ones(len(self.cells))

    def save(self, filename):
        ext = filename.split(".")[-1]
        if ext == "mesh":
            medit_write(
                filename,
                self.vertices,
                self.cells,
                self.cell_markers,
                self.vertex_markers,
            )
        elif ext == "vtk":
            vtk_write(filename, self.vertices, self.cells, self.cell_markers)
        elif ext == "geof":
            zebulon_write(filename, self.vertices, self.cells)
        elif ext == "inp":
            abaqus_write(filename, self.vertices, self.cells, self.cell_markers)
        elif ext == "gnu":
            gnuplot_write(filename, self.vertices, self.cells, self.cell_markers)
        elif ext == "pdf":
            pdf_write(filename, self.vertices, self.cells, self.cell_markers)
        elif ext == "bdf":
            nastran_write(
                filename,
                self.vertices,
                self.cells,
                self.vertex_markers,
                self.cell_markers,
            )
        else:
            raise ValueError("Unknown mesh type: %s" % ext)

    def renumber_cell_markers(self, reverse=False):
        unique_markers, indices = np.unique(self.cell_markers, return_inverse=True)
        if reverse:
            num_unique_markers = len(unique_markers) - 1
            self.cell_markers = num_unique_markers - indices
        else:
            self.cell_markers = indices - 1

    def remove_cells_with_marker(self, marker):

        dim = self.vertices.shape[1]
        if dim == 3:
            raise ValueError("Not implemented for dim == 3")

        old_vertices = self.vertices
        old_cells = self.cells
        old_ids = self.cell_markers

        num_old_cells = old_cells.shape[0]
        num_old_vertices = old_vertices.shape[0]

        renum = -1 * np.ones(num_old_vertices)

        for k in range(num_old_cells):
            if old_ids[k] != marker:
                renum[old_cells[k, 0:3]] = 1

        num_vert = 0
        for k in range(num_old_vertices):
            if renum[k] > -1:
                renum[k] = num_vert
                num_vert += 1

        new_vertices = old_vertices[renum > -1]
        new_cells = old_cells[old_ids != marker]
        for k in range(new_cells.shape[0]):
            new_cells[k, 0:3] = renum[new_cells[k, 0:3]]

        new_ids = old_ids[old_ids != marker]

        self.set_vertices(new_vertices)
        self.set_cells(new_cells)
        self.set_cell_markers(new_ids)

    def characteristic_length(self):
        lc = 0.0
        for cell in self.cells:
            l = np.linalg.norm(self.vertices[cell[0]] - self.vertices[cell[1]])
            l += np.linalg.norm(self.vertices[cell[1]] - self.vertices[cell[2]])
            l += np.linalg.norm(self.vertices[cell[2]] - self.vertices[cell[0]])
            lc += l / 3.0

        return lc / len(self.cells)

    def remove_orphan_vertices(self):

        uvertices = np.unique(np.ravel(self.cells))
        idx = -1 * np.ones(len(self.vertices), dtype=int)
        idx[uvertices] = np.arange(len(uvertices))
        self.vertices = self.vertices[idx > -1]
        if self.vertex_markers is not None:
            self.vertex_markers = self.vertex_markers[idx > -1]
        self.cells = idx[self.cells]


def medit_reader(filename):
    def get_next_line(stream, comment="#"):
        while 1:
            next_line = stream.readline()
            if not next_line:
                return None
            elif next_line.strip().startswith(comment) or len(next_line.strip()) == 0:
                pass
            else:
                return next_line.strip()

    f = open(filename, "r")

    vertices = []
    markers = []

    tri = []
    tri_markers = []
    ntri = 0

    quad = []
    quad_markers = []
    nquad = 0

    tet = []
    tet_markers = []
    ntet = 0

    hex = []
    hex_markers = []
    nhex = 0

    while True:
        next_line = get_next_line(f)
        if not next_line:
            break

        if next_line.startswith("Dimension"):
            if len(next_line.split()) > 1:
                dim = int(next_line.split()[1])
            else:
                dim = int(get_next_line(f).split()[0])

        if next_line.startswith("Vertices"):
            if len(next_line.split()) > 1:
                num_vertex = int(next_line[1])
            else:
                num_vertex = int(get_next_line(f).split()[0])
            for i in range(num_vertex):
                next_line = get_next_line(f)
                vertices.append([float(coord) for coord in next_line.split()[0:dim]])
                markers.append(int(next_line.split()[dim]))

        if next_line.startswith("Triangles"):
            if len(next_line.split()) > 1:
                num_tri = int(next_line[1])
            else:
                num_tri = int(get_next_line(f).split()[0])
            for i in range(num_tri):
                next_line = get_next_line(f)
                tri.append([int(coord) - 1 for coord in next_line.split()[0:3]])
                tri_markers.append(int(next_line.split()[3]))
                ntri += 1

        if next_line.startswith("Quadri"):
            if len(next_line.split()) > 1:
                num_quad = int(next_line[1])
            else:
                num_quad = int(get_next_line(f).split()[0])
            for i in range(num_quad):
                next_line = get_next_line(f)
                quad.append([int(coord) - 1 for coord in next_line.split()[0:4]])
                quad_markers.append(int(next_line.split()[4]))
                nquad += 1

        if next_line.startswith("Tetrahedra"):
            if len(next_line.split()) > 1:
                num_tet = int(next_line[1])
            else:
                num_tet = int(get_next_line(f).split()[0])
            for i in range(num_tet):
                next_line = get_next_line(f)
                tet.append([int(coord) - 1 for coord in next_line.split()[0:4]])
                tet_markers.append(int(next_line.split()[4]))
                ntet += 1

        if next_line.startswith("Hexahedra"):
            if len(next_line.split()) > 1:
                num_hex = int(next_line[1])
            else:
                num_hex = int(get_next_line(f).split()[0])
            for i in range(num_hex):
                next_line = get_next_line(f)
                hex.append([int(coord) - 1 for coord in next_line.split()[0:8]])
                hex_markers.append(int(next_line.split()[8]))
                nhex += 1

    if nhex > 0:
        cells = hex
        cell_markers = hex_markers
    elif ntet > 0:
        cells = tet
        cell_markers = tet_markers
    elif nquad > 0:
        cells = quad
        cell_markers = quad_markers
    else:
        cells = tri
        cell_markers = tri_markers

    mesh = MeshBase()
    mesh.set_vertices(vertices)
    mesh.set_vertex_markers(markers)
    mesh.set_cells(cells)
    mesh.set_cell_markers(cell_markers)

    return mesh


def read_mesh(filename):
    ext = filename.split(".")[-1]
    if ext == "mesh":
        return medit_reader(filename)
    else:
        raise ValueError("Unknown mesh type: %s" % ext)


def extrude_mesh(mesh, offset=None, subdivisions=None):
    def _get_connectivity(v):
        """
        Decompose a prism into three tetrahdra.

        Dompierre, J., Labbe, P., Vallet, M.-G., Camarero, R., 1999. How to
        subdivide pyramids, prisms and hexaedra into tetrahedra. In: 8th
        International Meshing Roundtable. South Lake Tahoe, CA.
        """
        imin = np.argmin(v)
        if imin == 1:
            v = (v[1], v[2], v[0], v[4], v[5], v[3])
        elif imin == 2:
            v = (v[2], v[0], v[1], v[5], v[3], v[4])
        elif imin == 3:
            v = (v[3], v[5], v[4], v[0], v[2], v[1])
        elif imin == 4:
            v = (v[4], v[3], v[5], v[1], v[0], v[2])
        elif imin == 5:
            v = (v[5], v[4], v[3], v[2], v[1], v[0])

        if min(v[1], v[5]) < min(v[2], v[4]):
            c = (
                [v[0], v[1], v[2], v[5]],
                [v[0], v[1], v[5], v[4]],
                [v[0], v[4], v[5], v[3]],
            )
        else:
            c = (
                [v[0], v[1], v[2], v[4]],
                [v[0], v[4], v[2], v[5]],
                [v[0], v[4], v[5], v[3]],
            )
        return c

    lc = mesh.characteristic_length()
    print(lc)

    if offset is not None and subdivisions is not None:
        delta_z = offset / subdivisions
    elif subdivisions is None:
        subdivisions = int(offset / lc) + 1
        delta_z = offset / subdivisions
    else:
        delta_z = lc
        offset = delta_z * subdivisions

    print(delta_z, offset)

    nvert = len(mesh.vertices)
    ncell = len(mesh.cells)

    v = np.zeros(((subdivisions + 1) * nvert, 3))
    v[:, 0:2] = np.tile(mesh.vertices, (subdivisions + 1, 1))

    nt = 3 * ncell
    c = np.zeros((nt * subdivisions, 4), dtype="int")
    m = np.zeros((nt * subdivisions), dtype="int")

    for i in range(ncell):
        (v0, v1, v2) = mesh.cells[i]
        conn = _get_connectivity((v0, v1, v2, v0 + nvert, v1 + nvert, v2 + nvert))
        c[3 * i] = conn[0]
        c[3 * i + 1] = conn[1]
        c[3 * i + 2] = conn[2]
        m[3 * i : 3 * i + 3] = mesh.cell_markers[i]

    for i in range(1, subdivisions):
        v[i * nvert : (i + 1) * nvert, 2] = i * delta_z
        c[i * nt : (i + 1) * nt] = c[(i - 1) * nt : i * nt] + nvert
        m[i * nt : (i + 1) * nt] = m[(i - 1) * nt : i * nt]
    v[subdivisions * nvert : (subdivisions + 1) * nvert, 2] = subdivisions * delta_z

    outmesh = MeshBase()
    outmesh.set_vertices(v)
    outmesh.set_cells(c, m)

    return outmesh


def find_edges(mesh, neighbours=False):

    nv = len(mesh.vertices)
    nt = len(mesh.cells)
    ne = 3 * nt

    last_edge = -1 * np.ones(nv, dtype="int")
    next_edge = -1 * np.ones(ne, dtype="int")
    edges = -1 * np.ones((ne, 4), dtype="int")

    nedges = 0
    for i, cell in enumerate(mesh.cells):
        for k in range(3):
            p = cell[k]
            q = cell[(k + 1) % 3]
            if p > q:
                p, q = q, p

            edge_found = False
            n = last_edge[p]
            while n != -1:
                if edges[n, 0] == p and edges[n, 1] == q:
                    edge_found = True
                    edges[n, 3] = i
                n = next_edge[n]
            if edge_found == False:
                edges[nedges, 0:3] = [p, q, i]
                next_edge[nedges] = last_edge[p]
                last_edge[p] = nedges
                nedges += 1

    if neighbours == False:
        edges = edges[:nedges, 0:2]
    else:
        edges = edges[:nedges]

    return edges


def find_boundary_edges(mesh):

    edges = find_edges(mesh, neighbours=True)
    boundary_edges = np.asarray(edges[edges[:, 3] == -1])

    for edge in boundary_edges:
        ## check edge orientation
        v0 = edge[0]
        v1 = edge[1]
        t = edge[2]

        dx = mesh.vertices[v1, 0] - mesh.vertices[v0, 0]
        dy = mesh.vertices[v1, 1] - mesh.vertices[v0, 1]

        le = np.sqrt(dx * dx + dy * dy)
        nx = dy / le
        ny = -dx / le

        bary = np.mean(mesh.vertices[mesh.cells[t]], axis=0)
        middle = 0.5 * (mesh.vertices[v0] + mesh.vertices[v1])

        if np.dot([nx, ny], bary - middle) > 0.0:
            edge[0], edge[1] = edge[1], edge[0]

    return boundary_edges[:, 0:2]


def remove_duplicate_faces(mesh):
    nv = len(mesh.vertices)
    nt = len(mesh.cells)

    last_face = -1 * np.ones(nv, dtype="int")
    next_face = -1 * np.ones(nt, dtype="int")
    faces = -1 * np.ones((nt, 4), dtype="int")

    idx = np.zeros(nt)

    nfaces = 0
    for i, cell in enumerate(mesh.cells):
        v = sorted(cell)
        face_found = False
        n = last_face[v[0]]
        while n != -1:
            if faces[n, 0] == v[0] and faces[n, 1] == v[1] and faces[n, 2] == v[2]:
                face_found = True
                break
            n = next_face[n]
        if face_found is False:
            faces[nfaces, :3] = v
            faces[nfaces, 3] = i
            next_face[nfaces] = last_face[v[0]]
            last_face[v[0]] = nfaces
            nfaces += 1
        if face_found is True:
            dupl_idx = faces[n, 3]
            #            print 'Duplicate face %d (%s) with face %d (%s)' % (i, v, dupl_idx, mesh.cells[dupl_idx])
            idx[i] = 1

    mesh.cells = mesh.cells[idx == 0]
    if mesh.cell_markers is not None:
        mesh.cell_markers = mesh.cell_markers[idx == 0]

    return mesh


def merge_meshes(m1, m2, tol=1.0e-6):

    import numpy as np

    all_vert = np.vstack((m1.vertices, m2.vertices))
    all_cell = np.vstack((m1.cells, m2.cells + len(m1.vertices)))

    all_mark = np.zeros(len(all_cell))

    if m1.cell_markers is not None:
        all_mark[: len(m1.cells)] = m1.cell_markers
    if m2.cell_markers is not None:
        all_mark[len(m1.cells) :] = m2.cell_markers

    m = MeshBase(all_vert, all_cell, cell_markers=all_mark)

    xmin = np.min(m.vertices[:, 0])
    ymin = np.min(m.vertices[:, 1])
    zmin = np.min(m.vertices[:, 2])

    lx = np.max(m.vertices[:, 0]) - np.min(m.vertices[:, 0])
    ly = np.max(m.vertices[:, 1]) - np.min(m.vertices[:, 1])
    lz = np.max(m.vertices[:, 2]) - np.min(m.vertices[:, 2])

    lc = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2) / 50.0

    nx = int(lx / lc) + 1
    ny = int(ly / lc) + 1
    nz = int(lz / lc) + 1
    N = nx * ny * nz
    box = []

    for _ in range(N):
        box.append([])

    for i, v in enumerate(m.vertices):
        vx = v[0] - xmin
        vy = v[1] - ymin
        vz = v[2] - zmin
        vi = int(vx / lc)
        vj = int(vy / lc)
        vk = int(vz / lc)
        n = vi + vj * nx + vk * (nx * ny)
        box[n].append(i)

    new_num = list(range(len(m.vertices)))
    for b in box:
        if len(b) > 1:
            for i in range(len(b) - 1):
                for j in range(i + 1, len(b)):
                    vi = m.vertices[b[i]]
                    vj = m.vertices[b[j]]

                    if np.linalg.norm(vi - vj) < tol:
                        bij = min(new_num[b[i]], new_num[b[j]])
                        new_num[b[i]] = new_num[b[j]] = bij

    for i, c in enumerate(m.cells):
        m.cells[i] = [new_num[v] for v in c]

    marker = -1 * np.ones(len(m.vertices))
    for cell in m.cells:
        for p in cell:
            marker[p] = 1
    np = 0
    for i in range(len(marker)):
        if marker[i] > -1:
            marker[i] = np
            np += 1

    m.vertices = m.vertices[marker > -1]
    for i, c in enumerate(m.cells):
        m.cells[i] = [marker[v] for v in c]

    return m


def medit_write(filename, vertices, cells, ids=None, vids=None):
    """Write a mesh to Medit format """
    dim = vertices.shape[1]

    if ids is None:
        ids = np.zeros((len(cells)), dtype="int")

    if vids is None:
        vids = np.zeros((len(vertices)), dtype="int")

    f = open(filename, "w")
    f.write("MeshVersionFormatted 1\n")
    f.write("Dimension\n%d\n" % dim)

    f.write("Vertices\n%d\n" % vertices.shape[0])
    for (v, m) in zip(vertices, vids):
        if dim == 2:
            f.write("%f\t%f\t%d\n" % (v[0], v[1], m))
        elif dim == 3:
            f.write("%f\t%f\t%f\t%d\n" % (v[0], v[1], v[2], m))

    if cells.shape[1] == 3:
        f.write("Triangles\n%d\n" % cells.shape[0])
        for c, m in zip(cells, ids):
            f.write("%d\t%d\t%d\t%d\n" % (c[0] + 1, c[1] + 1, c[2] + 1, m))
    elif dim == 2 and cells.shape[1] == 4:
        f.write("Quadrilaterals\n%d\n" % cells.shape[0])
        for c, m in zip(cells, ids):
            f.write(
                "%d\t%d\t%d\t%d\t%d\n" % (c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1, m)
            )
    elif dim == 3 and cells.shape[1] == 4:
        f.write("Tetrahedra\n%d\n" % cells.shape[0])
        for i in range(len(cells)):
            f.write(
                "%d\t%d\t%d\t%d\t%d\n"
                % (
                    cells[i, 0] + 1,
                    cells[i, 1] + 1,
                    cells[i, 2] + 1,
                    cells[i, 3] + 1,
                    ids[i],
                )
            )
    elif dim == 3 and cells.shape[1] == 8:
        f.write("Hexahedra\n%d\n" % cells.shape[0])
        for i in range(len(cells)):
            f.write(
                "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n"
                % (
                    cells[i, 0] + 1,
                    cells[i, 1] + 1,
                    cells[i, 2] + 1,
                    cells[i, 3] + 1,
                    cells[i, 4] + 1,
                    cells[i, 5] + 1,
                    cells[i, 6] + 1,
                    cells[i, 7] + 1,
                    ids[i],
                )
            )

    f.write("End")
    f.close()


def vtk_write(filename, vertices, cells, ids=None, point_data=None, cell_data=None):

    dim = vertices.shape[1]

    if cells.shape[1] == 3:
        cell_type = "triangle"
        cell_type_id = 5
    elif dim == 3 and cells.shape[1] == 4:
        cell_type = "tetrahedron"
        cell_type_id = 10
    elif dim == 2 and cells.shape[1] == 4:
        cell_type = "quadrilateral"
        cell_type_id = 9
    elif cells.shape[1] == 8:
        cell_type = "hexahedron"
        cell_type_id = 12

    # print cell_type

    f = open(filename, "w")
    f.write("# vtk DataFile Version 3.0\n")
    f.write("Generated by meshutils\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    f.write("POINTS %d float\n" % len(vertices))
    for i in range(len(vertices)):
        if dim == 2:
            f.write("%f %f %f\n" % (vertices[i, 0], vertices[i, 1], 0.0))
        else:
            f.write("%f %f %f\n" % (vertices[i, 0], vertices[i, 1], vertices[i, 2]))

    f.write("CELLS %d %d\n" % (len(cells), (cells.shape[1] + 1) * len(cells)))

    for i in range(len(cells)):
        if cell_type == "triangle":
            f.write("3 %d %d %d\n" % (cells[i, 0], cells[i, 1], cells[i, 2]))
        elif cell_type == "tetrahedron" or cell_type == "quadrilateral":
            f.write(
                "4 %d %d %d %d\n" % (cells[i, 0], cells[i, 1], cells[i, 2], cells[i, 3])
            )
        elif cell_type == "hexahedron":
            f.write(
                "8 %d %d %d %d %d %d %d %d\n"
                % (
                    cells[i, 0],
                    cells[i, 1],
                    cells[i, 2],
                    cells[i, 3],
                    cells[i, 4],
                    cells[i, 5],
                    cells[i, 6],
                    cells[i, 7],
                )
            )

    f.write("CELL_TYPES %d\n" % len(cells))
    for _ in range(len(cells)):
        f.write("%d\n" % cell_type_id)

    if ids is not None or cell_data is not None:
        f.write("CELL_DATA %s\n" % len(cells))

    if ids is not None:
        f.write("SCALARS marker int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for mark in ids:
            f.write("%s\n" % mark)

    if cell_data is not None:
        for key, data in list(cell_data.items()):
            data_dim = len(data) / len(cells)
            f.write("SCALARS %s float %d\n" % (key, data_dim))
            f.write("LOOKUP_TABLE default\n")
            for i in range(len(cells)):
                if data_dim == 1:
                    f.write("%e\n" % data[i])
                elif data_dim == 2:
                    f.write("%e %e\n" % (data[2 * i], data[2 * i + 1]))
                elif data_dim == 3:
                    f.write(
                        "%e %e %e\n" % (data[3 * i], data[3 * i + 1], data[3 * i + 2])
                    )

    if point_data is not None:
        f.write("POINT_DATA %d\n" % len(vertices))
        for key, data in list(point_data.items()):

            if len(data) > len(vertices):
                f.write("SCALARS %s float 3\n" % key)
                f.write("LOOKUP_TABLE default\n")
                for i in range(len(vertices)):
                    if dim == 2:
                        f.write("%e %e %e\n" % (data[2 * i], data[2 * i + 1], 0))
                    else:
                        f.write(
                            "%e %e %e\n"
                            % (data[3 * i], data[3 * i + 1], data[3 * i + 2])
                        )
            else:
                f.write("SCALARS %s float 1\n" % key)
                f.write("LOOKUP_TABLE default\n")
                for i in range(len(vertices)):
                    f.write("%e\n" % (data[i]))

    f.close()


def gnuplot_write(filename, vertices, cells, ids):
    f = open(filename, "w")
    for points in cells:
        for pt in points:
            f.write("%f %f\n" % tuple(vertices[pt]))
        f.write("%f %f\n\n" % tuple(vertices[points[0]]))
    f.close()


def pdf_write(filename, vertices, cells, ids):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    for points in cells:
        plt.fill(
            vertices[points, 0], vertices[points, 1], facecolor="0.9", edgecolor="k"
        )
        plt.gca().set_aspect("equal")
    plt.gca().set_axis_off()
    fig.savefig(filename)


def zebulon_write(filename, vertices, cells):
    """Write mesh to Zebulon format (*.geof)"""
    dim = vertices.shape[1]
    if cells.shape[1] == 3:
        cell_type = "c2d3"
    else:
        cell_type = "c3d4"
    f = open(filename, "w")
    f.write("***geometry\n")
    f.write("**node\n")
    f.write("%d %d\n" % (vertices.shape[0], vertices.shape[1]))
    for i, v in enumerate(vertices):
        if dim == 2:
            f.write("    %d %e %e\n" % (i + 1, v[0], v[1]))
        else:
            f.write("    %d %e %e %e\n" % (i + 1, v[0], v[1], v[2]))
    f.write("**element\n")
    f.write("%d\n" % cells.shape[0])
    for i, c in enumerate(cells):
        if cell_type == "c2d3":
            f.write(
                "    %d %s %d %d %d\n"
                % (i + 1, cell_type, c[0] + 1, c[1] + 1, c[2] + 1)
            )
        else:
            f.write(
                "    %d %s %d %d %d %d\n"
                % (i + 1, cell_type, c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1)
            )
    f.write("***return")
    f.close()


def _write_by_line(f, data, max_per_line=8, sep=","):
    out = ""
    nline = len(data) // max_per_line
    for k in range(nline):
        out += "".join(
            "%d%s " % (x, sep) for x in data[k * max_per_line : (k + 1) * max_per_line]
        )
        out += "\n"
    out += "".join("%d%s " % (x, sep) for x in data[nline * max_per_line :])
    if len(data) % max_per_line:
        out += "\n"

    f.write("%s" % out)


def abaqus_write(
    filename, vertices, cells, ids=None, nsets=None, elsets=None, cell_type=None, tol=1e-4
):
    """Write mesh to Abaqus formt (*.inp)"""
    dim = vertices.shape[1]

    if cell_type is None:
        if dim == 2 and cells.shape[1] == 3:
            cell_type = "cps3"
        elif dim == 2 and cells.shape[1] == 4:
            cell_type = "cps4"
        elif dim == 3 and cells.shape[1] == 4:
            cell_type = "c3d4"
        elif dim == 3 and cells.shape[1] == 8:
            cell_type = "c3d8"
        else:
            raise ValueError("Unknown element type in dim  %s" % dim)

    f = open(filename, "w")

    # Write nodes
    f.write("*node\n")
    for i, v in enumerate(vertices):
        f.write(str(i + 1))
        for c in v:
            f.write(", " + str(c))
        f.write("\n")

    # Write elements
    f.write("*element, type=" + cell_type + "\n")
    for i, c in enumerate(cells):
        f.write(str(i + 1))
        for v in c:
            f.write(", " + str(v + 1))
        f.write("\n")

    # Write elsets
    if ids is not None:
        for idx in np.unique(ids):
            f.write("*elset, elset=elset%d\n" % idx)
            elem = np.where(ids == idx)[0] + 1
            _write_by_line(f, elem)
    elif elsets:
        for key, value in elsets.items():
            f.write("*elset, elset=%s\n" % key)
            _write_by_line(f, value + 1)

    # Write nsets
    xmin = np.min(vertices, axis=0)
    xmax = np.max(vertices, axis=0)

    nset = np.where(np.abs(vertices[:, 0] - xmin[0]) < tol)[0]
    f.write("*nset, nset=xmin\n")
    _write_by_line(f, nset + 1)

    nset = np.where(np.abs(vertices[:, 0] - xmax[0]) < tol)[0]
    f.write("*nset, nset=xmax\n")
    _write_by_line(f, nset + 1)

    nset = np.where(np.abs(vertices[:, 1] - xmin[1]) < tol)[0]
    f.write("*nset, nset=ymin\n")
    _write_by_line(f, nset + 1)

    nset = np.where(np.abs(vertices[:, 1] - xmax[1]) < tol)[0]
    f.write("*nset, nset=ymax\n")
    _write_by_line(f, nset + 1)

    if dim == 3:
        nset = np.where(np.abs(vertices[:, 2] - xmin[2]) < tol)[0]
        f.write("*nset, nset=zmin\n")
        _write_by_line(f, nset + 1)

        nset = np.where(np.abs(vertices[:, 2] - xmax[2]) < tol)[0]
        f.write("*nset, nset=zmax\n")
        _write_by_line(f, nset + 1)

    f.write("*nset, nset=all, generate\n")
    f.write("1, %d\n" % len(vertices))

    if nsets:
        for key, value in nsets.items():
            f.write("*nset, nset=%s\n" % key)
            _write_by_line(f, value + 1)

    f.close()


def nastran_write(filename, vertices, cells, tags=None, ids=None):

    if tags is None:
        tags = np.zeros(len(vertices))

    if ids is None:
        ids = np.zeros(len(cells))

    nv = len(vertices)

    with open(filename, "w") as f:
        f.write("$ Created by meshutils.py\n")
        for k, (v, t) in enumerate(zip(vertices, tags)):
            f.write(
                "%-8s%16d%16d%16.6e%16.6e%-8s\n%-8s%16.6e\n"
                % ("GRID*", k + 1, 0, v[0], v[1], "*", "*", v[2])
            )
        # for k, (c,i) in enumerate(zip(cells, ids)):
        #     f.write('%-8s%16d%16d%16d%16d%-8s\n%-8s%16d%16d\n' % ('CTETRA*', k, i, c[0], c[1], '*N%d' % (k + nv), '*N%d' % (k + nv), c[2], c[3]))
        for k, (c, i) in enumerate(zip(cells, ids)):
            f.write(
                "%-8s%8d%8d%8d%8d%8d%8d\n"
                % ("CTETRA", k + 1, i, c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1)
            )

        f.write("ENDDATA\n")
