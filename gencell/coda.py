import os

import numpy as np


class DirichletBC:
    def __init__(self, vertices=None, ux=None, uy=None, uz=None):
        self.vertices = vertices
        self.ux = ux
        self.uy = uy
        self.uz = uz

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_ux(self, ux):
        self.ux = ux

    def set_uy(self, uy):
        self.uy = uy

    def set_uz(self, uz):
        self.uz = uz


class LoadBC:
    def __init__(self, vertices=None, fx=None, fy=None, fz=None):
        self.vertices = vertices
        self.fx = fx
        self.fy = fy
        self.fz = fz

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_fx(self, fx):
        self.fx = fx

    def set_fy(self, fy):
        self.fy = fy

    def set_fz(self, fz):
        self.fz = fz


class MPC:
    def __init__(self, slave=None, master=None, ux=None, uy=None, uz=None):
        self.slave = slave
        self.master = master
        self.ux = ux
        self.uy = uy
        self.uz = uz


class _wdir:
    def __init__(self, newpath):
        self.path = newpath

    def __enter__(self):
        self.oldpath = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.oldpath)


def _load_data(filename):
    f = open(filename, "r")
    data = f.read()
    return [float(value) for value in data.split()]


def run(
    mesh,
    bcs,
    materials,
    temperature=0.0,
    sig=None,
    mpcs=None,
    loads=None,
    output=None,
    verbose=True,
    prev=None,
):

    from tempfile import mkdtemp
    from shutil import rmtree

    tmpdir = mkdtemp()
    if verbose:
        print("Working in %s" % tmpdir)

    dim = mesh.vertices.shape[1]

    if mesh.cells.shape[1] == 3:
        cell_type = "tri"
    elif mesh.cells.shape[1] == 4:
        cell_type = "tet"
    else:
        cell_type = "hex"

    with _wdir(tmpdir):
        # mesh
        filename = "mesh.txt"
        f = open(filename, "w")
        f.write("%d\n" % mesh.vertices.shape[0])
        for i in range(mesh.vertices.shape[0]):
            if dim == 2:
                f.write("%e %e\n" % (mesh.vertices[i, 0], mesh.vertices[i, 1]))
            else:
                f.write(
                    "%e %e %e\n"
                    % (mesh.vertices[i, 0], mesh.vertices[i, 1], mesh.vertices[i, 2])
                )

        f.write("%d\n" % mesh.cells.shape[0])
        for i in range(mesh.cells.shape[0]):
            if cell_type == "tri":
                f.write(
                    "%d %d %d %d\n"
                    % (
                        mesh.cells[i, 0],
                        mesh.cells[i, 1],
                        mesh.cells[i, 2],
                        mesh.cell_markers[i],
                    )
                )
            elif cell_type == "tet":
                f.write(
                    "%d %d %d %d %d\n"
                    % (
                        mesh.cells[i, 0],
                        mesh.cells[i, 1],
                        mesh.cells[i, 2],
                        mesh.cells[i, 3],
                        mesh.cell_markers[i],
                    )
                )
            else:
                f.write(
                    "%d %d %d %d %d %d %d %d %d\n"
                    % (
                        mesh.cells[i, 0],
                        mesh.cells[i, 1],
                        mesh.cells[i, 2],
                        mesh.cells[i, 3],
                        mesh.cells[i, 4],
                        mesh.cells[i, 5],
                        mesh.cells[i, 6],
                        mesh.cells[i, 7],
                        mesh.cell_markers[i],
                    )
                )

        # nsets
        nnset = len(bcs)
        if loads is not None:
            nnset += len(loads)

        f.write("%d\n" % nnset)
        for bc in bcs:
            vertices = bc.vertices[0]
            f.write("%d\n" % len(vertices))
            for vertex in vertices:
                f.write("%d\n" % vertex)
        if loads is not None:
            for load in loads:
                vertices = load.vertices[0]
                f.write("%d\n" % len(vertices))
                for vertex in vertices:
                    f.write("%d\n" % vertex)
        f.close()

        # boundary conditions
        nbc = 0
        for bc in bcs:
            if bc.ux is not None:
                nbc += 1
            if bc.uy is not None:
                nbc += 1
            if dim == 3 and bc.uz is not None:
                nbc += 1

        f = open("bc.txt", "w")
        f.write("%d\n" % nbc)
        for i, bc in enumerate(bcs):
            if bc.ux is not None:
                f.write("%d 0 %e\n" % (i, bc.ux))
            if bc.uy is not None:
                f.write("%d 1 %e\n" % (i, bc.uy))
            if dim == 3 and bc.uz is not None:
                f.write("%d 2 %e\n" % (i, bc.uz))
        f.close()

        # loads
        nload = 0
        if loads is not None:
            for load in loads:
                if load.fx is not None:
                    nload += 1
                if load.fy is not None:
                    nload += 1
                if dim == 3 and load.uz is not None:
                    nload += 1

        f = open("load.txt", "w")
        f.write("%d\n" % nload)
        if nload > 0:
            for i, load in enumerate(loads):
                if load.fx is not None:
                    f.write("%d 0 %e\n" % (i + len(bcs), load.fx))
                if load.fy is not None:
                    f.write("%d 1 %e\n" % (i + len(bcs), load.fy))
                if dim == 3 and load.fz is not None:
                    f.write("%d 2 %e\n" % (i + len(bcs), load.fz))
        f.close()

        # multipoint constraints
        nmpc = 0
        if mpcs is not None:
            for mpc in mpcs:
                if mpc.ux is not None:
                    nmpc += 1
                if mpc.uy is not None:
                    nmpc += 1
                if dim == 3 and mpc.uz is not None:
                    nmpc += 1

        f = open("mpc.txt", "w")
        f.write("%d\n" % nmpc)
        if nmpc > 0:
            for i, mpc in enumerate(mpcs):
                if mpc.ux is not None:
                    f.write(
                        "%d %e %d %e %d %e\n"
                        % (mpc.slave, 1.0, mpc.master, -1.0, 0, mpc.ux)
                    )
                if mpc.uy is not None:
                    f.write(
                        "%d %e %d %e %d %e\n"
                        % (mpc.slave, 1.0, mpc.master, -1.0, 1, mpc.uy)
                    )
                if mpc.uz is not None:
                    f.write(
                        "%d %e %d %e %d %e\n"
                        % (mpc.slave, 1.0, mpc.master, -1.0, 2, mpc.uz)
                    )
        f.close()

        f = open("temperature.txt", "w")
        f.write("%d\n" % temperature)
        f.close()

        # materials
        filename = "materials.txt"
        f = open(filename, "w")
        f.write("%d\n" % len(materials))
        for C, alpha in materials:
            for i in range(6):
                for j in range(i, 6):
                    f.write("%e " % C[i][j])
            for i in range(6):
                f.write("%e " % alpha[i])
            f.write("\n")
        f.close()

        # previous sol
        filename = "u.txt"
        f = open(filename, "w")
        if prev is not None:
            uprev = np.asarray(prev)
        else:
            uprev = np.zeros(dim * mesh.vertices.shape[0])
        for u in uprev:
            f.write("%.16e\n" % u)

        # initial stress
        filename = "sig.txt"
        f = open(filename, "w")
        if sig is not None:
            f.write("%d\n" % mesh.cells.shape[0])
            for i in range(mesh.cells.shape[0]):
                f.write("%e %e %e\n" % (sig[i, 0], sig[i, 1], sig[i, 2]))
        else:
            f.write("0\n")
        f.close()

        # run coda
        import subprocess

        if cell_type == "tri":
            cmdline = ["fea2d"]
        elif cell_type == "tet":
            cmdline = ["fea3d"]
        else:
            cmdline = ["fea3d_hexa"]

        if not verbose:
            subprocess.call(cmdline, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        else:
            subprocess.call(cmdline)

        point_data = {}
        point_data["U"] = _load_data("u.txt")
        point_data["F"] = _load_data("fint.txt")

        cell_data = {}
        cell_data["S11"] = _load_data("sig11.txt")
        cell_data["S22"] = _load_data("sig22.txt")
        cell_data["S12"] = _load_data("sig12.txt")
        cell_data["E11"] = _load_data("eto11.txt")
        cell_data["E22"] = _load_data("eto22.txt")
        cell_data["E12"] = _load_data("eto12.txt")
        cell_data["EEL11"] = _load_data("eel11.txt")
        cell_data["EEL22"] = _load_data("eel22.txt")
        cell_data["EEL12"] = _load_data("eel12.txt")
        cell_data["ETH11"] = _load_data("eth11.txt")
        cell_data["ETH22"] = _load_data("eth22.txt")
        cell_data["ETH12"] = _load_data("eth12.txt")
        if dim == 3:
            cell_data["S33"] = _load_data("sig33.txt")
            cell_data["S23"] = _load_data("sig23.txt")
            cell_data["S31"] = _load_data("sig31.txt")
            cell_data["E33"] = _load_data("eto33.txt")
            cell_data["E23"] = _load_data("eto23.txt")
            cell_data["E31"] = _load_data("eto31.txt")

    # remove tmp directory
    # rmtree(tmpdir)

    # save results to VTK format
    if output is not None:
        if not output.endswith(".vtk"):
            output += ".vtk"

        from gencell.meshutils import vtk_write

        if verbose:
            print("Writing output")
        vtk_write(
            output,
            mesh.vertices,
            mesh.cells,
            mesh.cell_markers,
            cell_data=cell_data,
            point_data=point_data,
        )

    data = {}
    data["U"] = point_data["U"]
    data["F"] = point_data["F"]
    data["E11"] = cell_data["E11"]
    data["E22"] = cell_data["E22"]
    data["E12"] = cell_data["E12"]
    data["EEL11"] = cell_data["EEL11"]
    data["EEL22"] = cell_data["EEL22"]
    data["EEL12"] = cell_data["EEL12"]
    data["ETH11"] = cell_data["ETH11"]
    data["ETH22"] = cell_data["ETH22"]
    data["ETH12"] = cell_data["ETH12"]
    data["S11"] = cell_data["S11"]
    data["S22"] = cell_data["S22"]
    data["S12"] = cell_data["S12"]
    if dim == 3:
        data["E33"] = cell_data["E33"]
        data["E23"] = cell_data["E23"]
        data["E31"] = cell_data["E31"]
        data["S33"] = cell_data["S33"]
        data["S23"] = cell_data["S23"]
        data["S31"] = cell_data["S31"]

    return data
