import sys
import numpy as np
import coda


def compute_cell_volume(coords):
    npts, dim = coords.shape

    if dim == 2 and npts == 3:
        # triangle
        V = (coords[1][0] * coords[2][1] - coords[2][0] * coords[1][1])
        V += (coords[2][0] * coords[0][1] - coords[0][0] * coords[2][1])
        V += (coords[0][0] * coords[1][1] - coords[1][0] * coords[0][1])
        V *= 0.5
    elif dim == 3 and npts == 4:
        # tetrahedron
        x21 = coords[1][0] - coords[0][0]
        x31 = coords[2][0] - coords[0][0]
        x41 = coords[3][0] - coords[0][0]
        y21 = coords[1][1] - coords[0][1]
        y31 = coords[2][1] - coords[0][1]
        y41 = coords[3][1] - coords[0][1]
        z21 = coords[1][2] - coords[0][2]
        z31 = coords[2][2] - coords[0][2]
        z41 = coords[3][2] - coords[0][2]
        V = x21 * (y31 * z41 - y41 * z31)
        V += y21 * (x41 * z31 - x31 * z41)
        V += z21 * (x31 * y41 - x41 * y31)
        V /= 6.0
    elif dim == 3 and npts == 8:
        # voxel
        x21 = np.abs(coords[1][0] - coords[0][0])
        y41 = np.abs(coords[3][1] - coords[0][1])
        z51 = np.abs(coords[4][2] - coords[0][2])
        V = x21 * y41 * z51
    return V


def volumic_average(mesh, data):
    acc = 0.0
    for i in xrange(len(data)):
        acc += data[i] * mesh.cell_volumes[i]
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    A = np.prod(xmax-xmin)
    acc /= A
    return acc


def compute_effective_properties_PMUBC(mesh, materials, output=None):
    """
    Compute effective properties of a rectangular cell using Periodicty
    compatible Mixed Unfiform Boundary Conditions (PMUBC) [1]_.

    .. [1]  D.H. Pahr and P.  K. Zysset, Influence of boundary conditions on
           computed apparent elastic properties of cancellous bone, Biomech. Model.
           Mechanobiol., 7, pp. 463--476, 2008.
    """

    def output_name(i):
        if output:
            return output + '.%d.vtk' % i
        else:
            return None

    dim = mesh.vertices.shape[1]

    # Compute cell volume for latter volume averaging
    mesh.cell_volumes = np.zeros((len(mesh.cells)))
    for i in xrange(len(mesh.cells)):
        mesh.cell_volumes[i] = compute_cell_volume(mesh.vertices[mesh.cells[i]])

    # BCs
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    lx = xmax[0] - xmin[0]
    ly = xmax[1] - xmin[1]

    west = np.where(np.abs(mesh.vertices[:, 0] - xmin[0]) < 1e-4)
    east = np.where(np.abs(mesh.vertices[:, 0] - xmax[0]) < 1e-4)
    south = np.where(np.abs(mesh.vertices[:, 1] - xmin[1]) < 1e-4)
    north = np.where(np.abs(mesh.vertices[:, 1] - xmax[1]) < 1e-4)

    if dim == 2:
        Ceff = np.zeros((3,3))

        # Tensile 1
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux= 0.01*lx)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.0)
        bcs = [bc_west, bc_east, bc_south, bc_north]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(0))
        Ceff[0,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[0,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[0,2] = volumic_average(mesh, gp_data['S12'])

        # Tensile 2
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux= 0.0)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.01*ly)
        bcs = [bc_west, bc_east, bc_south, bc_north]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(1))
        Ceff[1,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[1,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[1,2] = volumic_average(mesh, gp_data['S12'])

        # Shear 12
        bc_west = coda.DirichletBC(west, uy=0.0)
        bc_east = coda.DirichletBC(east, uy= 0.005*lx)
        bc_south= coda.DirichletBC(south, ux=0.0)
        bc_north = coda.DirichletBC(north, ux=0.005*ly)
        bcs = [bc_west, bc_east, bc_south, bc_north]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(2))
        Ceff[2,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[2,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[2,2] = volumic_average(mesh, gp_data['S12'])

        # Thermal
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east,ux=0.0)
        bc_south= coda.DirichletBC(south,  uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.0)
        bcs = [bc_west, bc_east, bc_south, bc_north]
        gp_data = coda.run(mesh, bcs, materials, temperature=100.0, output=output_name(3))
        sig = [ volumic_average(mesh, gp_data['S11']),
                volumic_average(mesh, gp_data['S22']),
                volumic_average(mesh, gp_data['S12'])]
    else:
        # BCs
        lz = xmax[2] - xmin[2]
        bottom = np.where(np.abs(mesh.vertices[:, 2] - xmin[2]) < 1e-4)
        top = np.where(np.abs(mesh.vertices[:, 2] - xmax[2]) < 1e-4)
        Ceff = np.zeros((6,6))

        # Tensile 1
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux= 0.01*lx)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.0)
        bc_bottom = coda.DirichletBC(bottom, uz=0.0)
        bc_top = coda.DirichletBC(top, uz=0.0)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(0))
        Ceff[0,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[0,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[0,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[0,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[0,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[0,5] = volumic_average(mesh, gp_data['S31'])

        # Tensile 2
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux=0.0)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.01*ly)
        bc_bottom = coda.DirichletBC(bottom, uz=0.0)
        bc_top = coda.DirichletBC(top, uz=0.0)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(1))
        Ceff[1,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[1,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[1,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[1,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[1,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[1,5] = volumic_average(mesh, gp_data['S31'])
        
        # Tensile 3
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux=0.0)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.0)
        bc_bottom = coda.DirichletBC(bottom, uz=0.0)
        bc_top = coda.DirichletBC(top, uz=0.01*lz)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(2))
        Ceff[2,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[2,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[2,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[2,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[2,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[2,5] = volumic_average(mesh, gp_data['S31'])
        
        # Shear12
        bc_west = coda.DirichletBC(west, uy=0.0)
        bc_east = coda.DirichletBC(east, uy= 0.005*lx)
        bc_south= coda.DirichletBC(south, ux=0.0)
        bc_north = coda.DirichletBC(north, ux=0.005*ly)
        bc_bottom = coda.DirichletBC(bottom, uz=0)
        bc_top = coda.DirichletBC(top, uz=0)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(3))
        Ceff[3,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[3,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[3,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[3,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[3,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[3,5] = volumic_average(mesh, gp_data['S31'])
        
        # Shear 23
        bc_west = coda.DirichletBC(west, ux=0.0)
        bc_east = coda.DirichletBC(east, ux=0.0)
        bc_south= coda.DirichletBC(south, uz=0.0)
        bc_north = coda.DirichletBC(north, uz=0.005*ly)
        bc_bottom = coda.DirichletBC(bottom, uy=0.0)
        bc_top = coda.DirichletBC(top, uy=0.005*lz)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(4))
        Ceff[4,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[4,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[4,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[4,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[4,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[4,5] = volumic_average(mesh, gp_data['S31'])
        
        # Shear 31
        bc_west = coda.DirichletBC(west, uz=0.0)
        bc_east = coda.DirichletBC(east, uz= 0.005*lx)
        bc_south= coda.DirichletBC(south, uy=0.0)
        bc_north = coda.DirichletBC(north, uy=0.0)
        bc_bottom = coda.DirichletBC(bottom, ux=0.0)
        bc_top = coda.DirichletBC(top, ux=0.005*lz)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, output=output_name(5))
        Ceff[5,0] = volumic_average(mesh, gp_data['S11'])
        Ceff[5,1] = volumic_average(mesh, gp_data['S22'])
        Ceff[5,2] = volumic_average(mesh, gp_data['S33'])
        Ceff[5,3] = volumic_average(mesh, gp_data['S12'])
        Ceff[5,4] = volumic_average(mesh, gp_data['S23'])
        Ceff[5,5] = volumic_average(mesh, gp_data['S31'])
        
        # Thermal
        bc_east = coda.DirichletBC(east, ux=0)
        bc_west = coda.DirichletBC(west, ux=0)
        bc_south= coda.DirichletBC(south, uy=0)
        bc_north = coda.DirichletBC(north, uy=0)
        bc_bottom = coda.DirichletBC(bottom, uz=0)
        bc_top = coda.DirichletBC(top, uz=0)
        bcs = [bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top]
        gp_data = coda.run(mesh, bcs, materials, temperature=100.0, output=output_name(6))
        sig = [volumic_average(mesh, gp_data['S11']),
               volumic_average(mesh, gp_data['S22']),
               volumic_average(mesh, gp_data['S33']),
               volumic_average(mesh, gp_data['S12']),
               volumic_average(mesh, gp_data['S23']),
               volumic_average(mesh, gp_data['S31'])]

    Ceff = 0.5e2*(Ceff + Ceff.transpose())
    aeff = -0.01 * np.dot(np.linalg.inv(Ceff), sig)

    return Ceff, aeff


def compute_effective_properties_KUBC(mesh, materials, output=None):

    def output_name(i):
        if output:
            return output + '.%d.vtk' % i
        else:
            return None

    dim = mesh.vertices.shape[1]
    tol = 1e-4

    # Compute cell volume for latter volume averaging
    mesh.cell_volumes = np.zeros((len(mesh.cells)))
    for i in xrange(len(mesh.cells)):
        mesh.cell_volumes[i] = compute_cell_volume(mesh.vertices[mesh.cells[i]])

    # BCs
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    lx = xmax[0] - xmin[0]
    ly = xmax[1] - xmin[1]

    if dim==2:
        Ceff = np.zeros((3,3))
        contour = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) +
                           (np.abs(mesh.vertices[:,0] - xmax[0]) < tol) +
                           (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) +
                           (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))
        for k in range(3):
            emacro = [0., 0., 0.]
            emacro[k] = 0.01
            bcs=[]
            for p in contour[0]:
                bc = coda.DirichletBC(([p],), ux=emacro[0]*mesh.vertices[p,0]+0.5*emacro[2]*mesh.vertices[p,1],
                                              uy=0.5*emacro[2]*mesh.vertices[p,0]+emacro[1]*mesh.vertices[p,1])
                bcs.append(bc)
            
            gp_data = coda.run(mesh, bcs, materials, output=output_name(k))
            Ceff[k,0] = volumic_average(mesh, gp_data['S11'])
            Ceff[k,1] = volumic_average(mesh, gp_data['S22'])
            Ceff[k,2] = volumic_average(mesh, gp_data['S12'])

        # Thermal
        bcs = [coda.DirichletBC(contour, ux=0, uy=0), ]
        gp_data = coda.run(mesh, bcs, materials, temperature=100.0, output=output_name(6))
        sig = [volumic_average(mesh, gp_data['S11']),
               volumic_average(mesh, gp_data['S22']),
               volumic_average(mesh, gp_data['S12'])]
    else:
        lz = xmax[2] - xmin[2]
        Ceff = np.zeros((6,6))
        contour =  np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) +
                   (np.abs(mesh.vertices[:,0] - xmax[0]) < tol) +
                   (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) +
                   (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) +
                   (np.abs(mesh.vertices[:,2] - xmin[2]) < tol) +
                   (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        # Mechanical
        for k in range(6):
            emacro = [0., 0., 0., 0., 0., 0.]
            emacro[k] = 0.01

            bcs=[]
            for p in contour[0]:
                bc = coda.DirichletBC(([p],), ux=emacro[0]*mesh.vertices[p,0]+0.5*emacro[3]*mesh.vertices[p,1]+0.5*emacro[5]*mesh.vertices[p,2],
                                              uy=0.5*emacro[3]*mesh.vertices[p,0]+emacro[1]*mesh.vertices[p,1]+0.5*emacro[4]*mesh.vertices[p,2],
                                              uz=0.5*emacro[5]*mesh.vertices[p,0]+0.5*emacro[4]*mesh.vertices[p,1]+emacro[2]*mesh.vertices[p,2])
                bcs.append(bc)

            gp_data = coda.run(mesh, bcs, materials, output=output_name(k))
            Ceff[k,0] = volumic_average(mesh, gp_data['S11'])
            Ceff[k,1] = volumic_average(mesh, gp_data['S22'])
            Ceff[k,2] = volumic_average(mesh, gp_data['S33'])
            Ceff[k,3] = volumic_average(mesh, gp_data['S12'])
            Ceff[k,4] = volumic_average(mesh, gp_data['S23'])
            Ceff[k,5] = volumic_average(mesh, gp_data['S31'])

        # Thermal
        bcs = [coda.DirichletBC(contour, ux=0, uy=0, uz=0), ]
        gp_data = coda.run(mesh, bcs, materials, temperature=100.0, output=output_name(3))
        sig = [volumic_average(mesh, gp_data['S11']),
               volumic_average(mesh, gp_data['S22']),
               volumic_average(mesh, gp_data['S33']),
               volumic_average(mesh, gp_data['S12']),
               volumic_average(mesh, gp_data['S23']),
               volumic_average(mesh, gp_data['S31'])]

    # Output
    Ceff = 0.5e2*(Ceff + Ceff.transpose())
    aeff = -0.01 * np.dot(np.linalg.inv(Ceff), sig)

    return Ceff, aeff


def compute_effective_properties_PBC(mesh, materials, output=None):

    def output_name(i):
        if output:
            return output + '.%d.vtk' % i
        else:
            return None

    dim = mesh.vertices.shape[1]
    tol = 1e-6

    # Compute cell volume for latter volume averaging
    mesh.cell_volumes = np.zeros((len(mesh.cells)))
    for i in xrange(len(mesh.cells)):
        mesh.cell_volumes[i] = compute_cell_volume(mesh.vertices[mesh.cells[i]])

    # BCs
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    lx = xmax[0] - xmin[0]
    ly = xmax[1] - xmin[1]

    if dim==2:
        Ceff = np.zeros((3,3))

        southwest = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmin[1]) < tol))
        southeast = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmin[1]) < tol))
        northwest = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))
        northeast = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))

        (west,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol)
                           * (np.abs(mesh.vertices[:,1] - xmin[1]) > tol)
                           * (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (east,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol)
                           * (np.abs(mesh.vertices[:,1] - xmin[1]) > tol)
                           * (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (south,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) < tol)
                            * (np.abs(mesh.vertices[:,0] - xmin[0]) > tol)
                            * (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (north,) = np.where((np.abs(mesh.vertices[:,1] - xmax[1]) < tol)
                            * (np.abs(mesh.vertices[:,0] - xmin[0]) > tol)
                            * (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        east = east[np.argsort(mesh.vertices[east,1])]
        west = west[np.argsort(mesh.vertices[west,1])]
        south = south[np.argsort(mesh.vertices[south,0])]
        north = north[np.argsort(mesh.vertices[north,0])]

        for k in range(3):
            emacro = [0., 0., 0.]
            emacro[k] = 0.01

            mpcs=[]
            for i in xrange(len(west)):
                mpc0 = coda.MPC(west[i], east[i], ux=emacro[0]*lx, uy=0.5*emacro[2]*lx)
                mpcs.append(mpc0)

            for i in xrange(len(south)):
                mpc1 = coda.MPC(south[i], north[i], ux=0.5*emacro[2]*ly, uy=emacro[1]*ly)
                mpcs.append(mpc1)

            bc_southwest = coda.DirichletBC(southwest, ux=0.0, uy=0.0)
            bc_southeast = coda.DirichletBC(southeast, ux=emacro[0]*lx, uy=emacro[2]*lx/2.0)
            bc_northwest = coda.DirichletBC(northwest, ux=emacro[2]*ly/2.0, uy=emacro[1]*ly)
            bc_northeast = coda.DirichletBC(northeast, ux=emacro[0]*lx+emacro[2]*ly/2.0, uy=emacro[2]*lx/2.0+emacro[1]*ly)

            bcs = [bc_southwest, bc_southeast, bc_northwest, bc_northeast]

            gp_data = coda.run(mesh, bcs, materials, mpcs=mpcs, output=output_name(k))
            Ceff[k,0] = volumic_average(mesh, gp_data['S11'])
            Ceff[k,1] = volumic_average(mesh, gp_data['S22'])
            Ceff[k,2] = volumic_average(mesh, gp_data['S12'])
    else:
        lz = xmax[2] - xmin[2]
        Ceff = np.zeros((6,6))

        # corners
        c0 = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                       (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                       (np.abs(mesh.vertices[:,2] - xmin[2]) < tol))

        c1 = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmin[2]) < tol))

        c2 = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmin[2]) < tol))

        c3 = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmin[2]) < tol))

        c4 = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        c5 = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        c6 = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        c7 = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                      (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                      (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        # edges
        (e0,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) < tol) *
                         (np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (e1,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (e2,) = np.where((np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) < tol) *
                         (np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (e3,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (e4,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) < tol) *
                         (np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (e5,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (e6,) = np.where((np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) < tol) *
                         (np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (e7,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (e8,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol))

        (e9,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) < tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol))

        (e10,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) *
                          (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                          (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                          (np.abs(mesh.vertices[:,2] - xmax[2]) > tol))

        (e11,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) *
                          (np.abs(mesh.vertices[:,1] - xmax[1]) < tol) *
                          (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                          (np.abs(mesh.vertices[:,2] - xmax[2]) > tol))

        e0 = e0[np.argsort(mesh.vertices[e0,0])]
        e2 = e2[np.argsort(mesh.vertices[e2,0])]
        e4 = e4[np.argsort(mesh.vertices[e4,0])]
        e6 = e6[np.argsort(mesh.vertices[e6,0])]

        e1 = e1[np.argsort(mesh.vertices[e1,1])]
        e3 = e3[np.argsort(mesh.vertices[e3,1])]
        e5 = e5[np.argsort(mesh.vertices[e5,1])]
        e7 = e7[np.argsort(mesh.vertices[e7,1])]

        e8 = e8[np.argsort(mesh.vertices[e8,2])]
        e9 = e9[np.argsort(mesh.vertices[e9,2])]
        e10 = e10[np.argsort(mesh.vertices[e10,2])]
        e11 = e11[np.argsort(mesh.vertices[e11,2])]

        # faces
        (f0,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) < tol))

        (f1,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) < tol))

        (f2,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmin[0]) < tol))

        (f3,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) < tol))

        (f4,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmin[1]) < tol))

        (f5,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) > tol) *
                         (np.abs(mesh.vertices[:,0] - xmax[0]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmin[2]) > tol) *
                         (np.abs(mesh.vertices[:,2] - xmax[2]) > tol) *
                         (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))

        f0 = f0[np.argsort(mesh.vertices[f0,0]**2 + mesh.vertices[f0,1]**2)]
        f1 = f1[np.argsort(mesh.vertices[f1,0]**2 + mesh.vertices[f1,1]**2)]
        f2 = f2[np.argsort(mesh.vertices[f2,1]**2 + mesh.vertices[f2,2]**2)]
        f3 = f3[np.argsort(mesh.vertices[f3,1]**2 + mesh.vertices[f3,2]**2)]
        f4 = f4[np.argsort(mesh.vertices[f4,0]**2 + mesh.vertices[f4,2]**2)]
        f5 = f5[np.argsort(mesh.vertices[f5,0]**2 + mesh.vertices[f5,2]**2)]

        for k in range(6):
            emacro = [0., 0., 0., 0., 0., 0.]
            emacro[k] = 0.01

            bc_c0 = coda.DirichletBC(c0, ux=0.0, uy=0.0, uz=0.0)
            bc_c1 = coda.DirichletBC(c1, ux=emacro[0]*lx, uy=0.5*emacro[3]*lx, uz=0.5*emacro[5]*lx)
            bc_c2 = coda.DirichletBC(c2, ux=emacro[0]*lx+0.5*emacro[3]*ly, uy=0.5*emacro[3]*lx+emacro[1]*ly, uz=0.5*emacro[5]*lx+0.5*emacro[4]*ly)
            bc_c3 = coda.DirichletBC(c3, ux=0.5*emacro[3]*ly, uy=emacro[1]*ly, uz=0.5*emacro[4]*ly)

            bc_c4 = coda.DirichletBC(c4, ux=0.5*emacro[5]*lz, uy=0.5*emacro[4]*lz, uz=emacro[2]*lz)
            bc_c5 = coda.DirichletBC(c5, ux=emacro[0]*lx+0.5*emacro[5]*lz, uy=0.5*emacro[3]*lx+0.5*emacro[4]*lz, uz=0.5*emacro[5]*lx+emacro[2]*lz)
            bc_c6 = coda.DirichletBC(c6, ux=emacro[0]*lx+0.5*emacro[3]*ly+0.5*emacro[5]*lz, uy=0.5*emacro[3]*lx+emacro[1]*ly+0.5*emacro[4]*lz, uz=0.5*emacro[5]*lx+0.5*emacro[4]*ly+emacro[2]*lz)
            bc_c7 = coda.DirichletBC(c7, ux=0.5*emacro[3]*ly+0.5*emacro[5]*lz, uy=emacro[1]*ly+0.5*emacro[4]*lz, uz=0.5*emacro[4]*ly+emacro[2]*lz)

            bcs = [bc_c0, bc_c1, bc_c2, bc_c3,
                   bc_c4, bc_c5, bc_c6, bc_c7]

            mpcs=[]
            for i in xrange(len(e0)):
                mpc2 = coda.MPC(e0[i], e2[i], ux=0.5*emacro[3]*ly, uy=emacro[1]*ly, uz=0.5*emacro[4]*ly)
                mpc4 = coda.MPC(e0[i], e4[i], ux=0.5*emacro[5]*lz, uy=0.5*emacro[4]*lz, uz=emacro[2]*lz)
                mpc6 = coda.MPC(e0[i], e6[i], ux=0.5*emacro[3]*ly+0.5*emacro[5]*lz, uy=emacro[1]*ly+0.5*emacro[4]*lz, uz=0.5*emacro[4]*ly+emacro[2]*lz)
                mpcs.append(mpc2)
                mpcs.append(mpc4)
                mpcs.append(mpc6)
            for i in xrange(len(e1)):
                mpc3 = coda.MPC(e1[i], e3[i], ux=-emacro[0]*lx, uy=-0.5*emacro[3]*lx, uz=-0.5*emacro[5]*lx)
                mpc5 = coda.MPC(e1[i], e5[i], ux=0.5*emacro[5]*lz, uy=0.5*emacro[4]*lz, uz=emacro[2]*lz)
                mpc7 = coda.MPC(e1[i], e7[i], ux=-emacro[0]*lx+0.5*emacro[5]*lz, uy=-0.5*emacro[3]*lx+0.5*emacro[4]*lz, uz=-0.5*emacro[5]*lx+emacro[2]*lz)
                mpcs.append(mpc3)
                mpcs.append(mpc5)
                mpcs.append(mpc7)
            for i in xrange(len(e8)):
                mpc9 = coda.MPC(e8[i], e9[i], ux=emacro[0]*lx, uy=0.5*emacro[3]*lx, uz=0.5*emacro[5]*lx)
                mpc10 = coda.MPC(e8[i], e10[i], ux=emacro[0]*lx+0.5*emacro[3]*ly, uy=0.5*emacro[3]*lx+emacro[1]*ly, uz=0.5*emacro[5]*lx+0.5*emacro[4]*ly)
                mpc11 = coda.MPC(e8[i], e11[i], ux=0.5*emacro[3]*ly, uy=emacro[1]*ly, uz=0.5*emacro[4]*ly)
                mpcs.append(mpc9)
                mpcs.append(mpc10)
                mpcs.append(mpc11)

            for i in xrange(len(f0)):
                mpc0 = coda.MPC(f0[i], f1[i], ux=0.5*emacro[5]*lz, uy=0.5*emacro[4]*lz, uz=emacro[2]*lz)
                mpcs.append(mpc0)
            for i in xrange(len(f2)):
                mpc2 = coda.MPC(f2[i], f3[i],  ux=emacro[0]*lx, uy=0.5*emacro[3]*lx, uz=0.5*emacro[5]*lx)
                mpcs.append(mpc2)
            for i in xrange(len(f4)):
                mpc2 = coda.MPC(f4[i], f5[i],  ux=0.5*emacro[3]*ly, uy=emacro[1]*ly, uz=0.5*emacro[4]*ly)
                mpcs.append(mpc2)
            
            uguess = []
            for p in xrange(len(mesh.vertices)):
                ux=emacro[0]*mesh.vertices[p,0]+0.5*emacro[3]*mesh.vertices[p,1]+0.5*emacro[5]*mesh.vertices[p,2]
                uy=0.5*emacro[3]*mesh.vertices[p,0]+emacro[1]*mesh.vertices[p,1]+0.5*emacro[4]*mesh.vertices[p,2]
                uz=0.5*emacro[5]*mesh.vertices[p,0]+0.5*emacro[4]*mesh.vertices[p,1]+emacro[2]*mesh.vertices[p,2]
                uguess.append((ux,uy,uz))
            uguess = np.asarray(uguess).ravel()

            gp_data = coda.run(mesh, bcs, materials, mpcs=mpcs, output=output_name(k), prev=uguess)
            Ceff[k,0] = volumic_average(mesh, gp_data['S11'])
            Ceff[k,1] = volumic_average(mesh, gp_data['S22'])
            Ceff[k,2] = volumic_average(mesh, gp_data['S33'])
            Ceff[k,3] = volumic_average(mesh, gp_data['S12'])
            Ceff[k,4] = volumic_average(mesh, gp_data['S23'])
            Ceff[k,5] = volumic_average(mesh, gp_data['S31'])

        # Thermal case:
        bc_c0 = coda.DirichletBC(c0, ux=0.0, uy=0.0, uz=0.0)
        bc_c1 = coda.DirichletBC(c1, ux=0.0, uy=0.0, uz=0.0)
        bc_c2 = coda.DirichletBC(c2, ux=0.0, uy=0.0, uz=0.0)
        bc_c3 = coda.DirichletBC(c3, ux=0.0, uy=0.0, uz=0.0)

        bc_c4 = coda.DirichletBC(c4, ux=0.0, uy=0.0, uz=0.0)
        bc_c5 = coda.DirichletBC(c5, ux=0.0, uy=0.0, uz=0.0)
        bc_c6 = coda.DirichletBC(c6, ux=0.0, uy=0.0, uz=0.0)
        bc_c7 = coda.DirichletBC(c7, ux=0.0, uy=0.0, uz=0.0)

        bcs = [bc_c0, bc_c1, bc_c2, bc_c3,
               bc_c4, bc_c5, bc_c6, bc_c7]

        mpcs=[]
        for i in xrange(len(e0)):
            mpc2 = coda.MPC(e0[i], e2[i], ux=0.0, uy=0.0, uz=0.0)
            mpc4 = coda.MPC(e0[i], e4[i], ux=0.0, uy=0.0, uz=0.0)
            mpc6 = coda.MPC(e0[i], e6[i], ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc2)
            mpcs.append(mpc4)
            mpcs.append(mpc6)
        for i in xrange(len(e1)):
            mpc3 = coda.MPC(e1[i], e3[i], ux=0.0, uy=0.0, uz=0.0)
            mpc5 = coda.MPC(e1[i], e5[i], ux=0.0, uy=0.0, uz=0.0)
            mpc7 = coda.MPC(e1[i], e7[i], ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc3)
            mpcs.append(mpc5)
            mpcs.append(mpc7)
        for i in xrange(len(e8)):
            mpc9 = coda.MPC(e8[i], e9[i], ux=0.0, uy=0.0, uz=0.0)
            mpc10 = coda.MPC(e8[i], e10[i], ux=0.0, uy=0.0, uz=0.0)
            mpc11 = coda.MPC(e8[i], e11[i], ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc9)
            mpcs.append(mpc10)
            mpcs.append(mpc11)

        for i in xrange(len(f0)):
            mpc0 = coda.MPC(f0[i], f1[i], ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc0)
        for i in xrange(len(f2)):
            mpc2 = coda.MPC(f2[i], f3[i],  ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc2)
        for i in xrange(len(f4)):
            mpc2 = coda.MPC(f4[i], f5[i],  ux=0.0, uy=0.0, uz=0.0)
            mpcs.append(mpc2)

        gp_data = coda.run(mesh, bcs, materials, mpcs=mpcs, temperature=100.0, output=output_name(6))
        sig = [volumic_average(mesh, gp_data['S11']),
               volumic_average(mesh, gp_data['S22']),
               volumic_average(mesh, gp_data['S33']),
               volumic_average(mesh, gp_data['S12']),
               volumic_average(mesh, gp_data['S23']),
               volumic_average(mesh, gp_data['S31'])]

    # Output
    Ceff = 0.5e2*(Ceff + Ceff.transpose())
    aeff = -0.01 * np.dot(np.linalg.inv(Ceff), sig)

    return Ceff, aeff


def impose_periodic_strain(mesh, materials, emacro, output=None, verbose=False):

    dim = mesh.vertices.shape[1]
    tol = 1e-6

    # BCs
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    lx = xmax[0] - xmin[0]
    ly = xmax[1] - xmin[1]

    if dim==2:
        Ceff = np.zeros((3,3))

        southwest = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmin[1]) < tol))
        southeast = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmin[1]) < tol))
        northwest = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))
        northeast = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol) * (np.abs(mesh.vertices[:,1] - xmax[1]) < tol))

        (west,) = np.where((np.abs(mesh.vertices[:,0] - xmin[0]) < tol)
                * (np.abs(mesh.vertices[:,1] - xmin[1]) > tol)
                * (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (east,) = np.where((np.abs(mesh.vertices[:,0] - xmax[0]) < tol)
                * (np.abs(mesh.vertices[:,1] - xmin[1]) > tol)
                * (np.abs(mesh.vertices[:,1] - xmax[1]) > tol))

        (south,) = np.where((np.abs(mesh.vertices[:,1] - xmin[1]) < tol)
                * (np.abs(mesh.vertices[:,0] - xmin[0]) > tol)
                * (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        (north,) = np.where((np.abs(mesh.vertices[:,1] - xmax[1]) < tol)
                * (np.abs(mesh.vertices[:,0] - xmin[0]) > tol)
                * (np.abs(mesh.vertices[:,0] - xmax[0]) > tol))

        east = east[np.argsort(mesh.vertices[east,1])]
        west = west[np.argsort(mesh.vertices[west,1])]
        south = south[np.argsort(mesh.vertices[south,0])]
        north = north[np.argsort(mesh.vertices[north,0])]

        mpcs=[]
        for i in xrange(len(west)):
            mpc0 = coda.MPC(west[i], east[i], ux=emacro[0]*lx, uy=0.5*emacro[2]*lx)
            mpcs.append(mpc0)

        for i in xrange(len(south)):
            mpc1 = coda.MPC(south[i], north[i], ux=0.5*emacro[2]*ly, uy=emacro[1]*ly)
            mpcs.append(mpc1)

        bc_southwest = coda.DirichletBC(southwest, ux=0.0, uy=0.0)
        bc_southeast = coda.DirichletBC(southeast, ux=emacro[0]*lx, uy=emacro[2]*lx/2.0)
        bc_northwest = coda.DirichletBC(northwest, ux=emacro[2]*ly/2.0, uy=emacro[1]*ly)
        bc_northeast = coda.DirichletBC(northeast, ux=emacro[0]*lx+emacro[2]*ly/2.0, uy=emacro[2]*lx/2.0+emacro[1]*ly)
        bcs = [bc_southwest, bc_southeast, bc_northwest, bc_northeast]
        gp_data = coda.run(mesh, bcs, materials, mpcs=mpcs, output=output, verbose=verbose)

    else:
        print 'Not implemented in dim=3'
        sys.exit(1)

    return gp_data
