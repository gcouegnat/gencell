import numpy as np
import coda

def compute_cell_volume(coords):
    V = (coords[1][0] * coords[2][1] - coords[2][0] * coords[1][1])
    V += (coords[2][0] * coords[0][1] - coords[0][0] * coords[2][1])
    V += (coords[0][0] * coords[1][1] - coords[1][0] * coords[0][1])
    return 0.5 * V

def volumic_average(mesh, data):
    acc = 0.0
    for i in xrange(len(data)):
        V = compute_cell_volume(mesh.vertices[mesh.cells[i]])
        acc += data[i] * V
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

    # BCs
    xmin = np.min(mesh.vertices, axis=0)
    xmax = np.max(mesh.vertices, axis=0)
    lx = xmax[0] - xmin[0]
    ly = xmax[1] - xmin[1]
    
    left = np.where(np.abs(mesh.vertices[:, 0] - xmin[0]) < 1e-4)
    right = np.where(np.abs(mesh.vertices[:, 0] - xmax[0]) < 1e-4)
    bottom = np.where(np.abs(mesh.vertices[:, 1] - xmin[1]) < 1e-4)
    top = np.where(np.abs(mesh.vertices[:, 1] - xmax[1]) < 1e-4)

    Ceff = np.zeros((3,3))

    # Tensile 1
    bc_left = coda.DirichletBC(left, ux=-0.0005*lx)
    bc_right = coda.DirichletBC(right, ux=0.0005*lx)
    bc_bottom= coda.DirichletBC(bottom, uy=0.0)
    bc_top = coda.DirichletBC(top, uy=0.0)

    bcs = [bc_left, bc_right, bc_bottom, bc_top]
    gp_data = coda.run(mesh, bcs, materials, output=output_name(0))

    Ceff[0,0] = volumic_average(mesh, gp_data['S11'])
    Ceff[0,1] = volumic_average(mesh, gp_data['S22'])
    Ceff[0,2] = volumic_average(mesh, gp_data['S12'])

    # Tensile 2
    bc_left = coda.DirichletBC(left, ux=0.0)
    bc_right = coda.DirichletBC(right, ux=0.0)
    bc_bottom= coda.DirichletBC(bottom, uy=-0.0005*ly)
    bc_top = coda.DirichletBC(top, uy=0.0005*ly)

    bcs = [bc_left, bc_right, bc_bottom, bc_top]
    gp_data = coda.run(mesh, bcs, materials, output=output_name(1))
    Ceff[1,0] = volumic_average(mesh, gp_data['S11'])
    Ceff[1,1] = volumic_average(mesh, gp_data['S22'])
    Ceff[1,2] = volumic_average(mesh, gp_data['S12'])

    # Shear 12
    bc_left = coda.DirichletBC(left, uy=-0.00025*lx)
    bc_right = coda.DirichletBC(right, uy=0.00025*lx)
    bc_bottom= coda.DirichletBC(bottom, ux=-0.00025*ly)
    bc_top = coda.DirichletBC(top, ux=0.00025*ly)

    bcs = [bc_left, bc_right, bc_bottom, bc_top]
    gp_data = coda.run(mesh, bcs, materials,output=output_name(2))
    Ceff[2,0] = volumic_average(mesh, gp_data['S11'])
    Ceff[2,1] = volumic_average(mesh, gp_data['S22'])
    Ceff[2,2] = volumic_average(mesh, gp_data['S12'])

    # Thermal
    bc_left   =  coda.DirichletBC(left  ,  ux=0.0,  uy=0.0)
    bc_right  =  coda.DirichletBC(right ,  ux=0.0,  uy=0.0)
    bc_bottom =  coda.DirichletBC(bottom,  ux=0.0,  uy=0.0)
    bc_top    =  coda.DirichletBC(top   ,  ux=0.0,  uy=0.0)

    bcs = [bc_left, bc_right, bc_bottom, bc_top]
    gp_data = coda.run(mesh, bcs, materials,temperature=1.0, output=output_name(3))
    sig = [ volumic_average(mesh, gp_data['S11']),
            volumic_average(mesh, gp_data['S22']),
            volumic_average(mesh, gp_data['S12'])]

    Ceff = 0.5e3*(Ceff + Ceff.transpose())
    Seff = np.linalg.inv(Ceff)
    aeff = -1.0 * np.dot(Seff, sig)

    return Ceff, aeff
