import gencell
import numpy as np

m = gencell.meshutils.read_mesh('meshes/cube.mesh')

xmin = np.min(m.vertices, axis=0)
xmax = np.max(m.vertices, axis=0)

tol = 1e-6
nset_xmin = np.where(np.abs(m.vertices[:,0] - xmin[0]) < tol)
nset_xmax = np.where(np.abs(m.vertices[:,0] - xmax[0]) < tol)
nset_ymin = np.where(np.abs(m.vertices[:,1] - xmin[1]) < tol)
nset_ymax = np.where(np.abs(m.vertices[:,1] - xmax[1]) < tol)
nset_zmin = np.where(np.abs(m.vertices[:,2] - xmin[2]) < tol)
nset_zmax = np.where(np.abs(m.vertices[:,2] - xmax[2]) < tol)

mat = [(gencell.feutils.isotropic_stiffness(E=100e3, nu=0.2),
        gencell.feutils.isotropic_thermal_expansion(alpha=1e-6)),
       ]

bcs = [gencell.coda.DirichletBC(nset_xmin, ux=0),
       gencell.coda.DirichletBC(nset_xmax, ux=1.0),
       gencell.coda.DirichletBC(nset_ymin, uy=0),
       gencell.coda.DirichletBC(nset_ymax, uy=0),
       gencell.coda.DirichletBC(nset_zmin, uz=0),
       gencell.coda.DirichletBC(nset_zmax, uz=0)
       ]

results = gencell.coda.run(m, bcs, mat, verbose=True, output='test_coda')
