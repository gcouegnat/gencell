from gencell.meshutils import read_mesh
from gencell.feutils import isotropic_stiffness, isotropic_thermal_expansion
from gencell.fetools import compute_effective_properties_PMUBC

m = read_mesh('meshes/cube.mesh')

mat = [(isotropic_stiffness(E=100e3, nu=0.2),
        isotropic_thermal_expansion(alpha=1.0e-6)),
       ]

Ceff, aeff = compute_effective_properties_PMUBC(m, mat)
print(Ceff)
print(aeff)
