from gencell.meshutils import read_mesh
import numpy as np

m = read_mesh('meshes/mesh_with_orphan_nodes.mesh')

print 'Before...'
print '%d vertices' % len(m.vertices)
print '%d unique vertices' % len(np.unique(np.ravel(m.cells)))

m.remove_orphan_vertices()

print 'After...'
print '%d vertices' % len(m.vertices)
print '%d unique vertices' % len(np.unique(np.ravel(m.cells)))

