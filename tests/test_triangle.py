import gencell.triangle as tri

vertices = [(0, 0), (0, 1), (1, 1), (1, 0)]
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

m = tri.triangle(vertices, edges)
m.save('square.0.mesh')

m = tri.triangle(vertices, edges, quality=True, max_area=0.005, min_angle=30.0)
m.save('square.1.mesh')
