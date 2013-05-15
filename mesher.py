import numpy as np
from PIL import Image
from scipy.misc import imsave

from meshutils import *
from triangle import *

def _find_contours(img):
    out = np.zeros(img.shape, dtype=np.bool)
    nr,nc = img.shape
    print "First phase"
    for r in range(0,nr):
        for c in range(1,nc-1):
            if (img[r,c]>img[r,c+1] and img[r,c-1] == img[r,c]): 
                out[r,c]=1
            if (img[r,c]==img[r,c+1] and img[r,c-1] < img[r,c]): 
                out[r,c]=1
    print "Second phase"
    for c in range(0,nc):
        for r in range(1,nr-1):
            if (img[r,c]>img[r+1,c] and img[r-1,c] == img[r,c]): 
                out[r,c]=1
            if (img[r,c]==img[r+1,c] and img[r-1,c] < img[r,c]): 
                out[r,c]=1
    return out

def _assemble_contours(img):
    deltas=[(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1)]
    contours = {}
    index = 0
    nx,ny = img.shape
    xx=[0,nx-1]
    yy=[0,ny-1]

    for i in xrange(nx):
        for j in yy: 
            if img[i,j]== True:
                index += 1
                img[i,j]= False
                contours[index]=[(i,j)]
                si, sj = i, j
                next_found = 1
                while next_found:
                    next_found = 0
                    for dx,dy in deltas:
                        ti = max(0, min(si+dx, nx-1))
                        tj = max(0, min(sj+dy, ny-1))
                        if img[ti,tj] == True:
                            contours[index].append((ti,tj))
                            img[ti,tj] = False
                            si, sj = ti, tj
                            next_found = 1
                            break
    for i in xx:
        for j in xrange(ny): 
            if img[i,j]== True:
                index += 1
                img[i,j]= False
                contours[index]=[(i,j)]
                si, sj = i, j
                next_found = 1
                while next_found:
                    next_found = 0
                    for dx,dy in deltas:
                        ti = max(0, min(si+dx, nx-1))
                        tj = max(0, min(sj+dy, ny-1))
                        if img[ti,tj] == True:
                            contours[index].append((ti,tj))
                            img[ti,tj] = False
                            si, sj = ti, tj
                            next_found = 1
                            break
    for i in xrange(1,nx-1):
        for j in xrange(1,ny-1): 
            if img[i,j]== True:
                index += 1
                img[i,j]= False
                contours[index]=[(i,j)]
                si, sj = i, j
                next_found = 1
                while next_found:
                    next_found = 0
                    for dx,dy in deltas:
                        ti = max(0, min(si+dx, nx-1))
                        tj = max(0, min(sj+dy, ny-1))
                        if img[ti,tj] == True:
                            contours[index].append((ti,tj))
                            img[ti,tj] = False
                            si, sj = ti, tj
                            next_found = 1
                            break
                contours[index].append(contours[index][0])

    return contours

def _smooth(coords, iterations, closed=False):

    if iterations <= 0:
        return coords

    for it in xrange(iterations):
        depl = coords[1:-1,:] - 0.5 * (coords[0:-2,:] + coords[2:,:])
        coords[1:-1,:] += 0.4 * depl
        depl = coords[1:-1,:] - 0.5 * (coords[0:-2,:] + coords[2:,:])
        coords[1:-1,:] -= 0.416 * depl
        if closed:
            coords[0 ,:] = 0.5*(coords[1,:] + coords[-2,:])
            coords[-1,:] = coords[0,:]

    return coords

def _approximate_polygon(coords, tolerance):
    if tolerance <= 0:
        return coords
    chain = np.zeros(coords.shape[0], 'bool')
    # pre-allocate distance array for all points
    dists = np.zeros(coords.shape[0])
    chain[0] = True
    chain[-1] = True
    pos_stack = [(0, chain.shape[0] - 1)]
    end_of_chain = False
    while not end_of_chain:
        start, end = pos_stack.pop()
        # determine properties of current line segment
        r0, c0 = coords[start, :]
        r1, c1 = coords[end, :]
        dr = r1 - r0
        dc = c1 - c0
        segment_angle = - np.arctan2(dr, dc)
        segment_dist = c0 * np.sin(segment_angle) + r0 * np.cos(segment_angle)
        # select points in-between line segment
        segment_coords = coords[start + 1:end, :]
        segment_dists = dists[start + 1:end]
        # check whether to take perpendicular or euclidean distance with
        # inner product of vectors
        # vectors from points -> start and end
        dr0 = segment_coords[:, 0] - r0
        dc0 = segment_coords[:, 1] - c0
        dr1 = segment_coords[:, 0] - r1
        dc1 = segment_coords[:, 1] - c1
        # vectors points -> start and end projected on start -> end vector
        projected_lengths0 = dr0 * dr + dc0 * dc
        projected_lengths1 = - dr1 * dr - dc1 * dc
        perp = np.logical_and(projected_lengths0 > 0,
                              projected_lengths1 > 0)
        eucl = np.logical_not(perp)
        segment_dists[perp] = np.abs(
            segment_coords[perp, 0] * np.cos(segment_angle)
            + segment_coords[perp, 1] * np.sin(segment_angle)
            - segment_dist
        )
        segment_dists[eucl] = np.minimum(
            # distance to start point
            np.sqrt(dc0[eucl] ** 2 + dr0[eucl] ** 2),
            # distance to end point
            np.sqrt(dc1[eucl] ** 2 + dr1[eucl] ** 2)
        )
        if np.any(segment_dists > tolerance):
            # select point with maximum distance to line
            new_end = start + np.argmax(segment_dists) + 1
            pos_stack.append((new_end, end))
            pos_stack.append((start, new_end))
            chain[new_end] = True
        if len(pos_stack) == 0:
            end_of_chain = True
    return coords[chain, :]

def create_mesh_from_image(img, smooth_iter=1000, tolerance=1.3, max_area = None):

    def connect(start,end):
        result=[]
        for i in range(start,end):
            result.append((i,i+1))
        return result

    def connect_round_trip(start, end):
        result=[]
        for i in range(start,end):
            result.append((i, i+1))
        result.append((end,start))
        return result

    edges=[]
    vertices=[]
    nx,ny = img.shape

    print '*** Detecting image contours'
    bw = _find_contours(img)

    print '*** Assembling contours'
    contours = _assemble_contours(bw)

    print '*** Approximating pixels chains'
    for contour in contours.itervalues():
        pts = np.array(contour, 'float')
        if pts.shape[0] > 4:
            closed = False
            if abs(pts[0,0]-pts[-1,0]) < 1e-6 and abs(pts[0,1]-pts[-1,1]) < 1e-6:
                closed = True
            pts = _smooth(pts, iterations=smooth_iter, closed=closed)
            pts = _approximate_polygon(pts, tolerance=tolerance)
            if pts.shape[0] > 2:
                if closed :
                    edges.extend(connect_round_trip(len(vertices), len(vertices) + len(pts) -2))
                    vertices.extend(pts[:-1,:].tolist())
                else:
                    edges.extend(connect(len(vertices), len(vertices) + len(pts) - 1))
                    vertices.extend(pts.tolist())

    edges.extend(connect_round_trip(len(vertices), len(vertices) + 3))
    vertices.append((0.0,0.0))
    vertices.append((float(nx-1),0.0))
    vertices.append((float(nx-1),float(ny-1)))
    vertices.append((0.0, float(ny-1)))



    print '*** Meshing'
    mesh = triangle(vertices, edges, min_angle=30, max_area=max_area)
   
    print("*** Assigning materials")
    ids=[]
    for i in xrange(len(mesh.cells)):
        xc = 0.333333333*(mesh.vertices[mesh.cells[i][0]][0]
            + mesh.vertices[mesh.cells[i][1]][0] 
            + mesh.vertices[mesh.cells[i][2]][0])
        yc = 0.333333333*(mesh.vertices[mesh.cells[i][0]][1]+mesh.vertices[mesh.cells[i][1]][1]+mesh.vertices[mesh.cells[i][2]][1])
        ids.append(img[int(xc), int(yc)])
    mesh.set_cell_markers(ids)

    mesh.remove_cells_with_marker(0)
    mesh.renumber_cell_markers(reverse=True)


    return mesh


if __name__ == '__main__':


    image=Image.open('image.png')
    img=np.array(image)
    mesh = create_mesh_from_image(img, smooth_iter=100, tolerance=1.25)

    mesh.save('cell.mesh')
    mesh.save('cell.vtk')
    
    #bw = _find_contours(img)
    #imsave('bw.png', bw)
    
