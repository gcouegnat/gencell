import numpy as np
import matplotlib.pyplot as plt

def plot_circle(c,r,color='r',alpha=1.0):
    theta=np.linspace(0,2*np.pi,50)
    for i in xrange(len(c)):
        cx,cy = c
        px = cx + r*np.cos(theta)
        py = cy + r*np.sin(theta)
        #plt.fill(py,px,color,alpha=alpha)
        plt.plot(py,px,color)

def fill_circle(cx, cy, r, img, val=255):
    sx,sy = img.shape
    #if cx < -r or cx > sx + r or cy < r or cy > sy + r:
        #pass
    #else:
    for i in xrange(max(0, int(cx-r)), min(sx, int(cx+r)+1)):
        for j in xrange(max(0, int(cy-r)), min(sy, int(cy+r)+1)):
            if ((cx-i)**2 + (cy-j)**2) < r*r:
                img[i,j] = val
