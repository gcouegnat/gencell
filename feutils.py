import numpy as np

def isotropic_stiffness(E, nu):
    S = 1.0/E * np.array([[1, -nu, -nu, 0, 0, 0],
                          [-nu, 1, -nu, 0, 0, 0],
                          [-nu, -nu, 1, 0, 0, 0],
                          [0, 0, 0, 2*(1+nu), 0, 0],
                          [0, 0, 0, 0, 2*(1+nu), 0],
                          [0, 0, 0, 0, 0, 2*(1+nu)]])

    return np.linalg.inv(S)

def orthotropic_stiffness(E1, E2, E3, nu23, nu13, nu12, G23, G13, G12):
    S = np.array([[1.0/E1  , -nu12/E1, -nu13/E1, 0      , 0      , 0],
                      [-nu12/E1, 1.0/E2  , -nu23/E2, 0      , 0      , 0],
                      [-nu13/E1, -nu23/E2, 1.0/E3  , 0      , 0      , 0],
                      [0       , 0       , 0       , 1.0/G12, 0      , 0],
                      [0       , 0       , 0       , 0      , 1.0/G23, 0],
                      [0       , 0       , 0       , 0      , 0      , 1.0/G13]])
    return np.linalg.inv(S)

def isotropic_thermal_expansion(alpha):
    A = alpha * np.array([1.,1.,1, 0., 0., 0.])
    return A

def orthotropic_thermal_expansion(alpha1, alpha2, alpha3):
    A = np.array([alpha1, alpha2, alpha3, 0., 0., 0.])
    return A


def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    Q = np.array([[c*c, s*s, 0, 0, 0,  np.sqrt(2)*c*s],
                [s*s, c*c, 0, 0, 0, -np.sqrt(2)*c*s],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, -s, 0],
                [0, 0, 0, s, c, 0],
                [-np.sqrt(2)*c*s, np.sqrt(2)*c*s, 0, 0, 0, c*c-s*s]])
    
    return Q

def rotate_stiffness_tensor(C, theta): 
    """ 
    Transform the stiffness tensor C in local coordinate system x to stiffness
    tensor D in global coordinate system X. theta is the angle between X and x.
    """

    Q = rotation_matrix(theta)
    
    # Voigt notation to sqrt(2) notation
    C[3:,3:] *= 2.0
    C[:3,3:] *= np.sqrt(2)
    C[3:,:3] *= np.sqrt(2)

    D = np.dot(Q, np.dot(C, Q.transpose()))

    C[3:,3:] /= 2.0
    C[:3,3:] /= np.sqrt(2)
    C[3:,:3] /= np.sqrt(2)

    D[3:,3:] /= 2.0
    D[:3,3:] /= np.sqrt(2)
    D[3:,:3] /= np.sqrt(2)

    return D

def rotate_dilation_tensor(a, theta):
    Q = rotation_matrix(theta)
    
    a[3:] /= np.sqrt(2)     
    b = np.dot(Q, a)
    
    a[3:] *= np.sqrt(2)
    b[3:] *= np.sqrt(2)

    return b    

