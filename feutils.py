import numpy as np

def isotropic_stiffness(E, nu):
    S = 1.0/E * np.array([[1  , -nu, -nu, 0       , 0       , 0]         ,
                          [-nu, 1  , -nu, 0       , 0       , 0]         ,
                          [-nu, -nu, 1  , 0       , 0       , 0]         ,
                          [0  , 0  , 0  , 2*(1+nu), 0       , 0]         ,
                          [0  , 0  , 0  , 0       , 2*(1+nu), 0]         ,
                          [0  , 0  , 0  , 0       , 0       , 2*(1+nu)]])

    return np.linalg.inv(S)

def transverse_isotropic_stiffness(Et, El, nult, nutt, Glt):
    S = np.array([[1.0/Et      , -nutt/Et, -nult/El, 0              , 0      , 0]        ,
                      [-nutt/Et, 1.0/Et  , -nult/El, 0              , 0      , 0]        ,
                      [-nult/El, -nult/El, 1.0/El  , 0              , 0      , 0]        ,
                      [0       , 0       , 0       , 2*(1+nutt)/Et  , 0      , 0]        ,
                      [0       , 0       , 0       , 0              , 1.0/Glt, 0]        ,
                      [0       , 0       , 0       , 0              , 0      , 1.0/Glt]])
    return np.linalg.inv(S)

def orthotropic_stiffness(E1, E2, E3, nu23, nu13, nu12, G23, G13, G12):
    S = np.array([[1.0/E1      , -nu12/E1, -nu13/E1, 0      , 0      , 0]        ,
                      [-nu12/E1, 1.0/E2  , -nu23/E2, 0      , 0      , 0]        ,
                      [-nu13/E1, -nu23/E2, 1.0/E3  , 0      , 0      , 0]        ,
                      [0       , 0       , 0       , 1.0/G12, 0      , 0]        ,
                      [0       , 0       , 0       , 0      , 1.0/G23, 0]        ,
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

    sqrt2 = np.sqrt(2.0)
    Q = np.array([[c*c      , s*s       , 0, -sqrt2*c*s, 0 , 0]  ,
                  [s*s      , c*c       , 0, sqrt2*c*s , 0 , 0]  ,
                  [0        , 0         , 1, 0         , 0 , 0]  ,
                  [sqrt2*c*s, -sqrt2*c*s, 0, c*c-s*s   , 0 , 0]  ,
                  [0        , 0         , 0, 0         , c , s]  ,
                  [0        , 0         , 0, 0         , -s, c]])
    
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


def transform_matrix(Q):
    R11 = Q[0,0]
    R12 = Q[0,1]
    R13 = Q[0,2]
    
    R21 = Q[1,0]
    R22 = Q[1,1]    
    R23 = Q[1,2]    

    R31 = Q[2,0]
    R32 = Q[2,1]    
    R33 = Q[2,2]    

    sqrt2 = np.sqrt(2.0)

    # [Koay2009, eq.14]
    R = np.array([
        [R11**2       , R12**2       , R13**2       , sqrt2*R11*R12  , sqrt2*R12*R13  , sqrt2*R11*R13]  ,
        [R21**2       , R22**2       , R23**2       , sqrt2*R21*R22  , sqrt2*R22*R23  , sqrt2*R21*R23]  ,
        [R31**2       , R32**2       , R33**2       , sqrt2*R31*R32  , sqrt2*R32*R33  , sqrt2*R31*R33]  ,
        [sqrt2*R11*R21, sqrt2*R12*R22, sqrt2*R13*R22, R11*R22+R12*R21, R12*R23+R13*R22, R11*R23+R13*R21],
        [sqrt2*R21*R31, sqrt2*R22*R32, sqrt2*R23*R33, R21*R32+R22*R31, R22*R33+R23*R32, R21*R33+R23*R31],
        [sqrt2*R11*R31, sqrt2*R12*R32, sqrt2*R13*R33, R11*R32+R12*R31, R12*R33+R13*R32, R11*R33+R13*R31],
    ])

    return R


def tensor2_local_to_global(A, Q):

    R = transform_matrix(Q)

    A[3:] /= np.sqrt(2)
    B = np.dot(R.transpose(), A)

    A[3:] /= np.sqrt(2)
    B[3:] /= np.sqrt(2)

    return B

def tensor4_local_to_global(C, Q):

    R = transform_matrix(Q)
        
    # Add sqrt2
    C[3:,3:] *= 2.0
    C[:3,3:] *= np.sqrt(2)
    C[3:,:3] *= np.sqrt(2)

    D = np.dot(R.transpose(), np.dot(C, R))

    # Remove sqrt2
    C[3:,3:] /= 2.0
    C[:3,3:] /= np.sqrt(2)
    C[3:,:3] /= np.sqrt(2)
    D[3:,3:] /= 2.0
    D[:3,3:] /= np.sqrt(2)
    D[3:,:3] /= np.sqrt(2)
    
    return D
