from fdfdpy.derivatives import createDws
from fdfdpy.constants import DEFAULT_MATRIX_FORMAT
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as la
matrix_format=DEFAULT_MATRIX_FORMAT
import matplotlib.pyplot as plt
#from numpy.linalg import solve as bslash
from scipy.sparse.linalg import spsolve as bslash

def grid_average(center_array, w):
    # computes values at cell edges
    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array

def eigen1D_Efield(wvlen, eps_r, dx, num_modes=10):
    '''
    :param wvlen:
    :param eps_r:
    :param a:
    :param dx:
    :return:
    '''
    L0 = 1e-6;
    eps0 = 8.854e-12 *L0;
    mu0 = 4 * np.pi * 1e-7*L0;
    c0 = 1/np.sqrt(eps0*mu0)

    omega = 2 * np.pi * c0 / wvlen ;

    Nx = len(eps_r);
    dL = dx;
    Dxf = createDws('x', 'f', [dL,1], [Nx,1], matrix_format=matrix_format);
    Dxb = createDws('x', 'b', [dL,1], [Nx,1], matrix_format=matrix_format);
    #Epxx = grid_average(eps_r, 'x');
    #Tepxx = sp.spdiags(Epxx, 0, Nx, Nx, format=matrix_format)
    invTepzz = sp.spdiags(1 /(eps_r), 0, Nx, Nx, format=matrix_format)
    I = sp.identity(Nx, format = matrix_format);

    M = invTepzz;
    C = -invTepzz@(-1j * (Dxf + Dxb));
    K = -invTepzz@(Dxf@Dxb) - omega**2*mu0*eps0*I;

    OB = sp.bmat([[M,None],[None, I]], format = matrix_format);
    OA = sp.bmat([[C,K],[-I, None]], format = matrix_format);

    #print(np.linalg.cond(OA.todense())), conditioning is 10000, which is not amazing...
    #print(type(OB), type(OA))
    # get guess

    neff = np.sqrt(np.max(np.real(eps_r)));
    beta_est = abs(2*np.pi*neff / wvlen);
    sigma = beta_est;
    D = bslash(OB, OA);
    # plt.imshow(np.real(OA.toarray()))
    # plt.show();
    eigenvals, eigenmodes = sp.linalg.eigs(D, k=num_modes, sigma = 0)

    return eigenvals, eigenmodes;
