from fdfdpy.derivatives import createDws
from fdfdpy.constants import DEFAULT_MATRIX_FORMAT
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse.linalg import spsolve as bslash


matrix_format=DEFAULT_MATRIX_FORMAT
matrix_format = 'csc'
def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array

def eigenTE_Kx_Ky(wvlen, eps_r,  dL, Ky, num_modes = 10):
    '''
    the eigenvalues should be bounded due to periodicity, i.e. the system should have some degeneracy
    with respect to the bloch wavevectors...
    :param wvlen:
    :param eps_r:
    :param a:
    :param dx:
    :return:
    '''
    L0 = 1e-6;
    eps0 = 8.854e-12 * L0;
    mu0 = np.pi * 4e-7 * L0;
    c0 = 1 / np.sqrt(eps0 * mu0);

    #print(c0, mu0, eps0)
    omega = 2*np.pi*c0/wvlen;
    N = eps_r.shape;
    M = np.prod(N);
    Dxf = createDws('x', 'f', dL, N, matrix_format=matrix_format);
    Dxb = createDws('x', 'b', dL, N, matrix_format=matrix_format);
    Dyf = createDws('y', 'f', dL, N, matrix_format=matrix_format);
    Dyb = createDws('y', 'b', dL, N, matrix_format=matrix_format);

    # Epxx = grid_average(eps_r, 'x');
    # Epyy = grid_average(eps_r, 'y');
    #Tez =  sp.spdiags(np.diag(eps0*eps_r), 0, M,M, format = matrix_format)
    invTepzz = sp.spdiags(1 / eps_r.flatten(), 0, M,M, format=matrix_format)
    I = sp.identity(M, format = matrix_format);

    K = invTepzz@(-Dxf @ Dxb - Dyf @ Dyb - 1j*((Dyf + Dyb))*Ky + Ky**2*I) - omega ** 2*eps0 *mu0*I ;
    M = invTepzz;
    C = -invTepzz@(1j * (Dxf + Dxb)); #% lambda

    A = sp.bmat([[M, None], [None, I]], format = matrix_format); #A should just be the identity
    B = sp.bmat([[C, K], [-I, None]], format = matrix_format);

    D= bslash(A,B);
    neff = np.sqrt(np.max(np.real(eps_r)));
    beta_est = abs(2*np.pi*neff / wvlen);
    sigma = beta_est;
    sigma = 0;
    #get eigenvalues
    k, modes = sp.linalg.eigs(D, k=num_modes, sigma = sigma)

    return k, modes, A,B;

