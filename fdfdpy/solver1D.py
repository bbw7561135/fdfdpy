from fdfdpy.derivatives import createDws
from fdfdpy.constants import DEFAULT_MATRIX_FORMAT
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as la
matrix_format=DEFAULT_MATRIX_FORMAT

'''
default matrix format should be scipy.sparse
'''

def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array

def solverTM1D(wvlen, eps_r, a, dx, num_modes = 10):
    '''
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
    omega = 2*np.pi*c0/wvlen;
    Nx = len(eps_r);

    dL = dx;
    Dxf = createDws('x', 'f', [dL,1], [Nx,1], matrix_format=matrix_format);
    Dxb = createDws('x', 'b', [dL,1], [Nx,1], matrix_format=matrix_format);
    Epxx = grid_average(eps_r, 'x');
    invTepxx = sp.spdiags(1/(eps0*Epxx), 0, Nx, Nx, format = matrix_format)
    Tepzz = sp.spdiags(eps0*eps_r, 0, Nx,Nx, format = matrix_format)
    A = Tepzz@Dxf@(invTepxx)@Dxb + Tepzz@sp.spdiags(omega**2*mu0*np.ones((Nx,)), 0, Nx,Nx, format = matrix_format);
    A = A.astype('complex')

    # get guess
    neff = np.sqrt(np.max(np.real(eps_r)));
    beta_est = abs(2*np.pi*neff / wvlen);
    sigma = beta_est**2;

    #get eigenvalues
    ksqr, modes = sp.linalg.eigs(A, k=num_modes, sigma = sigma)

    return np.sqrt(ksqr), modes,A;


def solverTE1D(wvlen, eps_r, a, dx, num_modes = 10):
    '''
    :param wvlen:
    :param eps_r:
    :param a:
    :param dx:
    :return:
    '''
    c0 = 3e8;
    mu0 = 4 * np.pi * 1e-7;
    omega = 2 * np.pi * c0 / wvlen / 1e-6;
    
    Nx = a / dx;
    dL = dx;
    I = sp.identity(Nx);
    Dxf = createDws('x', 'f', [dL,1], [Nx,1], matrix_format=matrix_format);
    Dxb = createDws('x', 'f', [dL,1], [Nx,1], matrix_format=matrix_format);
    Epxx = grid_average(eps_r, 'x');
    Tepxx = sp.spdiags(Epxx, 0, Nx, Nx, format=matrix_format)
    Tepzz = sp.spdiags(1 / eps_r, 0, Nx, Nx, format=matrix_format)
    A = Dxf @ Dxb + omega ** 2 * mu0*Tepzz

    #get eigenvalues
    ksqr, modes = la.eigs(A, k=num_modes)

    return np.sqrt(ksqr), modes;



