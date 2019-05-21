import numpy as np
import scipy.sparse as sp

from fdfdpy.constants import DEFAULT_MATRIX_FORMAT


def createDws_bloch(w, s, dL, N, Kx, Ky, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates the derivative matrices
    # NOTE: python uses C ordering rather than Fortran ordering. Therefore the
    # derivative operators are constructed slightly differently than in MATLAB
    # bloch boundary conditions

    Nx = N[0]
    dx = dL[0]
    if len(N) is not 1:
        Ny = N[1]
        dy = dL[1]
    else: #N is one...
        Ny = 1
        dy = np.inf #...so we zero out dy?
    if w is 'x':
        if s is 'f':
            dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
            Dws = 1/dx*sp.kron(dxf, sp.eye(Ny), format=matrix_format)
        else:
            dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
            Dws = 1/dx*sp.kron(dxb, sp.eye(Ny), format=matrix_format)
    if w is 'y':
        if s is 'f':
            dyf = sp.diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
            Dws = 1/dy*sp.kron(sp.eye(Nx), dyf, format=matrix_format)
        else:
            dyb = sp.diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
            Dws = 1/dy*sp.kron(sp.eye(Nx), dyb, format=matrix_format)
    return Dws


