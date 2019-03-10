### comparison with FDFD 1D
import sys
sys.path.append('D:\\fdfdpy');
from fdfdpy.eigen1D import *
Nx = 400;
eps_r = np.ones((400,))
a = 10;
dx = a/Nx;
wvlen_scan = np.linspace(0.5, 3, 10);
for wvlen in wvlen_scan:
    print(wvlen)
    eigenvals, modes = eigen1D_Efield(wvlen, eps_r,  dx, num_modes=10)
    print(eigenvals)