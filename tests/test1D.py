import sys

sys.path.append('D:\\fdfdpy')

import numpy as np
import matplotlib.pyplot as plt
from fdfdpy.solver1D import *
L0 = 1e-6;

c0 = 3e8;
omega_p = 0.72*np.pi*1e15;
gamma = 5e12;

wvlen_scan = np.linspace(0.8, 4, 400);
epsilon_diel = 16;

a = 0.2; #lattice constant
Nx = 400
eps_r = epsilon_diel*np.ones((Nx, ))
eps_r = eps_r.astype('complex')
print(eps_r.shape)
fill_factor = 0.2;
dx= a/Nx;

kspectra = list();
for wvlen in wvlen_scan:
    omega = 2*np.pi*c0/wvlen/L0;
    #print(omega);
    epsilon_metal = 1-omega_p**2/(omega**2 - 1j*(gamma*omega))
    #print(epsilon_metal)
    eps_r[int(Nx/2-fill_factor*Nx/2): int(Nx/2+fill_factor*Nx/2)] = epsilon_metal;
    # plt.plot(eps_r);
    # plt.show();
    kvals, modes,A = solverTM1D(wvlen, eps_r, a, dx, num_modes = 2);
    # plt.spy(A);
    # plt.show()
    kspectra.append(kvals);

kspectra = np.array(kspectra);

omega_scan = 2*np.pi*c0/wvlen_scan/1e-6

plt.plot(np.real(kspectra), omega_scan/omega_p, '.')
plt.plot(np.imag(kspectra), omega_scan/omega_p, '.')
plt.show();




