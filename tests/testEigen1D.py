### comparison with FDFD 1D
import sys
sys.path.append('D:\\fdfdpy');
from fdfdpy.eigen1D import *
import matplotlib.pyplot as plt
Nx = 50;
eps_r = np.ones((Nx,))
eps_r[10:40] = 12;

x = np.linspace(-1/2, 1/2, Nx);
a = 1;
#eps_r = 1+np.sin(2*np.pi*x/a);
dx = a/Nx;
wvlen_scan = np.linspace(0.25, 10, 300);
wvlen_scan = np.logspace(np.log10(0.9), np.log10(10), 2000);
spectra = [];
plt.plot(eps_r);
plt.show()

for wvlen in wvlen_scan:
    print(wvlen)
    eigenvals, modes = eigen1D_Efield(wvlen, eps_r,  dx, num_modes=2)
    spectra.append(eigenvals);

spectra = np.array(spectra);
plt.plot(np.real(spectra), 1/wvlen_scan, '.b')
plt.plot(np.imag(spectra), 1/wvlen_scan, '.r')
plt.legend(('Re(k)', 'Im(k)'))
plt.xlabel('k')
plt.ylabel('$\lambda^{-1}$')
plt.savefig('sample_1D_complex_spectra.png')
plt.show();

