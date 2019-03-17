### comparison with FDFD 1D
import sys
sys.path.append('D:\\fdfdpy');
from fdfdpy.eigen2D import *
import matplotlib.pyplot as plt
Nx = 20;
Ny = 20;
N = np.array([Nx, Ny]);

eps_r = np.ones(N);

a = np.array([1,1]);
dL = a/N;
radius = 0.25;
## put in a circle;
ci = int(Nx/2); cj= int(Ny/2);

cr = (radius/a[0])*Nx;
I,J=np.meshgrid(np.arange(eps_r.shape[0]),np.arange(eps_r.shape[1]));

print(eps_r.shape)
dist = np.sqrt((I-ci)**2 + (J-cj)**2);
#print(np.where(dist<cr))
eps_r[np.where(dist<cr)] = 12.25;


wvlen_scan = np.linspace(1.01, 20, 100);
wvlen_scan = np.logspace(np.log10(0.9), np.log10(10), 400);

spectra = [];
plt.imshow(eps_r);
plt.show()
Ky = 0;
for wvlen in wvlen_scan:
    print(wvlen)
    eigenvals, modes, A,B = eigenTE_Kx_Ky(wvlen, eps_r,  dL, Ky, num_modes = 2)
    #print(eigenvals)
    spectra.append(eigenvals);

spectra = np.array(spectra);
plt.plot(np.real(spectra), 1/wvlen_scan, '.b')
plt.plot(np.imag(spectra), 1/wvlen_scan, '.r')
plt.legend(('Re(k)', 'Im(k)'))
plt.xlabel('k')
plt.ylabel('$\lambda^{-1}$')
plt.savefig('sample_1D_complex_spectra.png')
plt.show();

