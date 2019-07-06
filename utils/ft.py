import numpy as np

from numpy.fft import ifftshift, ifft2, fftshift, fft2
from numpy import exp, pi, mean, var, std, abs, sin, cos, sum

def ft_sh_phase_screen(Uin, N=512, Lout=10, Lin=1e-3, deltax=0.005, wvl=0.532e-6, Dz=50e3, nscreen=50, kpow=22/6, Rytov=0.2, Np=5):
    k=2*pi/wvl
    nn=np.arange(-np.floor(N/2), np.floor(N/2))
    nx, ny = np.meshgrid(nn, nn)
    nsq = nx**2 + ny**2
    z=np.linspace(0, Dz, nscreen)

    deltaz=z[1]-z[0]
    deltaf=1/(N*deltax)

    Cn2=Rytov/(1.2287075122549518 * k**(7/6) * Dz**(11/6)) #1.23 for Kolmogorov
    r0 = (.423*k**2*Cn2*Dz)**(-3/5)

    fx=nx*deltaf
    fy=ny*deltaf
    fsq=fx**2+fy**2

    Uin=Uin #np.ones([N, N])
    g=Uin

    for idx in range(nscreen-1):
        fm = 5.92/Lin/(2*pi)
        f0=1/Lout

        PSD_phi = 0.023 * r0**(-5/3) * exp(-(fsq/fm**2)) / (fsq + f0**2)**(kpow/2)
        PSD_phi[int(N/2),int(N/2)] = 0

        cnm = (np.random.randn(N, N) + 1j*np.random.randn(N, N) ) * np.sqrt(PSD_phi)*deltaf
        phz_hi = np.real(ifftshift(ifft2(ifftshift(cnm)))*(N*1)**2)
        phz_lo = np.zeros([N, N])
        #phz_lo = subharmonics(Np, phz_hi)
        phz = phz_hi + phz_lo

        Q2 = exp(-1j*pi**2*2*deltaz/k*fsq)
        G = Q2*fftshift(fft2(fftshift(g)))*deltax**2
        g = ifftshift(ifft2(ifftshift(G)))*(N*deltaf)**2

        g = exp(1j*phz) * g

        Uout = g
        return Uout
