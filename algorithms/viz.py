import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys, h5py

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen
from turbulence.utils.ft import ift_sh_phase_screen_ea as ift_sh_phase_screen

filename = '/home/jamunoz/git/turbulence/data/pristine_010000.hdf5'
f = h5py.File(filename, 'r')
Uin = f['PUMP']['Data'][0,:,:]
image=Uin

vmax = int(np.amax(Uin)) + 1
vmin = 0

population = np.load('/home/jamunoz/git/turbulence/population.npz')

#print(dir(population.f))
#population['arr_0'][0]

N = 256
Lout = 10
Lin = 10e-3
deltax = 0.005
wvl = 0.532e-6
Dz = 10e3
nscreen = 10
kpow = 22/6
Rytov = 0.02
Np = 5

image_wrapped = ft_sh_phase_screen(Uin, N=N, Lout=Lout, Lin=Lin, deltax=deltax, wvl=wvl, Dz=Dz, nscreen=nscreen, kpow=kpow, Rytov=Rytov, Np=Np, genetic_code=None)

image_wrapped = abs(image_wrapped)

guess = ft_sh_phase_screen(Uin, N=N, Lout=Lout, Lin=Lin, deltax=deltax, wvl=wvl, Dz=Dz, nscreen=nscreen, kpow=kpow, Rytov=Rytov, Np=Np, genetic_code=population['arr_0'][0])

#guess = ift_sh_phase_screen(image_wrapped, N=N, Lout=Lout, Lin=Lin, deltax=deltax, wvl=wvl, Dz=Dz, nscreen=nscreen, kpow=kpow, Rytov=Rytov, Np=Np, genetic_code=population['arr_0'][0])

image_unwrapped = abs(guess)
diff = abs(image_wrapped - image_unwrapped)

T = image_unwrapped - Uin

corrected = image_wrapped - T 

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

fig.colorbar(ax1.imshow(image, cmap='binary', vmin=vmin, vmax=vmax), ax=ax1)
ax1.set_title('Pristine')
ax1.set_xticks([])
ax1.set_yticks([])

fig.colorbar(ax2.imshow(image_wrapped, cmap='binary', vmin=vmin, vmax=vmax),ax=ax2)
ax2.set_title('Simulated turbulence')

fig.colorbar(ax3.imshow(image_unwrapped, cmap='binary', vmin=vmin, vmax=vmax), ax=ax3)
ax3.set_title('Evolutionary algorithm')

fig.colorbar(ax4.imshow(corrected, cmap='binary', vmin=vmin, vmax=vmax), ax=ax4)
ax4.set_title('Corrected')

population.close()

filename='filename3.png'
plt.savefig(filename)


print('Done.')
