import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys, h5py

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen
from turbulence.utils.ft import ift_sh_phase_screen_ea as ift_sh_phase_screen

filename = '/home/jamunoz/git/turbulence/data/pristine_010079.hdf5'
f = h5py.File(filename, 'r')
Uin = f['PUMP']['Data'][0,:,:]
image=Uin


vmax = int(np.amax(Uin)) + 1
#vmin = 0

#population = np.load('/home/jamunoz/git/turbulence/s_population_serial.npz')
population = np.load('/home/jamunoz/git/turbulence/t_population16.npz')

image = population['arr_0'][39]
#print(population['arr_0'][39])
#print(dir(population.f))
#population['arr_0'][0]

vmax = int(np.max(image))+100
#vmin = int(np.min(image))-100
vmin = 0

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

verbose = True

#phase screen parameter dictionary, hard-coded, WIP
phz_params_dict = {}
phz_params_dict['N'] = N
phz_params_dict['Lout'] = Lout
phz_params_dict['Lin'] = Lin
phz_params_dict['deltax'] = deltax
phz_params_dict['wvl'] = wvl
phz_params_dict['Dz'] = Dz
phz_params_dict['nscreen'] = nscreen
phz_params_dict['kpow'] = kpow
phz_params_dict['Rytov'] = Rytov
phz_params_dict['Np'] = Np

cmap = 'binary'
panel_title_dict = {
    'ul': 'Pristine',
    'ur': 'Simulated turbulence',
    'll': 'Evolutionary algorithm',
    'lr': 'Corrected'
}

image_wrapped = population['arr_0'][3]
#image_wrapped = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)
#image_wrapped = abs(image_wrapped)

guess = population['arr_0'][39]
#guess = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=population['arr_0'][39])

image_unwrapped = population['arr_0'][79]
#image_unwrapped = abs(guess)

#diff = abs(image_wrapped - image_unwrapped)

#T = image_unwrapped - Uin

corrected = population['arr_0'][50]
#corrected = image_wrapped - T 

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

fig.colorbar(ax1.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax1)
ax1.set_title(panel_title_dict['ul'])

fig.colorbar(ax2.imshow(image_wrapped, cmap=cmap, vmin=vmin, vmax=vmax),ax=ax2)
ax2.set_title(panel_title_dict['ur'])

fig.colorbar(ax3.imshow(image_unwrapped, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax3)
ax3.set_title(panel_title_dict['ll'])

fig.colorbar(ax4.imshow(corrected, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax4)
ax4.set_title(panel_title_dict['lr'])
ax4.set_xticks([])
ax4.set_yticks([])

population.close()

filename='_r.png'
plt.savefig(filename)


print('Done.')
