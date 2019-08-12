import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys, h5py

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen
from turbulence.utils.ft import ift_sh_phase_screen_ea as ift_sh_phase_screen

def embody(N, tiles_per_side, genetic_code):
    tile_indices = generate_tile_indices(N, tiles_per_side)
    tile_side = int(N/tiles_per_side)
    canvas = np.zeros([N, N])
    for i in range(len(tile_indices)):
        (x1, x2), (y1, y2) = tile_indices[i]
        canvas[x1:x2, y1:y2] = np.ones([tile_side, tile_side]) * genetic_code[i]

    return canvas

def generate_tile_indices(N, tiles_per_side):
    aList = [0]
    nom = 1
    for i in range(tiles_per_side):
        aList.append(int(nom*N/tiles_per_side))#-1)
        nom += 1

    anotherList = []
    for i in range(len(aList)-1):
        anotherList.append((aList[i], aList[i+1]))

    finalList = []
    for u in anotherList:
        for v in anotherList:
            finalList.append((u, v))

    return finalList

#filename = '/home/jamunoz/git/turbulence/data/pristine_010000.hdf5'
filename = '/Volumes/jorgeamu/AFRL/dump1/hdf5_files/pristine_010000.hdf5'
f = h5py.File(filename, 'r')
Uin = f['PUMP']['Data'][0,:,:]

#phase screen parameter dictionary, hard-coded, WIP
phz_params_dict = {}
phz_params_dict['N'] = 256
phz_params_dict['Lout'] = 10
phz_params_dict['Lin'] = 10e-3
phz_params_dict['deltax'] = 0.005
phz_params_dict['wvl'] = 0.532e-6
phz_params_dict['Dz'] = 10e3
phz_params_dict['nscreen'] = 10
phz_params_dict['kpow'] = 22/6
phz_params_dict['Rytov'] = 0.05
phz_params_dict['Np'] = 5

Uout = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)

Uin = abs(Uin)
Uout = abs(Uout)

image_00 = Uout
image_01 = Uin

#vmax = int(np.amax(Uin)) + 1
#vmin = 0

#population = np.load('/home/jamunoz/git/turbulence/s_population_serial.npz')
#population = np.load('/home/jamunoz/git/turbulence/u_population16.npz')
population = np.load('/Users/jamunoz/OneDrive/git/turbulence/x05_010000_population16.npz')
genetic_code = population['arr_0'][0]
image_11 = embody(256, 16, genetic_code=genetic_code)
image_10 = abs(ft_sh_phase_screen(image_11, phz_params_dict=phz_params_dict, genetic_code=None))

#population = np.load('/home/jamunoz/git/turbulence/t_population32.npz')
population = np.load('/Users/jamunoz/OneDrive/git/turbulence/x05_010000_population32.npz')
genetic_code = population['arr_0'][0]
image_21 = embody(256, 32, genetic_code=genetic_code)
image_20 = abs(ft_sh_phase_screen(image_21, phz_params_dict=phz_params_dict, genetic_code=None))

#population = np.load('/home/jamunoz/git/turbulence/t_population64.npz')
population = np.load('/Users/jamunoz/OneDrive/git/turbulence/x05_010000_population64.npz')
genetic_code = population['arr_0'][0]
image_31 = embody(256, 64, genetic_code=genetic_code)
image_30 = abs(ft_sh_phase_screen(image_31, phz_params_dict=phz_params_dict, genetic_code=None))

#population = np.load('/home/jamunoz/git/turbulence/t_population64.npz')
population = np.load('/Users/jamunoz/OneDrive/git/turbulence/x05_010000_population128.npz')
genetic_code = population['arr_0'][0]
image_41 = embody(256, 128, genetic_code=genetic_code)
image_40 = abs(ft_sh_phase_screen(image_41, phz_params_dict=phz_params_dict, genetic_code=None))

#population = np.load('/home/jamunoz/git/turbulence/t_population128.npz')
#genetic_code = population['arr_0'][0]
#image_lr = embody(256, 32, genetic_code=genetic_code)
#image_lr = image_ll

#print(population['arr_0'][39])
#print(dir(population.f))
#population['arr_0'][0]

#vmax = int(np.max(image))+100
#vmin = int(np.min(image))-100
vmin = 0
vmax = 400 #np.max(np.array(genetic_code))+10

#cmap = 'binary'
cmap = 'gray'
panel_title_dict = {
    '00': 'Input',
    '01': 'Pristine image',
    '10': 'Turbulent',
    '11': 'Output after 16 x 16',
    '20': 'Turbulent',
    '21': 'Output after 32 x 32',
    '30': 'Turbulent',
    '31': 'Output after 64 x 64',
    '40': 'Turbulent',
    '41': 'Output after 128 x 128'   
}

#gnetic_code = population['arr_0'][3]
#image_wrapped = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)
#image_wrapped = abs(image_wrapped)

#guess = population['arr_0'][39]
#guess = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=population['arr_0'][39])

#image_unwrapped = population['arr_0'][79]
#image_unwrapped = abs(guess)

#diff = abs(image_wrapped - image_unwrapped)

#T = image_unwrapped - Uin

#corrected = population['arr_0'][50]
#corrected = image_wrapped - T 


fig, ax = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10, 20))
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = ax.ravel()

fig.colorbar(ax1.imshow(image_00, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax1)
ax1.set_title(panel_title_dict['00'])

fig.colorbar(ax2.imshow(image_01, cmap=cmap, vmin=vmin, vmax=vmax),ax=ax2)
ax2.set_title(panel_title_dict['01'])

fig.colorbar(ax3.imshow(image_10, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax3)
ax3.set_title(panel_title_dict['10'])

fig.colorbar(ax4.imshow(image_11, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax4)
ax4.set_title(panel_title_dict['11'])
ax4.set_xticks([])
ax4.set_yticks([])

fig.colorbar(ax5.imshow(image_20, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax5)
ax5.set_title(panel_title_dict['20'])

fig.colorbar(ax6.imshow(image_21, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax6)
ax6.set_title(panel_title_dict['21'])

fig.colorbar(ax7.imshow(image_30, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax7)
ax7.set_title(panel_title_dict['30'])

fig.colorbar(ax8.imshow(image_31, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax8)
ax8.set_title(panel_title_dict['31'])

fig.colorbar(ax9.imshow(image_40, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax9)
ax9.set_title(panel_title_dict['40'])

fig.colorbar(ax10.imshow(image_41, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax10)
ax10.set_title(panel_title_dict['41'])


population.close()

filename='_x05_010000.png'
plt.savefig(filename)


print('Done.')
