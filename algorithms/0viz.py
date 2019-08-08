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

#filename = '/home/jamunoz/git/turbulence/data/pristine_010079.hdf5'
#f = h5py.File(filename, 'r')
#Uin = f['PUMP']['Data'][0,:,:]
#image=Uin

#vmax = int(np.amax(Uin)) + 1
#vmin = 0

#population = np.load('/home/jamunoz/git/turbulence/s_population_serial.npz')
population = np.load('/home/jamunoz/git/turbulence/t_population16.npz')
genetic_code = population['arr_0'][0]
image_ul = embody(256, 16, genetic_code=genetic_code)

population = np.load('/home/jamunoz/git/turbulence/t_population32.npz')
genetic_code = population['arr_0'][0]
image_ur = embody(256, 32, genetic_code=genetic_code)

population = np.load('/home/jamunoz/git/turbulence/t_population64.npz')
genetic_code = population['arr_0'][0]
image_ll = embody(256, 64, genetic_code=genetic_code)

#population = np.load('/home/jamunoz/git/turbulence/t_population128.npz')
#genetic_code = population['arr_0'][0]
#image_lr = embody(256, 32, genetic_code=genetic_code)
image_lr = image_ll

#print(population['arr_0'][39])
#print(dir(population.f))
#population['arr_0'][0]

#vmax = int(np.max(image))+100
#vmin = int(np.min(image))-100
vmin = 0
vmax = 400 #np.max(np.array(genetic_code))+10

cmap = 'binary'
panel_title_dict = {
    'ul': 'Pristine',
    'ur': 'Simulated turbulence',
    'll': 'Evolutionary algorithm',
    'lr': 'Corrected'
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

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()

fig.colorbar(ax1.imshow(image_ul, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax1)
ax1.set_title(panel_title_dict['ul'])

fig.colorbar(ax2.imshow(image_ur, cmap=cmap, vmin=vmin, vmax=vmax),ax=ax2)
ax2.set_title(panel_title_dict['ur'])

fig.colorbar(ax3.imshow(image_ll, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax3)
ax3.set_title(panel_title_dict['ll'])

fig.colorbar(ax4.imshow(image_lr, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax4)
ax4.set_title(panel_title_dict['lr'])
ax4.set_xticks([])
ax4.set_yticks([])

population.close()

filename='_s.png'
plt.savefig(filename)


print('Done.')
