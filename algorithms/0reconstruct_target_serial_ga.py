import sys
import numpy as np
import argparse
import h5py
import time
import matplotlib.pyplot as plt

from numpy import mean
sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen

import scipy as sp
import scipy.ndimage

# Does not need to be a list, but easier to understand first
def generate_orig_population(N, nscreen, Nindividuals):
    population = []
    for i in range(Nindividuals):
        genetic_code = np.zeros([N, N, nscreen, 2])
        for idx in range(nscreen-1):
            genetic_code[:,:,idx,0] = np.random.randn(N, N)
            genetic_code[:,:,idx,1] = np.random.randn(N, N)
																		        
        population.append(genetic_code)

    return population

def generate_orig_population_arr(N, nscreen, Nindividuals):
    population = np.zeros([Nindividuals, N, N, nscreen, 2])
    for ind in range(Nindividuals):
        for idx in range(nscreen-1):
            population[ind,:,:,idx,0] = np.random.randn(N, N)
            population[ind,:,:,idx,1] = np.random.randn(N, N)

    return population

def r_generate_orig_population_arr(N, nscreen, Nindividuals):
    population = np.zeros([Nindividuals, N, N])
    for ind in range(Nindividuals):
        population[ind,:,:] = np.random.randint(512, size=(N, N))

    return population

def pick_top(target, population, Ntop, phz_params_dict):
    N, N = target.shape
    newPopulation = []
    nscreen = phz_params_dict['nscreen']
    Dz = phz_params_dict['Dz']
    Rytov = phz_params_dict['Rytov']
    for idx in range(len(population)):
        individual = population[idx]
        w = ft_sh_phase_screen(target, phz_params_dict=phz_params_dict, genetic_code=individual)
        w = abs(w)
        lse = sum(sum((target-w)**2))
        #lse = sum(sum((target - w)**2)) # maybe 1 sum is enough # why is there a negative here?
        newPopulation.append((individual, lse))

    toplse = np.array([np.inf]*Ntop)
    topInds = [np.zeros([N, N, nscreen, 2]) for i in range(Ntop)]
    for individual, lse in newPopulation:
        idx = np.argmax(toplse)
        if lse < toplse[idx]:
            toplse[idx] = lse
            topInds[idx] = individual

    return topInds, toplse

def pick_top_arr(target, population, Ntop, phz_params_dict):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
	 
    lse_arr = np.zeros([len(population)])
    for ind in range(len(population)):
        individual = population[ind]
        w = ft_sh_phase_screen(target, phz_params_dict=phz_params_dict, genetic_code=individual)
        w = abs(w)
        lse_arr[ind] = sum(sum((target-w)**2))

    #topInds = [np.zeros([N, N, nscreen, 2]) for i in range(Ntop)]
    topInds = np.zeros([Ntop,N,N,nscreen,2])
    topIndices = np.argpartition(lse_arr, Ntop)[:Ntop] #check behavior here
    u = 0
    for idx in topIndices:
        topInds[u] = population[idx]
        u += 1

    return topInds

def r_pick_top_arr(target, population, Ntop, phz_params_dict):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']

    lse_arr = np.zeros([len(population)])
    for ind in range(len(population)):
        individual = population[ind]
        w = ft_sh_phase_screen(individual, phz_params_dict=phz_params_dict, genetic_code=None)
        w = abs(w)
        lse_arr[ind] = sum(sum((target-w)**2))

    topInds = np.zeros([Ntop,N,N])
    topIndices = np.argpartition(lse_arr, Ntop)[:Ntop]
    u = 0
    for idx in topIndices:
        topInds[u] = population[idx]
        u += 1

    return topInds

def generate_population(Nindividuals, phz_params_dict, topInds=None): # pry want to avoid having N in phz dict
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    population = []
    if topInds is None:
        pass

    else:
        for idx in range(Nindividuals):
            p1, p2 = np.random.choice(len(topInds), 2)
            parent1 = topInds[p1][:,:,:,0]
            parent2 = topInds[p1][:,:,:,1]
            new = np.zeros([N, N, nscreen, 2], dtype=complex)
            new[:,:,:,0] = parent1
            new[:,:,:,1] = parent2
            population.append(new)

    return population

def generate_population_arr(Nindividuals, phz_params_dict, topInds=None):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    population = np.zeros([Nindividuals, N, N, nscreen, 2])
    if topInds is None:
        pass

    else:
        choices = [np.random.choice(len(topInds), 2) for idx in range(Nindividuals)]
        for ind in range(Nindividuals):
            p1, p2 = choices[ind]
            population[ind,:,:,:,0] = topInds[p1,:,:,:,0]
            population[ind,:,:,:,1] = topInds[p2,:,:,:,1]

    return population


def r_generate_population_arr(Nindividuals, phz_params_dict, topInds=None):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    population = np.zeros([Nindividuals, N, N])
    if topInds is None:
        pass

    else:
        choices = [np.random.choice(len(topInds), 2) for idx in range(Nindividuals)]
        for ind in range(Nindividuals):
            p1, p2 = choices[ind]
            p1 = topInds[p1]
            p2 = topInds[p2]
            arr1 = np.random.randint(2, size=(N,N))
            arr2 = abs(arr1 - np.ones([N,N]))
            population[ind,:,:] = p1 * arr1 + p2 * arr2

    return population

def mutate(population, phz_params_dict, mutation_rate=0.1):
    newPopulation = []
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    for idx in range(len(population)):
        individual = population[idx]

        mutation = np.zeros([N, N], dtype=complex)
        re = np.random.normal(loc=0.0, scale=1.0, size=[N, N])
        mutation = re*mutation_rate

        which_screen = np.random.choice(nscreen, 1)[0] # which phase screen to mutate # currently, mutate 1 in each ind
        real_or_img = np.random.choice(2, 1)[0]
        individual[:,:,which_screen,real_or_img] = individual[:,:,which_screen,real_or_img] + mutation
        newPopulation.append(individual)

    return newPopulation

def mutate_arr(population, phz_params_dict, mutation_rate=0.1):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    for ind in range(len(population)):
        mutation = np.zeros([N, N], dtype=complex)
        re = np.random.normal(loc=0.0, scale=1.0, size=[N, N])
        mutation = re*mutation_rate

        which_screen = np.random.choice(nscreen, 1)[0]
        real_or_img = np.random.choice(2, 1)[0]
        
        gene = population[ind,:,:,which_screen,real_or_img]
        population[ind,:,:,which_screen,real_or_img] = gene + mutation

    return population

def r_mutate_arr(population, phz_params_dict, mutation_rate=0.1, side=30):
    N = phz_params_dict['N']
    nscreen = phz_params_dict['nscreen']
    how_many = int(len(population) * mutation_rate)
    lucky_ones = np.random.choice(len(population), how_many)

    for ind in lucky_ones:
		  #mutation = np.random.normal(loc=0.0, scale=1.0, size=[N, N])
        gene = population[ind,:,:]
        x2 = np.random.randint(256)
        y2 = np.random.randint(256)
        if side < x2:
            x1 = x2 - side
        else:
            x2 = side
        if side < y2:
             y1 = y2 - side
        else:
            y2 = side
        x1 = x2 - side
        y1 = y2 - side
        mutation = np.zeros([N,N])
        #print(mutation[x1:x2,y1:y2])
        #print(np.random.normal(loc=0.0, scale=1.0, size=[10, 10]))
        mutation[x1:x2,y1:y2] = np.ones([side, side])
		  #mutation[x1:x2,y1:y2] = np.random.normal(loc=0.0, scale=1.0, size=[10, 10])
        if 1 == np.random.randint(2):
            mutation = mutation * 100
        else:
            mutation = mutation * -100
		  #population[ind,:,:] = gene + mutation
        #sigma_y = 0.8
        #sigma_x = 0.6
        #sigma = [sigma_y, sigma_x]
        #population[ind,:,:] = sp.ndimage.filters.gaussian_filter(gene, sigma, mode='constant')
        population[ind,:,:] = gene + mutation

    return population

def main():
    parser = argparse.ArgumentParser(description='Genetic algorithm.')
    parser.add_argument('--ntrain', type=int, nargs=1, default=50, help='Number of samples to generate for the training dataset')
    parser.add_argument('--npixels', type=int, nargs=1, default=256, help='Number of pixels per side for each image')
    parser.add_argument('--filename', type=str, nargs=1, default='ufo.npz', help='Output filename (default is ufo.npz)')

    args = parser.parse_args()
    ntrain = int(args.ntrain) 
    ntest = int(ntrain*0.20) # TODO: Parse from command line
    N = int(args.npixels)
    nchan = 1 # RBG, grayscale, etc. Better than 4Chan
    nscreen = 10

    filename = '/home/jamunoz/git/turbulence/data/pristine_010000.hdf5'
    f = h5py.File(filename, 'r')

    # generate target
    Uin = f['PUMP']['Data'][0,:,:]

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

    Uout = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)

    target = abs(Uout)
    Nindividuals = 100
    population_orig = r_generate_orig_population_arr(N=N, nscreen=nscreen, Nindividuals=Nindividuals)

    Ngenerations = 600
    elitism = 0.2
    Ntop = int(Nindividuals * elitism)

    population = population_orig
    for generation in range(Ngenerations):
        start = time.time()
        side = int((Ngenerations-generation)*(1/2))
        if 0 == side :
            side = 1
        topInds = r_pick_top_arr(target, population, Ntop=Ntop, phz_params_dict=phz_params_dict)
        population = r_generate_population_arr(Nindividuals, phz_params_dict=phz_params_dict, topInds=topInds)
        population = r_mutate_arr(population, phz_params_dict=phz_params_dict, mutation_rate=0.4, side=side)
        if 0 == generation % 100:
            np.savez('r_population_serial.npz', population)
        print(generation, time.time()-start)

    np.savez('r_population_serial.npz', population)
#fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)

#fig.colorbar(ax1.imshow(topInds[0], cmap=cmap, vmin=vmin, vmax=vmax), ax=ax1)
#filename='_r.png'
#plt.savefig(filename)



main()
