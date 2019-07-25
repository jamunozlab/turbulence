import sys, os
from multiprocessing import Process, Queue, cpu_count
from multiprocessing import Pool
import numpy as np
import argparse, h5py
import time

from numpy import mean

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen

from turbulence.algorithms.ga import generate_orig_population, generate_population, pick_top, mutate

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

Uout = ft_sh_phase_screen(Uin, phz_params_dict, genetic_code=None)

target = abs(Uout)
Nindividuals = 100
population_orig = generate_orig_population(N=N, nscreen=nscreen, Nindividuals=Nindividuals)

print(len(population_orig))

Ngenerations = 10
elitism = 0.2
Ntop = int(Nindividuals * elitism)


population = population_orig
#topInds, toplse = pick_top(target, population, Ntop=cpu_count(), phz_params_dict=phz_params_dict)

def evolve(target, population, Ntop, phz_params_dict):
    topInds, toplse = pick_top_mp(target, population, Ntop=Ntop, phz_params_dict=phz_params_dict)
    population = generate_population(Nindividuals, phz_params_dict=phz_params_dict,  topInds=topInds)
    population = mutate(population, phz_params_dict=phz_params_dict, mutation_rate=0.1)

    return population

def funct(target, phz_params_dict, individual):
    w = ft_sh_phase_screen(target, phz_params_dict, individual)
    lse = sum(sum((target-abs(w))**2))

    return individual, lse

def pick_top_mp(target, population, Ntop, phz_params_dict):
    with Pool(processes=40) as pool:
        results = [pool.apply_async(funct, (target, phz_params_dict, population[i])) for i in range(len(population))]
        new_population = [res.get() for res in results]

    toplse = np.array([np.inf]*Ntop)
    topInds = [np.zeros([N, N, nscreen, 2]) for i in range(Ntop)]
    for individual, lse in new_population:
        idx = np.argmax(toplse)
        if lse < toplse[idx]:
            toplse[idx] = lse
            topInds[idx] = individual

    return topInds, toplse

for generation in range(Ngenerations):
    start = time.time()
    population = evolve(target, population, Ntop, phz_params_dict)
    if 0 == generation % 100:
        np.savez('population_mp.npz', population)
    print(time.time()-start)

sys.exit()


