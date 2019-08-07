import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import time, copy, os, h5py

from numpy.fft import ifftshift, ifft2, fftshift, fft2
from numpy import exp, pi, mean, var, std, abs, sin, cos, sum

filename = '/Volumes/jorgeamu/AFRL/dump1/hdf5_files/pristine_010000.hdf5'
f = h5py.File(filename, 'r')

# generate target
Uin = f['PUMP']['Data'][0,:,:]

def ft_sh_phase_screen(Uin, phz_params_dict, genetic_code=None):
    N = phz_params_dict['N']
    Lout = phz_params_dict['Lout']
    Lin = phz_params_dict['Lin']
    deltax = phz_params_dict['deltax']
    wvl = phz_params_dict['wvl']
    Dz = phz_params_dict['Dz']
    nscreen = phz_params_dict['nscreen']
    kpow = phz_params_dict['kpow']
    Rytov = phz_params_dict['Rytov']
    Np = phz_params_dict['Np']

    k=2*pi/wvl
    nn=np.arange(-np.floor(N/2), np.floor(N/2))
    nx, ny = np.meshgrid(nn, nn)
    nsq = nx**2 + ny**2
    z=np.linspace(0, Dz, nscreen)

    deltaz=z[1]-z[0]
    deltaf=1/(N*deltax)

    Cn2=Rytov/(1.2287075122549518 * k**(7/6) * Dz**(11/6))
    r0 = (.423*k**2*Cn2*Dz)**(-3/5) #check

    fx=nx*deltaf
    fy=ny*deltaf
    fsq=fx**2+fy**2

    Uin=Uin #np.ones([N, N])
    g=Uin

    # Generate 2x nscreens NxN arrays
    if genetic_code is None:
        genetic_code = np.zeros([N, N, nscreen, 2])
        for idx in range(nscreen-1):
            genetic_code[:,:,idx,0] = np.random.randn(N, N)
            genetic_code[:,:,idx,1] = np.random.randn(N, N)

    for idx in range(nscreen-1):
        #r0 = (.423*(k**2)*Cn2*deltaz)**(-3/5) # Constant Cn2
					 
        fm = 5.92/Lin/(2*pi)
        f0=1/Lout

        PSD_phi = 0.023 * r0**(-5/3) * exp(-(fsq/fm**2)) / (fsq + f0**2)**(kpow/2)
        PSD_phi[int(N/2),int(N/2)] = 0
        
        cnm = (genetic_code[:,:,idx,0] + 1j*genetic_code[:,:,idx,1]) * np.sqrt(PSD_phi)*deltaf

        phz_hi = np.real(ifftshift(ifft2(ifftshift(cnm)))*(N*1)**2)
        phz_lo = np.zeros([N, N])
		#phz_lo = subharmonics(Np, phz_hi)
        phz = phz_hi + phz_lo
		  
        Q2 = exp(-1j*pi**2*2*deltaz/k*fsq)
        G= Q2*fftshift(fft2(fftshift(g)))*deltax**2
        g = ifftshift(ifft2(ifftshift(G)))*(N*deltaf)**2
        g = exp(1j*phz) * g
		  
    Uout = g
    
    return Uout

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
    
def generate_individual(N, tiles_per_side):
    genetic_code = []
    for i in range(tiles_per_side**2):
        genetic_code.append(abs(np.random.normal(loc=0.0, scale=1.0)*100))

    #canvas = embody(N, tiles_per_side, genetic_code=genetic_code)
        
    return genetic_code
    
def embody(N, tiles_per_side, genetic_code):
    tile_indices = generate_tile_indices(N, tiles_per_side)
    tile_side = int(N/tiles_per_side)
    canvas = np.zeros([N, N])
    for i in range(len(tile_indices)):
        (x1, x2), (y1, y2) = tile_indices[i]
        canvas[x1:x2, y1:y2] = np.ones([tile_side, tile_side]) * genetic_code[i]
        
    return canvas
    
def generate_orig_population(N, Nindividuals, tiles_per_side):
    population = []
    for i in range(Nindividuals):
        population.append(generate_individual(N, tiles_per_side))
    
    return population
    
def pick_top_i(N, tiles_per_side, population, Ntop, Uout, minEntropy=False):
    lses = []
    entropies = []
    free_energies = []
    for genetic_code in population:
        canvas = embody(N, tiles_per_side, genetic_code=genetic_code)
        Cout = abs(ft_sh_phase_screen(canvas, phz_params_dict=phz_params_dict, 
                                      genetic_code=None)) 
        lse = sum(sum((Cout-Uout)**2)) / N**2
        
        #print(lse)
        T = np.log10(lse) #- 1
        T = 10**T
        #lses.append(lse)
        
        normalized_canvas = np.array(canvas) / canvas.sum()
        mean_value = np.mean(normalized_canvas)
        #shannon = - np.array([p * np.log2(p) for p in normalized_canvas]).sum()
        #entropies.append(shannon)
        #free_energy = lse + shannon * T
        
        #kl = - np.array([p * np.log2(mean_value/p) for p in normalized_canvas]).sum()
        #entropies.append(kl)
        #free_energy = lse + kl * T
        
        mean_value = np.mean(canvas)
        upper_mean = np.mean(canvas[canvas > mean_value])
        lower_mean = np.mean(canvas[canvas <= mean_value])
        distance = upper_mean - lower_mean
        inv_dist = 0.5/distance
        #print(upper_mean, lower_mean, distance)
        
        alpha = lse/inv_dist # makes them equally important
        free_energy = lse + alpha * inv_dist
        
        lses.append(free_energy)
        free_energies.append(free_energy) # separates the values, but position don't matter

    #topIndices = pd.Series(lses).nsmallest(Ntop).index
    topIndices = pd.Series(free_energies).nsmallest(Ntop).index
    topInds = []
    topLses = []
    for ti in topIndices:
        topInds.append(population[ti])
        topLses.append(lses[ti]) 
    
    return topInds, topLses
    
def generate_population(Nindividuals, topInds, topLses):
    population = []
    for i in range(Nindividuals):
        p1, p2 = np.random.randint(len(topInds), size=(2))
        q1 = int(topLses[p1] / (topLses[p1] + topLses[p2]) * len(topInds[p1]))
        q2 = len(topInds[p1]) - q1
        
        list(np.random.choice(len(topInds[p1]), len(topInds[p1]), replace=False))
        
        genesIdx = list(np.random.choice(len(topInds[p1]), len(topInds[p1]), replace=False))
        newInd = [0 for x in range(len(topInds[p1]))]
        for gidx in genesIdx:
            if 0 == gidx % 2:
                newInd[gidx] = topInds[p1][gidx]
            else:
                newInd[gidx] = topInds[p2][gidx]
            
        population.append(newInd)
        
    return population    
    
def generate_population_i(Nindividuals, topInds, topLses):
    population = []
    for i in range(Nindividuals):
        p1, p2 = np.random.randint(len(topInds), size=(2))
        q1 = int(topLses[p1] / (topLses[p1] + topLses[p2]) * len(topInds[p1]))
        q2 = len(topInds[p1]) - q1
        
        genesIdx = range(len(topInds[p1]))
        s1 = set(np.random.choice(genesIdx, q1, replace=False))
        s2 = set(genesIdx) - s1
        
        newInd = [0 for x in range(len(topInds[p1]))]
        for gidx in s1:
            newInd[gidx] = topInds[p1][gidx]
        for gidx in s2:
            newInd[gidx] = topInds[p2][gidx]
        
        population.append(newInd)
        
    return population   
    
def mutate(population, mutation_rate=0.1, area_affected=1):
    for ind in population:
        genesIdx = list(np.random.randint(len(ind), size=(area_affected)))
        for gidx in genesIdx:
            if 1 == np.random.randint(2):
                ind[gidx] = ind[gidx] + ind[gidx] * mutation_rate
            else:
                ind[gidx] = ind[gidx] - ind[gidx] * mutation_rate
                    
    return population
    
def generate_individual_from_code(N, tiles_per_side, original_tiles_per_side, genetic_code):
    
    if 0 == tiles_per_side / original_tiles_per_side % 2:
        pass
    else:
        return None
    
    original_tile_indices = generate_tile_indices(N, tiles_per_side=original_tiles_per_side)
    new_tile_indices = generate_tile_indices(N, tiles_per_side=tiles_per_side)
    new_genetic_code = [0 for x in new_tile_indices]
    
    for ntidx in range(len(new_tile_indices)):
        for otidx in range(len(original_tile_indices)):
            (x1o, x2o), (y1o, y2o) = original_tile_indices[otidx]
            (x1n, x2n), (y1n, y2n) = new_tile_indices[ntidx]
            
            if x1n >= x1o and x1n <= x2o:
                if x2n >= x1o and x2n <= x2o:
                    if y1n >= y1o and y1n <= y2o:
                        if y2n >= y1o and y2n <= y2o:
                            new_genetic_code[ntidx] = genetic_code[otidx]
                            continue
        
    return new_genetic_code


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
phz_params_dict['Rytov'] = 0.02
phz_params_dict['Np'] = 5

Uout = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)

Uin = abs(Uin)
Uout = abs(Uout)

N = phz_params_dict['N']
Nindividuals = 20
tiles_per_side = 16
elitism = 0.1
Ntop = int(Nindividuals * elitism)
Ngenerations = 5
population_orig = generate_orig_population(N, Nindividuals, tiles_per_side)

topLsesList = []
population16 = population_orig
for generation in range(Ngenerations):
    mutation_rate = 0.2 #(Ngenerations/(generation+1))-1
    area_affected = int((tiles_per_side**2)/16)
    start = time.time()
    topInds, topLses = pick_top_i(N, tiles_per_side, population16, Ntop, Uout)
    population16 = generate_population_i(Nindividuals, topInds, topLses)
    population16 = mutate(population16, mutation_rate=0.3, area_affected=area_affected)
    print(generation, time.time()-start, np.mean(topLses),
          np.median(topLses), np.log10(np.mean(topLses)))
    topLsesList.append(topLses)
    
np.savez('t_population16.npz', population16)

population32 = []
for idx in range(len(population16)):
    nind = generate_individual_from_code(256, 32, 16, genetic_code=population16[idx])
    population32.append(nind)

topLsesList = []
population32 = population32
for generation in range(Ngenerations):
    mutation_rate = 0.2 #(Ngenerations/(generation+1))-1
    area_affected = int((tiles_per_side**2)/16)
    start = time.time()
    topInds, topLses = pick_top_i(N, tiles_per_side, population32, Ntop, Uout)
    population32 = generate_population_i(Nindividuals, topInds, topLses)
    population32 = mutate(population32, mutation_rate=0.3, area_affected=area_affected)
    print(generation, time.time()-start, np.mean(topLses),
          np.median(topLses), np.log10(np.mean(topLses)))
    topLsesList.append(topLses)

np.savez('t_population32.npz', population32)

population64 = []
for idx in range(len(population32)):
    nind = generate_individual_from_code(256, 64, 32, genetic_code=population32[idx])
    population64.append(nind)

topLsesList = []
population64 = population64
for generation in range(Ngenerations):
    mutation_rate = 0.2 #(Ngenerations/(generation+1))-1
    area_affected = int((tiles_per_side**2)/16)
    start = time.time()
    topInds, topLses = pick_top_i(N, tiles_per_side, population64, Ntop, Uout)
    population64 = generate_population_i(Nindividuals, topInds, topLses)
    population64 = mutate(population64, mutation_rate=0.3, area_affected=area_affected)
    print(generation, time.time()-start, np.mean(topLses),
          np.median(topLses), np.log10(np.mean(topLses)))
    topLsesList.append(topLses)

np.savez('t_population64.npz', population64)

population128 = []
for idx in range(len(population32)):
    nind = generate_individual_from_code(256, 128, 64, genetic_code=population64[idx])
    population128.append(nind)

topLsesList = []
population128 = population128
for generation in range(Ngenerations):
    mutation_rate = 0.2 #(Ngenerations/(generation+1))-1
    area_affected = int((tiles_per_side**2)/16)
    start = time.time()
    topInds, topLses = pick_top_i(N, tiles_per_side, population128, Ntop, Uout)
    population128 = generate_population_i(Nindividuals, topInds, topLses)
    population128 = mutate(population128, mutation_rate=0.3, area_affected=area_affected)
    print(generation, time.time()-start, np.mean(topLses),
          np.median(topLses), np.log10(np.mean(topLses)))
    topLsesList.append(topLses)
    
np.savez('t_population128.npz', population128)

print('Done.')
