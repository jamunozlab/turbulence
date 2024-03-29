import sys
import numpy as np
import argparse
import h5py

from numpy import mean

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen

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

def pick_top(targets, population, Ntop, phz_params_dict):
    N, N = targets[0].shape
    newPopulation = []
    nscreen = phz_params_dict['nscreen']
    Dz = phz_params_dict['Dz']
    Rytov = phz_params_dict['Rytov']
    for idx in range(len(population)):
        individual = population[idx]
        lses = []
        for target in targets:
            w = ft_sh_phase_screen(target, phz_params_dict=phz_params_dict, genetic_code=individual)
            #w = ft_sh_phase_screen(target, N=N, Lout=10, Lin=1e-3, deltax=0.005, wvl=0.532e-6, Dz=Dz, \
		      #            nscreen=nscreen, kpow=22/6, Rytov=Rytov, Np=5, genetic_code=individual)
            w = abs(w)
            lse = sum(sum((target-w)**2))
            lses.append(lse)
            #lse = sum(sum((target - w)**2)) # maybe 1 sum is enough # why is there a negative here?
        lse = mean(np.array(lses))
        newPopulation.append((individual, lse))

    toplse = np.array([np.inf]*Ntop)
    topInds = [np.zeros([N, N, nscreen, 2]) for i in range(Ntop)]
    for individual, lse in newPopulation:
        idx = np.argmax(toplse)
        if lse < toplse[idx]:
            toplse[idx] = lse
            topInds[idx] = individual

    return topInds, toplse

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

    root = '/home/jamunoz/git/turbulence/data/pristine_'
    files = ['010000.hdf5','010019.hdf5','010039.hdf5','010059.hdf5','010079.hdf5','010099.hdf5']
    Uins = []
    for fi in files:
        filename = root + fi
        f = h5py.File(filename, 'r')
        Uin = f['PUMP']['Data'][0,:,:]
        Uins.append(Uin)

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

    targets = []
    for Uin in Uins:
        Uout = ft_sh_phase_screen(Uin, phz_params_dict=phz_params_dict, genetic_code=None)
        #Uout = ft_sh_phase_screen(Uin, N=N, Lout=Lout, Lin=Lin, deltax=deltax, wvl=wvl, Dz=Dz, nscreen=nscreen, \
	     #            kpow=kpow, Rytov=Rytov, Np=Np, genetic_code=None)
        target = abs(Uout)
        targets.append(Uout)

    np.savez('targets.npz', targets)

    Nindividuals = 100
    population_orig = generate_orig_population(N=N, nscreen=nscreen, Nindividuals=Nindividuals)

    Ngenerations = 2000
    elitism = 0.2
    Ntop = int(Nindividuals * elitism)

    population = population_orig
    for generation in range(Ngenerations):
        topInds, toplse = pick_top(targets, population, Ntop=Ntop, phz_params_dict=phz_params_dict)
        population = generate_population(Nindividuals, phz_params_dict=phz_params_dict,  topInds=topInds)
        population = mutate(population, phz_params_dict=phz_params_dict, mutation_rate=0.1)
        # print(generation, mean(toplse)) #verbose
        if 0 == generation % 100:
            np.savez('population.npz', population)

main()
#if __name__ == "__main__":
#    main()

