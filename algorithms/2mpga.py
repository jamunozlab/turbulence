import sys, os
from multiprocessing import Process, Queue, cpu_count
from multiprocessing import Pool
import numpy as np
import argparse, h5py

from numpy import mean

sys.path.append('../')
from turbulence.utils.ft import ft_sh_phase_screen_ea as ft_sh_phase_screen
from turbulence.algorithms.ga import generate_orig_population, generate_population, pick_top, mutate
from turbulence.utils.mp import WorkPool, Worker


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

#Uout = ft_sh_phase_screen(Uin, N=N, Lout=Lout, Lin=Lin, deltax=deltax, wvl=wvl, Dz=Dz, nscreen=nscreen, \
#                                 kpow=kpow, Rytov=Rytov, Np=Np, genetic_code=None)

target = abs(Uout)
Nindividuals = 100
population_orig = generate_orig_population(N=N, nscreen=nscreen, Nindividuals=Nindividuals)

print(len(population_orig))

#===============================================================================
# Globals
#===============================================================================
class _Globs():
    def __init__(self):
        self.nr = 0
        self.population = population_orig

g = _Globs()


#===============================================================================
# Work - worker class instantiated for each worker process
#===============================================================================
class Work(Worker):
    def __init__(self, *args, **kwargs):
        Worker.__init__(self, *args, **kwargs)

    def work(self, name, data):
        kwargs         = self.wrk_kwargs
        phz_params_dict = data["phz_params_dict"]

        print(phz_params_dict['Rytov'])


def queue_access(queue, phz_params_dict):
    data = {}
    g.nr += 1
    data['phz_params_dict'] = phz_params_dict
    queue.put(data)

wrk_kwargs = {}
wrk_kwargs["phz_params_dict"] = phz_params_dict
wp = WorkPool(Work, wrk_kwargs=wrk_kwargs)

nworkers = 20
wp.start(nworkers, quitOnEmpty=False)


for t in range(25):
   queue_access(wp.queue, phz_params_dict)


wp.run()
wp.quit()
print('overall done.')

sys.exit()

Ngenerations = 1
elitism = 0.2
Ntop = int(Nindividuals * elitism)


population = population_orig
#topInds, toplse = pick_top(target, population, Ntop=cpu_count(), phz_params_dict=phz_params_dict)

def evolve(target, population, Ntop, phz_params_dict):
    topInds, toplse = pick_top(target, population, Ntop=Ntop, phz_params_dict=phz_params_dict)
    population = generate_population(Nindividuals, phz_params_dict=phz_params_dict,  topInds=topInds)
    population = mutate(population, phz_params_dict=phz_params_dict, mutation_rate=0.1)

    return population


#for generation in range(Ngenerations):
#    population = evolve(target, population, Ntop, phz_params_dict)

with Pool() as pool:
    results = [pool.apply_async(evolve, (target, population, Ntop, phz_params_dict)) for i in range(20)]
    print([res.get() for res in results])
    #print('Here', len(result), len(result[0]))

#print(len(population))

#def evolve(q):
#    pick_top(target, population, Ntop=Ntop, phz_params_dict=phz_params_dict)


def f(q):
    q.put([42, None, 'hello'])

q = Queue()
p = Process(target=f, args=(q,))
p.start()
print(q.get())    # prints "[42, None, 'hello']"
p.join()

print(cpu_count())

