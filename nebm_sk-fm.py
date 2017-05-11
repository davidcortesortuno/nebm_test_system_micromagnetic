"""
NEBM Simulation for a toy model made of Fe-like atoms arranged in a square
lattice. This system is described by 21 x 21 spins with interfacial DMI
A very strong magnetic field B is applied perpendicular to the sample, which
stabilises a metastable skyrmion. The ground state is the uniform state
For the NEBM we use Geodesic distances and vector projections into the tangents
space

Micromagnetic equivalent

Magnetic parameters:
    J    = 10 meV  -->  A = 1.602 pJm-1   Exchange
    D    = 6 meV   -->  D = 3.84 mJm-2    DMI
    B    = 25 T    -->  H = B / mu0       Magnetic Field
    mu_s = 2 mu_B  -->  Ms = 148367       Magnetisation

Created by David Cortes on Thu 11 May 2017 09:02:27
University of Southampton
Contact to: d.i.cortes@soton.ac.uk
"""

import os, shutil
# Numpy utilities
import numpy as np

# FIDIMAG Simulation imports:
from fidimag.micro import Sim
from fidimag.common import CuboidMesh
from fidimag.micro import DMI
from fidimag.micro import UniformExchange
from fidimag.micro import Zeeman
# Import physical constants from fidimag
import fidimag.common.constant as c
# Import the NEB method
from fidimag.common.nebm_geodesic import NEBM_Geodesic


# NEBM Simulation Function ----------------------------------------------------

def relax_neb(mesh, k, maxst, simname, init_im, interp,
              save_every=10000, stopping_dYdt=0.01):
    """
    Execute a simulation with the NEBM algorithm of the FIDIMAG code
    Here we use always the 21x21 Spins Mesh and don't vary the material
    parameters. This can be changed adding those parameters as variables.
    We create a new Simulation object every time this function is called
    since it can be modified in the process
    k           :: NEBM spring constant
    maxst       :: Maximum number of iterations
    simname     :: Simulation name. VTK and NPY files are saved in folders
                   starting with the 'simname' string
    init_im     :: A list with magnetisation states (usually loaded from
                   NPY files or from a function) that will be used as
                   images in the energy band, e.g. for two states:
                        [np.load('skyrmion.npy'), np.load('ferromagnet.npy')]
    interp      :: Array or list with the numbers of interpolations between
                   every pair of the 'init_im' list. The length of this array
                   is: len(__init_im) - 1
    save_every  :: Save VTK and NPY files every 'save_every' number of steps
    """

    # Initialise a simulation object and set the default gamma for the LLG
    # equation
    sim = Sim(mesh, name=simname)
    # sim.gamma = const.gamma

    # Magnetisation in A/m
    sim.Ms = 148367

    # Interactions ------------------------------------------------------------

    # Exchange constant
    A = 1.602e-12
    exch = UniformExchange(A)
    sim.add(exch)

    # DMI constant
    D = 3.84e-3
    dmi = DMI(D, dmi_type='interfacial')
    sim.add(dmi)

    # Zeeman field
    sim.add(Zeeman((0, 0, 25. / c.mu_0)))

    # -------------------------------------------------------------------------

    # Set the initial images from the list
    init_images = init_im

    # The number of interpolations must always be
    # equal to 'the number of initial states specified', minus one.
    interpolations = interp

    # Start a NEB simulation passing the Simulation object and all the NEB
    # parameters
    neb = NEBM_Geodesic(sim,
                        init_images,
                        interpolations=interpolations,
                        spring_constant=k,
                        name=simname,
                        )

    # Finally start the energy band relaxation
    neb.relax(max_iterations=maxst,
              save_vtks_every=save_every,
              save_npys_every=save_every,
              stopping_dYdt=stopping_dYdt
              )

    # Produce a file with the data from a cubic interpolation for the band
    interp_data = np.zeros((200, 2))
    interp_data[:, 0], interp_data[:, 1] = neb.compute_polynomial_approximation(200)
    np.savetxt(simname + 'interpolation.dat', interp_data)

# -----------------------------------------------------------------------------


# #############################################################################
# SIMULATION ##################################################################
# #############################################################################

# We will simulate the transitions for different mesh
# discretisations

meshes = [[0.25, 80, 80],
          [0.5, 40, 40],
          [0.75, 27, 27],
          [1.0, 20, 20],
          ]

for params in meshes:
    mesh = CuboidMesh(nx=params[1], ny=params[2], nz=1,
                      dx=params[0], dy=params[0], dz=0.5,
                      unit_length=1e-9,
                      periodicity=(True, True, False)
                      )
    d = int(params[0] * 100)

    # Here we load the skyrmion and the ferromagnetic state
    # Get the latest relaxed state:
    # Sort according to the number in the npy files,
    # Files are named: m_156.npy, for example
    # We will use the last element from these lists
    basedir_sk = 'relaxation/npys/skyrmion_d-{}e-2_npys/'.format(d)
    basedir_fm = 'relaxation/npys/ferromagnetic_d-{}e-2_npys/'.format(d)

    sk_npys = sorted(os.listdir(basedir_sk),
                     key=lambda x: int(x[2:-4]))
    fm_npys = sorted(os.listdir(basedir_fm),
                     key=lambda x: int(x[2:-4]))

    print(basedir_sk + sk_npys[-1])

    # listdir only gives the file names so we add the base directory
    init_im = [np.load(basedir_sk + sk_npys[-1]),
               np.load(basedir_fm + fm_npys[-1])]

    # We specify 16 interpolations in between the sk and fm, so we have
    # something like (as initial state)
    #
    # Energy                ...
    #   ^
    #   |          2    , - ~ ~ ~ - ,  15
    #   |           O '               O ,
    #   |     1   ,                       ,  16
    #   |        O                         O
    #   |       ,                           ,
    #   |       O                           O
    #   |      SK                           FM
    #   ________________________
    #   Distance
    #
    # So we will have 18 images in total in the Energy Band
    interp = [16]

    # Relax the NEBM simulation with a spring constant of k=1e4
    relax_neb(mesh,
              1e4, 2000,
              'nebm_micro_sk-fm_d-{}e-2'.format(d),
              init_im,
              interp,
              save_every=200,
              )
