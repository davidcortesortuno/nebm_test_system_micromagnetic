"""
Relaxation to get a skyrmion using Fidimag for a toy model made of
Fe-like atoms arranged in a square lattice. 

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


# Define an initial state for the magnetisation as a function of space
# This function must return a tuple with (mx, my, mz)

# Initial skyrmion radius
in_radius = 5


# Initial state To get a skyrmion. It is only a small region
# at the center of the square with inverted spins, which will
# be the skyrmion core
def init_m(pos, sign=-1):

    x, y = pos[0] - 10.5, pos[1] - 10.5

    if np.sqrt(x ** 2 + y ** 2) <= in_radius:
        # Polar coordinates:
        r = (x ** 2 + y ** 2) ** 0.5
        phi = np.arctan2(y, x)
        # This determines the profile we want for the
        # skyrmion
        # Single twisting: k = pi / R
        k = np.pi / (in_radius)

        # We define here a 'hedgehog' skyrmion pointing down
        return (sign * np.sin(k * r) * np.cos(phi),
                sign * np.sin(k * r) * np.sin(phi),
                -np.cos(k * r))
    else:
        return (0, 0, 1)


# MESH --------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SIMULATION ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_simulation(mesh, simname):
    # Initiate a simulation object. PBCs are specified in the mesh
    sim = Sim(mesh, name=simname)
    # Use default gamma value
    # sim.gamma = const.gamma

    # Magnetisation in A/m
    sim.Ms = 148367

    # We could change the parameters using this option
    # sim.set_options(gamma=const.gamma)

    # Initial magnetisation profile from the function
    sim.set_m(init_m)

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

    # Tune the damping for faster convergence
    sim.driver.alpha = 0.5
    # Remove precession
    sim.driver.do_precession = False
    sim.driver.set_tols(rtol=1e-12, atol=1e-12)

    return sim

# Relax the system

# We will simulate skyrmions for different mesh
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
    sim = create_simulation(mesh, 'skyrmion_d-{}e-2'.format(d))

    sim.relax(dt=1e-13,
              stopping_dmdt=0.01,
              max_steps=5000,
              save_m_steps=100, save_vtk_steps=100)

folders = ['vtks', 'npys', 'txts']
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

for item in os.listdir('./'):
    for f in folders[:-1]:
        if item.endswith(f):
            shutil.move(item, f)
    if item.endswith('.txt'):
        shutil.move(item, 'txts')
