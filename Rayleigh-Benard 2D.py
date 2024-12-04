"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import os
import logging
logger = logging.getLogger(__name__)

# Output Control Panel
BUOYANCY    = True
VORTICITY   = True
KE          = True
NUSSELT     = True
FLUXES      = True


# Parameters
Lx, Lz = 1, 1
Nx, Nz = 256, 256
Rayleigh = 1e7
Prandtl = 1
dealias = 3/2
stop_sim_time = 800
timestepper = d3.RK222
max_timestep = 0.01
dtype = np.float64


# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

ux = d3.DotProduct(u , ex)
uz = d3.DotProduct(u , ez)
db_dz = d3.DotProduct(d3.grad(b) , ez)
dux_dz = d3.DotProduct(d3.grad(ux) , ez)

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("b(z=0) = Lz")
problem.add_equation("b(z=Lz) = 0")

# Stress-free boundary conditions
problem.add_equation("dux_dz(z=0) = 0")
problem.add_equation("dux_dz(z=Lz) = 0")
problem.add_equation("uz(z=0) = 0")
problem.add_equation("uz(z=Lz) = 0")


# No-slip boundary conditions
# problem.add_equation("u(z=0) = 0")
# problem.add_equation("u(z=Lz) = 0")


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# Initial conditions
#ux['g'] += 0.1 * (2*z - 1)

b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b['g'] += Lz - z # Add linear background

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10000)
if BUOYANCY: snapshots.add_task(b, name='buoyancy')
if VORTICITY: snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# KE & Nusselt from Tom's github > Rayleigh-Benard > rb_convect.py (written in d2 so tried modifying to d3)
if KE: snapshots.add_task(d3.Integrate( (0.5*(u*u)) , ('x', 'z')), layout="g", name="KE")
if NUSSELT: 
    snapshots.add_task( (1 + d3.Integrate( b*uz , ('x' , 'z')) /kappa) , layout="g", name="Nusselt1")
    snapshots.add_task( (1 + d3.Integrate( (b*uz)/(-db_dz) , ('x' , 'z')) /kappa) , layout="g", name="Nusselt2")
    snapshots.add_task( (1 + d3.Integrate( b*uz , ('x' , 'z')) / d3.Integrate( -db_dz , ('x' , 'z')) /kappa) , layout="g", name="Nusselt3")


if FLUXES:
    snapshots.add_task( d3.Average( b*uz , 'x') , layout='g' , name='flux_convective')
    snapshots.add_task( d3.Average( -db_dz , 'x') , layout='g' , name='flux_conductive')
    snapshots.add_task( d3.Average( b*uz -db_dz , 'x') , layout='g' , name='flux_total')


# Taken from Matt's (d2) example
# analysis.add_task(" 1.0 + integ( (integ(b*w,'x')/Lx)/(integ((-1.0*P)*bz, 'x')/Lx), 'z')/Lz", layout='g',name='Nu1')
# analysis.add_task(" 1.0 + integ(integ(b*w/(-1.0*P*bz), 'x'), 'z')/(Lx*Lz)", layout='g', name='Nu2')
# analysis.add_task(" 1.0 + integ(integ(b*w, 'x'), 'z')/(P*Lx*Lz)", layout='g', name='Nu3')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info(f'Iteration={solver.iteration:.0f}, Time={solver.sim_time:.8g}, dt={timestep:.6e}, max(Re)={max_Re:.7g} \t\t Up to t={stop_sim_time:.0f}: {100*solver.sim_time/stop_sim_time:.3f}%')
    
    os.rename(os.getcwd()+'/snapshots' , os.getcwd()+f'/snapshots_{Rayleigh:.0e}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()