import copy, time
import numpy as np
import matplotlib.pyplot as plt

from req_functions import *
from utils import compute_source_terms, pressure_rescaling, output_file_headers, initialize, write_output

# ************ Parameters for meshing *************
imax = 9               # Number of points in the x-direction (use odd numbers only)
jmax = 9               # Number of points in the y-direction (use odd numbers only)
neq = 3                # Number of equations to be solved (3: mass, x-mtm, y-mtm)

# Cavity dimensions...
xmin = 0.0             # Minimum x location (m)
xmax = 0.05            # Maximum x location (m)
ymin = 0.0             # Minimum y location (m)
ymax = 0.05            # Maximum y location (m)

# --------- User sets inputs here --------
nmax = 500000          # Maximum number of iterations
iterout = 5000         # Number of time steps between solution output
imms = 0               # Manufactured solution flag: 1 for manuf. sol., 0 otherwise
solver_method = 'coupled' # 'coupled' or 'fractional_step'
isgs = 1               # Symmetric Gauss-Seidel flag: 1 for SGS, 0 for point Jacobi
irstr = 0              # Restart flag: 1 for restart, 0 for initial run
cfl = 0.5              # CFL number used to determine time step
Cx = 0.01              # Parameter for 4th order artificial viscosity in x
Cy = 0.01              # Parameter for 4th order artificial viscosity in y
toler = 1e-10          # Tolerance for iterative residual convergence
rkappa = 0.5           # Time derivative preconditioning constant
Re = 10.0              # Reynolds number = rho*Uinf*L/rmu
uinf = 1.0             # Lid velocity (m/s)
rho = 1.0              # Density (kg/m^3)
fsmall = 1e-20         # Small parameter

# --------- User does not need to change --------
ipgorder = 0           # Order of pressure gradient: 0 = 2nd, 1 = 3rd (not needed in this final project)
lim = 1                # Variable to be used as the limiter sensor (= 1 for pressure)
pinf = 0.801333844662  # Initial pressure (N/m^2)
Cx2 = 0.0              # Coefficient for 2nd order damping (not required)
Cy2 = 0.0              # Coefficient for 2nd order damping (not required)
iterations = 1000     # Number of iterations for fractional step pressure solver
p_toler = 1e-3          # Tolerance for iterative residual convergence for the pressure solver

# Derived input quantities (initialized as placeholders)
rhoinv = 1/rho         # Inverse density, 1/rho (m^3/kg)
rlength = -99.9        # Characteristic length (m) [cavity width]
rmu = -99.9            # Viscosity (N*s/m^2)
vel2ref = -99.9        # Reference velocity squared (m^2/s^2)
dx = -99.9             # Delta x (m)
dy = -99.9             # Delta y (m)

# Constants for manufactured solutions
phi0 = [0.25, 0.3, 0.2]
phix = [0.5, 0.15, 1.0/6.0]
phiy = [0.4, 0.2, 0.25]
phixy = [1.0/3.0, 0.25, 0.1]
apx = [0.5, 1.0/3.0, 7.0/17.0]
apy = [0.2, 0.25, 1.0/6.0]
apxy = [2.0/7.0, 0.4, 1.0/3.0]
fsinx = [0.0, 1.0, 0.0]
fsiny = [1.0, 0.0, 0.0]
fsinxy = [1.0, 1.0, 0.0]

# -------- Main Function Initialization --------

# Looping indices
i, j, k, n = 0, 0, 0, 0
conv = -99.9             # Minimum of iterative residual norms from three equations

# Solution variables
ninit = 0                # Initial iteration number (used for restart file)
res = np.array([0.0, 0.0, 0.0])
resinit = np.array([0.0, 0.0, 0.0])
rL1norm = np.array([0.0, 0.0, 0.0])
rL2norm = np.array([0.0, 0.0, 0.0])
rLinfnorm = np.array([0.0, 0.0, 0.0])
rtime = -99.9            # Variable to estimate simulation time
dtmin = 1.0e99           # Minimum time step for a given iteration (initialized large)
x = -99.9                # Temporary variable for x location
y = -99.9                # Temporary variable for y location

# Initialize arrays with placeholder values
dt = np.full((imax, jmax), -99.9)
artviscx = np.full((imax, jmax), -99.9)
artviscy = np.full((imax, jmax), -99.9)
u = np.full((imax, jmax, neq), -99.9)
uold = np.full((imax, jmax, neq), -99.9)
s = np.full((imax, jmax, neq), -99.9)
ummsArray = np.full((imax, jmax, neq), -99.9)

# Set derived input quantities
rlength = xmax - xmin;                       # Characteristic length (m) [cavity width] */
rmu = rho*uinf*rlength/Re;                   # Viscosity (N*s/m^2) */
vel2ref = uinf*uinf;                         # Reference velocity squared (m^2/s^2) */
dx = (xmax - xmin)/(imax - 1);               # Delta x (m) */
dy = (ymax - ymin)/(jmax - 1);               # Delta y (m) */

vectorize = False

# Set up headers for output files
fp1, fp2 = output_file_headers(imms)  # Define this function separately

# Set Initial Profile for u vector
ninit, rtime, resinit, ummsArray = initialize(ninit, rtime, resinit, irstr, neq, uinf, pinf, xmax, xmin, ymax, ymin, u, s, ummsArray,
                                   rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy)  # Define this function separately

# Set Boundary Conditions for u
u = set_boundary_conditions(u, uinf, ummsArray, neq, imms)  # Define this function separately

# Write out initial conditions to solution file
write_output(n, resinit, rtime, xmax, xmin, ymax, ymin, imms, u, ummsArray, fp2) # Define this function separately

# Initialize Artificial Viscosity arrays to zero
artviscx[:, :] = 0
artviscy[:, :] = 0

# Evaluate Source Terms Once at Beginning
s = compute_source_terms(s, imax, jmax, imms, xmax, xmin, ymax, ymin,
                         rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy)  # Define this function separately

# ========== Main Loop ==========
isConverged = False

# Assuming required functions are defined:
# compute_time_step, Compute_Artificial_Viscosity, SGS_forward_sweep,
# set_boundary_conditions, SGS_backward_sweep, point_Jacobi, pressure_rescaling,
# check_iterative_convergence, write_output

starttime = time.time()
resPMatrix = []
convVector = []
# Main Loop
for n in range(ninit, nmax):
    # Calculate time step
    dt, dtmin = compute_time_step(u, vel2ref, rmu, rho, dx, dy, cfl, rkappa, vectorize)
    
    # Save u values at time level n (u and uold are 2D arrays)
    uold = np.copy(u)
    
    if solver_method == 'coupled':
        if isgs == 1:  # Symmetric Gauss-Seidel
            
            # Artificial Viscosity
            artviscx, artviscy = Compute_Artificial_Viscosity(u, dx, dy, Cx, Cy, vel2ref, rkappa, vectorize)
            
            # Symmetric Gauss-Siedel: Forward Sweep
            u = SGS_forward_sweep(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy)
            ## uncomment the below line and comment out the above line if you're considering to solve the Phase 2 BONUS section
            # u = SGS_forward_sweep_acc(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy)

            # Set Boundary Conditions for u
            u = set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize)
            
            # Artificial Viscosity
            artviscx, artviscy = Compute_Artificial_Viscosity(u, dx, dy, Cx, Cy, vel2ref, rkappa, vectorize)
            
            # Symmetric Gauss-Seidel: Backward Sweep
            u = SGS_backward_sweep(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy)
            ## uncomment the below line and comment out the above line if you're considering to solve the Phase 2 BONUS section
            # u = SGS_backward_sweep_acc(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy)

            # Set Boundary Conditions for u
            u = set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize)
        
        elif isgs == 0:  # Point Jacobi
            
            # Artificial Viscosity
            artviscx, artviscy = Compute_Artificial_Viscosity(u, dx, dy, Cx, Cy, vel2ref, rkappa, vectorize)
            
            # Point Jacobi: Forward Sweep
            u = point_Jacobi(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy, vectorize)
            
            # Set Boundary Conditions for u
            u = set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize)
        
        else:
            print('ERROR: isgs must equal 0 or 1!')
            break
            
    elif solver_method == 'fractional_step':
        # Step 1: Predictor (Intermediate Velocity)
        u = compute_intermediate_velocity(u, uold, dt, s, rho, dx, dy, rmu, vectorize)
        
        # Step 2: Boundary for Intermediate Velocity
        u = set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize)
        
        # Step 3: Solve Pressure Poisson Equation
        if isgs == 1:
            u = solve_PPE_SGS(u, dt, dx, dy, rho, toler=0.001, vectorize=vectorize)
            ## uncomment the below line and comment out the above line if you're considering to solve the Phase 2 BONUS section
            # u = solve_PPE_SGS_acc(u, dt, s, dx, dy, rho, p_toler=p_toler, iterations=iterations, vectorize=vectorize)

        elif isgs == 0:
            u = solve_PPE_Jacobi(u, dt, s, dx, dy, rho, p_toler=p_toler, iterations=iterations, vectorize=vectorize)
            
        # Step 4: Corrector (Update Velocity with new pressure field)
        u = correct_velocity(u, dt, dx, dy, rho, vectorize)
        
        # Step 5: Set final boundary conditions
        u = set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize)
        
    else:
        print('ERROR: unknown solver_method!')
        break
    
    # Pressure Rescaling (based on center point)
    u = pressure_rescaling(u, imms, xmax, xmin, ymax, ymin, pinf, imax, jmax,
                           k, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy)
    
    # Update the time
    rtime += dtmin
    
    # Check iterative convergence using L2 norms of iterative residuals
    res, resinit, conv = check_iterative_convergence(n, res, resinit, ninit, rtime, dtmin, u, uold, dt, imax, jmax, neq, fsmall, fp1, vectorize)
    
    # Store residuals for each equation
    resPMatrix.append(copy.deepcopy(res))
    convVector.append(copy.deepcopy(conv))
    
    # Check convergence
    if conv < toler:
        print(f'{n} {rtime:.6e} {res[0]:.6e} {res[1]:.6e} {res[2]:.6e}')
        isConverged = True
        break
    
    # Output solution and restart file every 'iterout' steps
    if n % iterout == 0:
        write_output(n, resinit, rtime, xmax, xmin, ymax, ymin, imms, u, ummsArray, fp2)

# ========== End Main Loop ==========
print('Elapsed time is ', (time.time() - starttime), "sec")
rL1norm, rL2norm, rLinfnorm = discretization_error_norms(imax, jmax, neq, imms, u, ummsArray)
print(rL1norm, rL2norm, rLinfnorm)
write_output(n, resinit, rtime, xmax, xmin, ymax, ymin, imms, u, ummsArray, fp2)

# ========== Plotting ==========
# Assuming imax, jmax, u, resPMatrix, and convVector are already defined
x = np.arange(1, imax + 1)
y = np.arange(1, jmax + 1)
X, Y = np.meshgrid(y, x)  # MATLAB's meshgrid flips x and y axes

fontsize=12
# Figure 1: Contour plot for u
plt.figure(1)
contour_u = plt.contourf(X, Y, u[:, :, 1].T, 10, linestyles='--')
plt.colorbar(contour_u)
plt.title('u (m/s)', fontsize=fontsize)
plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('ucontour.png')

# Figure 2: Contour plot for v
plt.figure(2)
contour_v = plt.contourf(X, Y, u[:, :, 2].T, 10, linestyles='--')
plt.colorbar(contour_v)
plt.title('v (m/s)', fontsize=fontsize)
plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('vcontour.png')

# Figure 3: Contour plot for p
plt.figure(3)
contour_p = plt.contourf(X, Y, u[:, :, 0].T, 10, linestyles='--')
plt.colorbar(contour_p)
plt.title('p (N/m^2)', fontsize=fontsize)
plt.xlabel('x', fontsize=fontsize)
plt.ylabel('y', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('pcontour.png')

resPMatrix = np.stack(resPMatrix).T
# Figure 4: Plot residuals for u, v, P
plt.figure(4)
plt.semilogy(resPMatrix[0], '-r', linewidth=2, label='u residual')
plt.semilogy(resPMatrix[1], '--b', linewidth=2, label='v residual')
plt.semilogy(resPMatrix[2], '-.k', linewidth=2, label='P residual')
plt.legend(loc='upper right', fontsize=fontsize, frameon=False)
plt.xlabel('iterations', fontsize=fontsize)
plt.ylabel('residual', fontsize=fontsize)
plt.title('Residual for u, v, P', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('residualcomponent.png')

# Figure 5: Plot overall residual convergence
plt.figure(5)
plt.semilogy(convVector, linewidth=3)
plt.title('Residual conv', fontsize=fontsize)
plt.xlabel('iterations', fontsize=fontsize)
plt.ylabel('residual', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig('residual.png')

# plt.show()
