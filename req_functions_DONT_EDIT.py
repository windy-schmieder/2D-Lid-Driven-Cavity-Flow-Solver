import numpy as np
# BONUS for phase 2 and 4 (Not compulsory)
import numba

################################## Time step ##################################

def compute_time_step(u, vel2ref, rmu, rho, dx, dy, cfl, rkappa, vectorize=False):
    """
    Compute the minimum time step for a CFD problem using local stability criteria.
    
    Parameters:
    u (numpy.ndarray): Velocity field array of shape (imax, jmax, 3)
                       u[:,:,1] - x-velocity, u[:,:,2] - y-velocity.
    dtmin (float): Initial value of dtmin, typically a large number.
    vel2ref (float): Reference velocity squared.
    rmu (float): Dynamic viscosity.
    rho (float): Fluid density.
    dx (float): Grid spacing in x-direction.
    dy (float): Grid spacing in y-direction.
    cfl (float): CFL number.
    rkappa (float): Preconditioning parameter.
    imax (int): Number of grid points in x-direction.
    jmax (int): Number of grid points in y-direction.

    Returns:
    float: Minimum time step (dtmin).

    HINT: Use the artificial compressibility wave speeds and viscous diffusion terms 
    to evaluate local time stepping `dt` safely bounded by the provided CFL condition.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    dt = np.zeros((imax, jmax))
    nu = rmu / rho

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

    if vectorize:
        pass
    else:
        pass

    return dt, np.min(dt)

################################## Artificial Viscosity ##################################

def Compute_Artificial_Viscosity(u, dx, dy, Cx, Cy, vel2ref, rkappa, vectorize=False):
    """
    Compute artificial viscosity terms in x and y directions.
    
    Parameters:
    - u: 3D numpy array of shape (imax, jmax, 3) representing the velocity and pressure field.
    - imax, jmax: Grid size in x and y directions.
    - rho: Density of the fluid.
    - dx, dy: Grid spacing in x and y directions.
    - Cx, Cy: Coefficients for artificial viscosity in x and y directions.
    - vel2ref: Reference velocity squared.
    - rkappa: Preconditioning parameter.
    
    Returns:
    - artviscx: 2D numpy array of shape (imax, jmax) representing artificial viscosity in the x-direction.
    - artviscy: 2D numpy array of shape (imax, jmax) representing artificial viscosity in the y-direction.    
    
    HINT: This stabilization term acts as a high-frequency spatial filter for the pressure field.
    You must evaluate the maximum local pseudo-acoustic wave speeds ($|\lambda|_{max}$) across the domain to properly scale this fourth-order damping parameter within the continuity update.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    artviscx = np.zeros((imax, jmax))
    artviscy = np.zeros((imax, jmax))

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    return artviscx, artviscy

################################## Boundary Conditions ##################################

def set_boundary_conditions(u, uinf, ummsArray, neq, imms, vectorize=False):
    """
    Sets the boundary conditions based on the value of 'imms'.
    
    Parameters:
    - u: 3D numpy array representing the velocity and pressure fields.
    - imms: Integer flag to determine which boundary condition to apply.
            - 0: Apply standard boundary conditions.
            - 1: Apply MMS (Manufactured Solution) boundary conditions.
    
    Modifies:
    - u: Updates the boundary values.
    
    Returns:
    - None
    """
    if imms == 0:
        u = bndry(u, uinf, vectorize)
    elif imms == 1:
        u = bndrymms(u, ummsArray, neq, vectorize)
    else:
        print('ERROR: imms must equal 0 or 1!')
    return u

def bndry(u, uinf, vectorize=False):
    """
    Apply boundary conditions for a lid-driven cavity.
    
    Parameters:
    - u: 3D numpy array representing the flow field [pressure, u-velocity, v-velocity]
    - uinf: Lid velocity (top boundary condition)
    
    Modifies:
    - u: Updates the boundary conditions directly on the array.    
    
    HINT: Enforce physical bounding box velocities (no penetration or slip),
    and set artificial pressure boundaries linearly interpolating interior points. 
    Use 2nd order extrpolation at the boundaries.
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    return u

def bndrymms(u, ummsArray, neq, vectorize=False):
    """
    Apply boundary conditions for the manufactured solution.

    Parameters:
    - u: 3D numpy array representing the flow field [pressure, u-velocity, v-velocity]
    - ummsArray: Manufactured solution array (same shape as u)
    - neq: Number of equations (typically 3 for [pressure, u, v])

    Modifies:
    - u: Updates the boundary conditions directly on the array.

    HINT: Anchor boundary values to the provided MMS exact analytical fields 
    so the interior accuracy can be strictly evaluated against them. You must 
    still linearly interpolate the pressure boundary values from the interior 
    cells of the physical domain. Use 2nd order extrpolation at the boundaries
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    return u

################################## Point Jacobi method ##################################

def point_Jacobi(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy, vectorize=False):
    """
    %Uses global variable(s): two, three, six, half
    %Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
    %                      xmax, xmin, ymax, ymin, rmu, vel2ref
    %Uses: artviscx, artviscy, dt, s
    %To Modify: u

    % i                        % i index (x direction)
    % j                        % j index (y direction)

    % dpdx         % First derivative of pressure w.r.t. x
    % dudx         % First derivative of x velocity w.r.t. x
    % dvdx         % First derivative of y velocity w.r.t. x
    % dpdy         % First derivative of pressure w.r.t. y
    % dudy         % First derivative of x velocity w.r.t. y
    % dvdy         % First derivative of y velocity w.r.t. y
    % d2udx2       % Second derivative of x velocity w.r.t. x
    % d2vdx2       % Second derivative of y velocity w.r.t. x
    % d2udy2       % Second derivative of x velocity w.r.t. y
    % d2vdy2       % Second derivative of y velocity w.r.t. y
    % beta2        % Beta squared parameter for time derivative preconditioning
    % uvel2        % Velocity squared

    HINT: Update U variables completely explicitly using interior array points from 
    the previous iteration sweep (uold), relying on the mapped spatial differences.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu * rhoinv

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass
    return u

################################## Symmetric Gauss-Seidel (SGS) ##################################

def SGS_forward_sweep(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy):
    """
    Perform the Symmetric Gauss-Seidel (SGS) forward sweep.
    
    Parameters:
    - u: 3D numpy array of shape (imax, jmax, 3) representing the current velocity and pressure fields.
    - uold: 3D numpy array of shape (imax, jmax, 3) representing the previous velocity and pressure fields.
    - dt: 2D numpy array of shape (imax, jmax) representing the time step at each grid point.
    - s: 3D numpy array of shape (imax, jmax, 3) representing source terms.
    - imax, jmax: Grid size in the x and y directions.
    - rho: Fluid density.
    - rhoinv: Inverse of fluid density (1/rho).
    - dx, dy: Grid spacing in the x and y directions.
    - rkappa: Preconditioning parameter for low-speed flows.
    - rmu: Dynamic viscosity.
    - vel2ref: Reference velocity squared.
    - artviscx: 2D numpy array of shape (imax, jmax) for artificial viscosity in the x-direction.
    - artviscy: 2D numpy array of shape (imax, jmax) for artificial viscosity in the y-direction.
    
    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Sweep through grid array indices [i][j] from bottom-left to top-right.
    Unlike Jacobi, immediately replace updated values within the sweep sequence!
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu * rhoinv


    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

    return u

def SGS_backward_sweep(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy):
    """
    Symmetric Gauss-Seidel: Backward Sweep for updating the flow field.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - uold: 3D numpy array representing the flow field from the previous iteration
    - dt: 2D numpy array for time step size at each grid point
    - s: 3D numpy array representing source terms
    - rho: Density
    - rhoinv: Inverse of density
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rkappa: Parameter for time derivative preconditioning
    - rmu: Viscosity
    - vel2ref: Reference velocity squared
    - artviscx: 2D numpy array for artificial viscosity in x direction
    - artviscy: 2D numpy array for artificial viscosity in y direction
    
    Modifies:
    - u: Updates the flow field directly on the array.

    HINT: Sweep through grid array indices [i][j] precisely in reverse! 
    Should you use u_old in the backward sweep? Think about the difference between 
    Gauss-Seidel and Jacobi methods.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu * rhoinv

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

    return u

@numba.njit(parallel=True, fastmath=True)
def SGS_forward_sweep_acc(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy):
    """
    u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    uold: 3D numpy array representing the flow field from the previous iteration
    dt: 2D numpy array for time step size at each grid point
    s: 3D numpy array representing source terms
    rho: Density
    rhoinv: Inverse of density
    dx: Grid spacing in the x direction
    dy: Grid spacing in the y direction
    rkappa: Parameter for time derivative preconditioning
    rmu: Viscosity
    vel2ref: Reference velocity squared
    artviscx: 2D numpy array for artificial viscosity in x direction
    artviscy: 2D numpy array for artificial viscosity in y direction

    HINT: Which nodes are not mutually dependent on each other during the forward sweep?
    Use Numba's parallel prange for traversing on these nodes.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu * rhoinv

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

@numba.njit(parallel=True, fastmath=True)
def SGS_backward_sweep_acc(u, uold, dt, s, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref, artviscx, artviscy):
    """
    u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    uold: 3D numpy array representing the flow field from the previous iteration
    dt: 2D numpy array for time step size at each grid point
    s: 3D numpy array representing source terms
    rho: Density
    rhoinv: Inverse of density
    dx: Grid spacing in the x direction
    dy: Grid spacing in the y direction
    rkappa: Parameter for time derivative preconditioning
    rmu: Viscosity
    vel2ref: Reference velocity squared
    artviscx: 2D numpy array for artificial viscosity in x direction
    artviscy: 2D numpy array for artificial viscosity in y direction

    HINT: Which nodes are not mutually dependent on each other during the backward sweep?
    Use Numba's parallel prange for traversing on these nodes.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu * rhoinv

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
################################## Convergence & Errors ##################################

def check_iterative_convergence(n, res, resinit, ninit, rtime, dtmin, u, uold, dt, imax, jmax, neq, fsmall, fp1, vectorize=False):
    """
    Checks the iterative convergence by calculating residuals and updating them over time.
    
    Parameters:
    - n: Current iteration number
    - res: Residuals array [P, u, v]
    - resinit: Initial residuals array [P, u, v]
    - ninit: Initial iteration number
    - rtime: Current runtime
    - dtmin: Minimum time step size
    - u: Current solution array (imax x jmax x neq)
    - uold: Solution from previous iteration (imax x jmax x neq)
    - dt: Time step array (imax x jmax)
    - imax: Number of grid points in x direction
    - jmax: Number of grid points in y direction
    - neq: Number of equations (3 for P, u, v)
    - fsmall: A small value for tolerance
    - fp1: File pointer for logging

    Returns:
    - res: Updated residuals
    - resinit: Updated initial residuals
    - conv: Convergence criterion (max relative residual)

    HINT: Calculate the maximum residual for each of the three equations (Pressure, u-velocity, v-velocity)
    by comparing the current solution (u) with the previous iteration's solution (uold).
    """
    for k in range(neq):
        res[k] = 0.0

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    if n == ninit:
        resinit = np.copy(res)
    conv = np.max(res)

    if (n % 10 == 0 or n == ninit):
        fp1.write(f'{n}   {rtime:.6e}   {res[0]:.6e}   {res[1]:.6e}   {res[2]:.6e}\n')

    if ((n+1) % 200 == 0 or n == ninit):
        print(f'{n}   {rtime:.6e}   {dtmin:.6e}   {res[0]:.6e}   {res[1]:.6e}   {res[2]:.6e}')

    return res, resinit, conv

def discretization_error_norms(imax, jmax, neq, imms, u, ummsArray):
    """
    Calculate the L1, L2, and Linf norms of the discretization error.

    Parameters:
    - imax, jmax: Grid size
    - neq: Number of equations
    - imms: Manufactured solution flag
    - u: Solution array
    - ummsArray: Manufactured solution array

    HINT: Calculate the L1, L2, and Linf norms of the discretization error
    by comparing the interior nodes of the current solution (u)
    with the manufactured solution (ummsArray). If not manufactured solution, 
    return zeros.
    """
    rL1norm = np.zeros(neq)
    rL2norm = np.zeros(neq)
    rLinfnorm = np.zeros(neq)

    if imms == 1:
        # !************************************************************** */
        # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
        # !************************************************************** */
        pass

    return rL1norm, rL2norm, rLinfnorm


#####################################################################################
################################## Fractional Step ##################################
#####################################################################################

def compute_intermediate_velocity(u, uold, dt, s, rho, dx, dy, rmu, vectorize=False):
    """
    Predictor: Calculate velocity field without pressure constraint implicitly.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - uold: 3D numpy array representing the flow field from the previous iteration
    - dt: 2D numpy array for time step size at each grid point
    - s: 3D numpy array representing source terms
    - rho: Density
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rmu: Viscosity
    - vectorize: Boolean flag for vectorized operations

    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Exclude mass projection dpdx gradients.
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    nu = rmu / rho

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    return u

def solve_PPE_Jacobi(u, dt, s, dx, dy, rho, p_toler=0.001, iterations=250, vectorize=False):
    """
    Solve Pressure Poisson Equation (PPE) via Point Jacobi formulation.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - dt: 2D numpy array for time step size at each grid point
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rho: Density
    - toler: Tolerance for convergence
    - vectorize: Boolean flag for vectorized operations

    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Update the pressure prediction by solving the PPE using the Jacobi iterative method.
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    return u

def solve_PPE_SGS(u, dt, s, dx, dy, rho, p_toler=0.001, iterations=250, vectorize=False):
    """
    Solve Pressure Poisson Equation (PPE) via implicit Symmetric Gauss Seidel.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - dt: 2D numpy array for time step size at each grid point
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rho: Density
    - p_toler: Tolerance for pressure convergence
    - iterations: Maximum number of iterations for Pressure Poisson Equation
    - vectorize: Boolean flag for vectorized operations

    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Execute iterative symmetric loops resolving spatial divergence of velocity.
    Break out of solver loop internally once local tolerances hit specified threshold.
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass

    return u
@numba.njit(parallel=True, fastmath=True)
def solve_PPE_SGS_acc(u, dt, s, dx, dy, rho, p_toler=0.001, iterations=250, vectorize=False):
    """
    Solve Pressure Poisson Equation (PPE) via implicit Symmetric Gauss Seidel.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - dt: 2D numpy array for time step size at each grid point
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rho: Density
    - p_toler: Tolerance for pressure convergence
    - iterations: Maximum number of iterations for Pressure Poisson Equation
    - vectorize: Boolean flag for vectorized operations

    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Find the nodes that are mutually independent and update them in parallel using numba.
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass
    return u

def correct_velocity(u, dt, dx, dy, rho, vectorize=False):
    """
    Corrector Stage: Modify spatial velocity mapping using converged PPE gradient.

    Parameters:
    - u: 3D numpy array representing the updated flow field [pressure, u-velocity, v-velocity]
    - dt: 2D numpy array for time step size at each grid point
    - dx: Grid spacing in the x direction
    - dy: Grid spacing in the y direction
    - rho: Density
    - vectorize: Boolean flag for vectorized operations

    Returns:
    - u: Updated 3D numpy array of shape (imax, jmax, 3) for velocity and pressure fields.

    HINT: Correct the intermediate momentum components explicitly against dpdx/dpdy arrays.
    """
    imax = u.shape[0]
    jmax = u.shape[1]

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    if vectorize:
        pass
    else:
        pass
    return u