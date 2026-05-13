"""
req_functions.py
MAE 4100 Final Project
Tomas Schmieder
"""

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

    eps = 1.0e-20
    if vectorize:
        uvel = u[1:-1, 1:-1, 1]
        vvel = u[1:-1, 1:-1, 2]
        beta2 = np.maximum(uvel * uvel + vvel * vvel, rkappa * vel2ref)

        lamx = 0.5 * (np.abs(uvel) + np.sqrt(uvel * uvel + 4.0 * beta2))
        lamy = 0.5 * (np.abs(vvel) + np.sqrt(vvel * vvel + 4.0 * beta2))

        dtconv = 1.0 / (lamx / dx + lamy / dy + eps)
        dtvis = 0.5 / (nu * (1.0 / (dx * dx) + 1.0 / (dy * dy)) + eps)
        dtloc = cfl * np.minimum(dtconv, dtvis)
        dt[1:-1, 1:-1] = dtloc
    else:
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                uvel = u[i, j, 1]
                vvel = u[i, j, 2]
                beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)
                lamx = 0.5 * (abs(uvel) + np.sqrt(uvel * uvel + 4.0 * beta2))
                lamy = 0.5 * (abs(vvel) + np.sqrt(vvel * vvel + 4.0 * beta2))

                dtconv = 1.0 / (lamx / dx + lamy / dy + eps)
                dtvis = 0.5 / (nu * (1.0 / (dx * dx) + 1.0 / (dy * dy)) + eps)
                dt[i, j] = cfl * min(dtconv, dtvis)

    dtmin = np.min(dt[1:-1, 1:-1])
    dt[0, :] = dtmin
    dt[-1, :] = dtmin
    dt[:, 0] = dtmin
    dt[:, -1] = dtmin

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
        uvel = u[1:-1, 1:-1, 1]
        vvel = u[1:-1, 1:-1, 2]
        beta2 = np.maximum(uvel * uvel + vvel * vvel, rkappa * vel2ref)
        lamx = 0.5 * (np.abs(uvel) + np.sqrt(uvel * uvel + 4.0 * beta2))
        lamy = 0.5 * (np.abs(vvel) + np.sqrt(vvel * vvel + 4.0 * beta2))
        lamxmax = np.max(lamx)
        lamymax = np.max(lamy)
    else:
        lamxmax = 0.0
        lamymax = 0.0
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                uvel = u[i, j, 1]
                vvel = u[i, j, 2]
                beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)
                lamx = 0.5 * (abs(uvel) + np.sqrt(uvel * uvel + 4.0 * beta2))
                lamy = 0.5 * (abs(vvel) + np.sqrt(vvel * vvel + 4.0 * beta2))
                lamxmax = max(lamxmax, lamx)
                lamymax = max(lamymax, lamy)

    artviscx[:, :] = -lamxmax * Cx * dx ** 3
    artviscy[:, :] = -lamymax * Cy * dy ** 3

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
        u[0, :, 1] = 0.0
        u[-1, :, 1] = 0.0
        u[:, 0, 1] = 0.0
        u[:, -1, 1] = uinf

        u[0, :, 2] = 0.0
        u[-1, :, 2] = 0.0
        u[:, 0, 2] = 0.0
        u[:, -1, 2] = 0.0

        u[0, 1:-1, 0] = 2.0 * u[1, 1:-1, 0] - u[2, 1:-1, 0]
        u[-1, 1:-1, 0] = 2.0 * u[-2, 1:-1, 0] - u[-3, 1:-1, 0]
        u[1:-1, 0, 0] = 2.0 * u[1:-1, 1, 0] - u[1:-1, 2, 0]
        u[1:-1, -1, 0] = 2.0 * u[1:-1, -2, 0] - u[1:-1, -3, 0]
        u[0, 0, 0] = 0.5 * (u[1, 0, 0] + u[0, 1, 0])
        u[0, -1, 0] = 0.5 * (u[1, -1, 0] + u[0, -2, 0])
        u[-1, 0, 0] = 0.5 * (u[-2, 0, 0] + u[-1, 1, 0])
        u[-1, -1, 0] = 0.5 * (u[-2, -1, 0] + u[-1, -2, 0])
    else:
        for j in range(jmax):
            u[0, j, 1] = 0.0
            u[imax - 1, j, 1] = 0.0
            u[0, j, 2] = 0.0
            u[imax - 1, j, 2] = 0.0

        for i in range(imax):
            u[i, 0, 1] = 0.0
            u[i, jmax - 1, 1] = uinf
            u[i, 0, 2] = 0.0
            u[i, jmax - 1, 2] = 0.0

        for j in range(1, jmax - 1):
            u[0, j, 0] = 2.0 * u[1, j, 0] - u[2, j, 0]
            u[imax - 1, j, 0] = 2.0 * u[imax - 2, j, 0] - u[imax - 3, j, 0]

        for i in range(1, imax - 1):
            u[i, 0, 0] = 2.0 * u[i, 1, 0] - u[i, 2, 0]
            u[i, jmax - 1, 0] = 2.0 * u[i, jmax - 2, 0] - u[i, jmax - 3, 0]

        u[0, 0, 0] = 0.5 * (u[1, 0, 0] + u[0, 1, 0])
        u[0, jmax - 1, 0] = 0.5 * (u[1, jmax - 1, 0] + u[0, jmax - 2, 0])
        u[imax - 1, 0, 0] = 0.5 * (u[imax - 2, 0, 0] + u[imax - 1, 1, 0])
        u[imax - 1, jmax - 1, 0] = 0.5 * (u[imax - 2, jmax - 1, 0] + u[imax - 1, jmax - 2, 0])

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
        u[0, :, 1] = ummsArray[0, :, 1]
        u[-1, :, 1] = ummsArray[-1, :, 1]
        u[:, 0, 1] = ummsArray[:, 0, 1]
        u[:, -1, 1] = ummsArray[:, -1, 1]

        u[0, :, 2] = ummsArray[0, :, 2]
        u[-1, :, 2] = ummsArray[-1, :, 2]
        u[:, 0, 2] = ummsArray[:, 0, 2]
        u[:, -1, 2] = ummsArray[:, -1, 2]

        u[0, 1:-1, 0] = 2.0 * u[1, 1:-1, 0] - u[2, 1:-1, 0]
        u[-1, 1:-1, 0] = 2.0 * u[-2, 1:-1, 0] - u[-3, 1:-1, 0]
        u[1:-1, 0, 0] = 2.0 * u[1:-1, 1, 0] - u[1:-1, 2, 0]
        u[1:-1, -1, 0] = 2.0 * u[1:-1, -2, 0] - u[1:-1, -3, 0]
        u[0, 0, 0] = 0.5 * (u[1, 0, 0] + u[0, 1, 0])
        u[0, -1, 0] = 0.5 * (u[1, -1, 0] + u[0, -2, 0])
        u[-1, 0, 0] = 0.5 * (u[-2, 0, 0] + u[-1, 1, 0])
        u[-1, -1, 0] = 0.5 * (u[-2, -1, 0] + u[-1, -2, 0])
    else:
        for j in range(jmax):
            u[0, j, 1] = ummsArray[0, j, 1]
            u[imax - 1, j, 1] = ummsArray[imax - 1, j, 1]
            u[0, j, 2] = ummsArray[0, j, 2]
            u[imax - 1, j, 2] = ummsArray[imax - 1, j, 2]

        for i in range(imax):
            u[i, 0, 1] = ummsArray[i, 0, 1]
            u[i, jmax - 1, 1] = ummsArray[i, jmax - 1, 1]
            u[i, 0, 2] = ummsArray[i, 0, 2]
            u[i, jmax - 1, 2] = ummsArray[i, jmax - 1, 2]

        for j in range(1, jmax - 1):
            u[0, j, 0] = 2.0 * u[1, j, 0] - u[2, j, 0]
            u[imax - 1, j, 0] = 2.0 * u[imax - 2, j, 0] - u[imax - 3, j, 0]

        for i in range(1, imax - 1):
            u[i, 0, 0] = 2.0 * u[i, 1, 0] - u[i, 2, 0]
            u[i, jmax - 1, 0] = 2.0 * u[i, jmax - 2, 0] - u[i, jmax - 3, 0]

        u[0, 0, 0] = 0.5 * (u[1, 0, 0] + u[0, 1, 0])
        u[0, jmax - 1, 0] = 0.5 * (u[1, jmax - 1, 0] + u[0, jmax - 2, 0])
        u[imax - 1, 0, 0] = 0.5 * (u[imax - 2, 0, 0] + u[imax - 1, 1, 0])
        u[imax - 1, jmax - 1, 0] = 0.5 * (u[imax - 2, jmax - 1, 0] + u[imax - 1, jmax - 2, 0])

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

    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    if vectorize:
        uvel = uold[1:-1, 1:-1, 1]
        vvel = uold[1:-1, 1:-1, 2]
        beta2 = np.maximum(uvel * uvel + vvel * vvel, rkappa * vel2ref)

        dpdx = (uold[2:, 1:-1, 0] - uold[:-2, 1:-1, 0]) * inv2dx
        dpdy = (uold[1:-1, 2:, 0] - uold[1:-1, :-2, 0]) * inv2dy

        dudx = (uold[2:, 1:-1, 1] - uold[:-2, 1:-1, 1]) * inv2dx
        dudy = (uold[1:-1, 2:, 1] - uold[1:-1, :-2, 1]) * inv2dy
        dvdx = (uold[2:, 1:-1, 2] - uold[:-2, 1:-1, 2]) * inv2dx
        dvdy = (uold[1:-1, 2:, 2] - uold[1:-1, :-2, 2]) * inv2dy

        d2udx2 = (uold[2:, 1:-1, 1] - 2.0 * uold[1:-1, 1:-1, 1] + uold[:-2, 1:-1, 1]) * idx2
        d2udy2 = (uold[1:-1, 2:, 1] - 2.0 * uold[1:-1, 1:-1, 1] + uold[1:-1, :-2, 1]) * idy2
        d2vdx2 = (uold[2:, 1:-1, 2] - 2.0 * uold[1:-1, 1:-1, 2] + uold[:-2, 1:-1, 2]) * idx2
        d2vdy2 = (uold[1:-1, 2:, 2] - 2.0 * uold[1:-1, 1:-1, 2] + uold[1:-1, :-2, 2]) * idy2

        d4pdx4 = np.zeros((imax - 2, jmax - 2))
        d4pdy4 = np.zeros((imax - 2, jmax - 2))
        if imax > 4:
            d4pdx4[1:-1, :] = (
                uold[:-4, 1:-1, 0] - 4.0 * uold[1:-3, 1:-1, 0] + 6.0 * uold[2:-2, 1:-1, 0] - 4.0 * uold[3:-1, 1:-1, 0] + uold[4:, 1:-1, 0]) / (dx ** 4)
        if jmax > 4:
            d4pdy4[:, 1:-1] = (uold[1:-1, :-4, 0] - 4.0 * uold[1:-1, 1:-3, 0] + 6.0 * uold[1:-1, 2:-2, 0] - 4.0 * uold[1:-1, 3:-1, 0] + uold[1:-1, 4:, 0]) / (dy ** 4)

        cont_rhs = (s[1:-1, 1:-1, 0] - rho * (dudx + dvdy) + artviscx[1:-1, 1:-1] * d4pdx4 + artviscy[1:-1, 1:-1] * d4pdy4)
        xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[1:-1, 1:-1, 1]
        ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[1:-1, 1:-1, 2]

        u[1:-1, 1:-1, 0] = uold[1:-1, 1:-1, 0] + dt[1:-1, 1:-1] * beta2 * cont_rhs
        u[1:-1, 1:-1, 1] = uold[1:-1, 1:-1, 1] + dt[1:-1, 1:-1] * rhoinv * xmom_rhs
        u[1:-1, 1:-1, 2] = uold[1:-1, 1:-1, 2] + dt[1:-1, 1:-1] * rhoinv * ymom_rhs
    else:
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                uvel = uold[i, j, 1]
                vvel = uold[i, j, 2]
                beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)

                dpdx = (uold[i + 1, j, 0] - uold[i - 1, j, 0]) * inv2dx
                dpdy = (uold[i, j + 1, 0] - uold[i, j - 1, 0]) * inv2dy

                dudx = (uold[i + 1, j, 1] - uold[i - 1, j, 1]) * inv2dx
                dudy = (uold[i, j + 1, 1] - uold[i, j - 1, 1]) * inv2dy
                dvdx = (uold[i + 1, j, 2] - uold[i - 1, j, 2]) * inv2dx
                dvdy = (uold[i, j + 1, 2] - uold[i, j - 1, 2]) * inv2dy

                d2udx2 = (uold[i + 1, j, 1] - 2.0 * uold[i, j, 1] + uold[i - 1, j, 1]) * idx2
                d2udy2 = (uold[i, j + 1, 1] - 2.0 * uold[i, j, 1] + uold[i, j - 1, 1]) * idy2
                d2vdx2 = (uold[i + 1, j, 2] - 2.0 * uold[i, j, 2] + uold[i - 1, j, 2]) * idx2
                d2vdy2 = (uold[i, j + 1, 2] - 2.0 * uold[i, j, 2] + uold[i, j - 1, 2]) * idy2

                d4pdx4 = 0.0
                d4pdy4 = 0.0
                if 1 < i < imax - 2:
                    d4pdx4 = (uold[i - 2, j, 0] - 4.0 * uold[i - 1, j, 0] + 6.0 * uold[i, j, 0] - 4.0 * uold[i + 1, j, 0] + uold[i + 2, j, 0]) / (dx ** 4)
                if 1 < j < jmax - 2:
                    d4pdy4 = (uold[i, j - 2, 0] - 4.0 * uold[i, j - 1, 0] + 6.0 * uold[i, j, 0] - 4.0 * uold[i, j + 1, 0] + uold[i, j + 2, 0]) / (dy ** 4)

                cont_rhs = (s[i, j, 0] - rho * (dudx + dvdy) + artviscx[i, j] * d4pdx4 + artviscy[i, j] * d4pdy4)
                xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[i, j, 1]
                ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[i, j, 2]

                u[i, j, 0] = uold[i, j, 0] + dt[i, j] * beta2 * cont_rhs
                u[i, j, 1] = uold[i, j, 1] + dt[i, j] * rhoinv * xmom_rhs
                u[i, j, 2] = uold[i, j, 2] + dt[i, j] * rhoinv * ymom_rhs
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

    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    for i in range(1, imax - 1):
        for j in range(1, jmax - 1):
            uvel = u[i, j, 1]
            vvel = u[i, j, 2]
            beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)

            dpdx = (u[i + 1, j, 0] - u[i - 1, j, 0]) * inv2dx
            dpdy = (u[i, j + 1, 0] - u[i, j - 1, 0]) * inv2dy

            dudx = (u[i + 1, j, 1] - u[i - 1, j, 1]) * inv2dx
            dudy = (u[i, j + 1, 1] - u[i, j - 1, 1]) * inv2dy
            dvdx = (u[i + 1, j, 2] - u[i - 1, j, 2]) * inv2dx
            dvdy = (u[i, j + 1, 2] - u[i, j - 1, 2]) * inv2dy

            d2udx2 = (u[i + 1, j, 1] - 2.0 * u[i, j, 1] + u[i - 1, j, 1]) * idx2
            d2udy2 = (u[i, j + 1, 1] - 2.0 * u[i, j, 1] + u[i, j - 1, 1]) * idy2
            d2vdx2 = (u[i + 1, j, 2] - 2.0 * u[i, j, 2] + u[i - 1, j, 2]) * idx2
            d2vdy2 = (u[i, j + 1, 2] - 2.0 * u[i, j, 2] + u[i, j - 1, 2]) * idy2

            d4pdx4 = 0.0
            d4pdy4 = 0.0
            if 1 < i < imax - 2:
                d4pdx4 = (u[i - 2, j, 0] - 4.0 * u[i - 1, j, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i + 1, j, 0] + u[i + 2, j, 0]) / (dx ** 4)
            if 1 < j < jmax - 2:
                d4pdy4 = (u[i, j - 2, 0] - 4.0 * u[i, j - 1, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i, j + 1, 0] + u[i, j + 2, 0]) / (dy ** 4)

            cont_rhs = s[i, j, 0] - rho * (dudx + dvdy) + artviscx[i, j] * d4pdx4 + artviscy[i, j] * d4pdy4
            xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[i, j, 1]
            ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[i, j, 2]

            u[i, j, 0] += dt[i, j] * beta2 * cont_rhs
            u[i, j, 1] += dt[i, j] * rhoinv * xmom_rhs
            u[i, j, 2] += dt[i, j] * rhoinv * ymom_rhs

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

    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    for i in range(imax - 2, 0, -1):
        for j in range(jmax - 2, 0, -1):
            uvel = u[i, j, 1]
            vvel = u[i, j, 2]
            beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)

            dpdx = (u[i + 1, j, 0] - u[i - 1, j, 0]) * inv2dx
            dpdy = (u[i, j + 1, 0] - u[i, j - 1, 0]) * inv2dy

            dudx = (u[i + 1, j, 1] - u[i - 1, j, 1]) * inv2dx
            dudy = (u[i, j + 1, 1] - u[i, j - 1, 1]) * inv2dy
            dvdx = (u[i + 1, j, 2] - u[i - 1, j, 2]) * inv2dx
            dvdy = (u[i, j + 1, 2] - u[i, j - 1, 2]) * inv2dy

            d2udx2 = (u[i + 1, j, 1] - 2.0 * u[i, j, 1] + u[i - 1, j, 1]) * idx2
            d2udy2 = (u[i, j + 1, 1] - 2.0 * u[i, j, 1] + u[i, j - 1, 1]) * idy2
            d2vdx2 = (u[i + 1, j, 2] - 2.0 * u[i, j, 2] + u[i - 1, j, 2]) * idx2
            d2vdy2 = (u[i, j + 1, 2] - 2.0 * u[i, j, 2] + u[i, j - 1, 2]) * idy2

            d4pdx4 = 0.0
            d4pdy4 = 0.0
            if 1 < i < imax - 2:
                d4pdx4 = (u[i - 2, j, 0] - 4.0 * u[i - 1, j, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i + 1, j, 0] + u[i + 2, j, 0]) / (dx ** 4)
            if 1 < j < jmax - 2:
                d4pdy4 = (u[i, j - 2, 0] - 4.0 * u[i, j - 1, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i, j + 1, 0] + u[i, j + 2, 0]) / (dy ** 4)

            cont_rhs = s[i, j, 0] - rho * (dudx + dvdy) + artviscx[i, j] * d4pdx4 + artviscy[i, j] * d4pdy4
            xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[i, j, 1]
            ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[i, j, 2]

            u[i, j, 0] += dt[i, j] * beta2 * cont_rhs
            u[i, j, 1] += dt[i, j] * rhoinv * xmom_rhs
            u[i, j, 2] += dt[i, j] * rhoinv * ymom_rhs

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
    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    dx4 = dx * dx * dx * dx
    dy4 = dy * dy * dy * dy

    # Wavefront ordering preserves Gauss-Seidel dependencies while letting
    # nodes with the same i+j index update in parallel.
    for diag in range(2, imax + jmax - 3):
        for i in numba.prange(1, imax - 1):
            j = diag - i
            if 1 <= j < jmax - 1:
                uvel = u[i, j, 1]
                vvel = u[i, j, 2]
                beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)

                dpdx = (u[i + 1, j, 0] - u[i - 1, j, 0]) * inv2dx
                dpdy = (u[i, j + 1, 0] - u[i, j - 1, 0]) * inv2dy

                dudx = (u[i + 1, j, 1] - u[i - 1, j, 1]) * inv2dx
                dudy = (u[i, j + 1, 1] - u[i, j - 1, 1]) * inv2dy
                dvdx = (u[i + 1, j, 2] - u[i - 1, j, 2]) * inv2dx
                dvdy = (u[i, j + 1, 2] - u[i, j - 1, 2]) * inv2dy

                d2udx2 = (u[i + 1, j, 1] - 2.0 * u[i, j, 1] + u[i - 1, j, 1]) * idx2
                d2udy2 = (u[i, j + 1, 1] - 2.0 * u[i, j, 1] + u[i, j - 1, 1]) * idy2
                d2vdx2 = (u[i + 1, j, 2] - 2.0 * u[i, j, 2] + u[i - 1, j, 2]) * idx2
                d2vdy2 = (u[i, j + 1, 2] - 2.0 * u[i, j, 2] + u[i, j - 1, 2]) * idy2

                d4pdx4 = 0.0
                d4pdy4 = 0.0
                if 1 < i < imax - 2:
                    d4pdx4 = (u[i - 2, j, 0] - 4.0 * u[i - 1, j, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i + 1, j, 0] + u[i + 2, j, 0]) / dx4
                if 1 < j < jmax - 2:
                    d4pdy4 = (u[i, j - 2, 0] - 4.0 * u[i, j - 1, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i, j + 1, 0] + u[i, j + 2, 0]) / dy4

                cont_rhs = s[i, j, 0] - rho * (dudx + dvdy) + artviscx[i, j] * d4pdx4 + artviscy[i, j] * d4pdy4
                xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[i, j, 1]
                ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[i, j, 2]

                u[i, j, 0] += dt[i, j] * beta2 * cont_rhs
                u[i, j, 1] += dt[i, j] * rhoinv * xmom_rhs
                u[i, j, 2] += dt[i, j] * rhoinv * ymom_rhs

    return u

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
    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    dx4 = dx * dx * dx * dx
    dy4 = dy * dy * dy * dy

    for diag in range(imax + jmax - 4, 1, -1):
        for i in numba.prange(1, imax - 1):
            j = diag - i
            if 1 <= j < jmax - 1:
                uvel = u[i, j, 1]
                vvel = u[i, j, 2]
                beta2 = max(uvel * uvel + vvel * vvel, rkappa * vel2ref)

                dpdx = (u[i + 1, j, 0] - u[i - 1, j, 0]) * inv2dx
                dpdy = (u[i, j + 1, 0] - u[i, j - 1, 0]) * inv2dy

                dudx = (u[i + 1, j, 1] - u[i - 1, j, 1]) * inv2dx
                dudy = (u[i, j + 1, 1] - u[i, j - 1, 1]) * inv2dy
                dvdx = (u[i + 1, j, 2] - u[i - 1, j, 2]) * inv2dx
                dvdy = (u[i, j + 1, 2] - u[i, j - 1, 2]) * inv2dy

                d2udx2 = (u[i + 1, j, 1] - 2.0 * u[i, j, 1] + u[i - 1, j, 1]) * idx2
                d2udy2 = (u[i, j + 1, 1] - 2.0 * u[i, j, 1] + u[i, j - 1, 1]) * idy2
                d2vdx2 = (u[i + 1, j, 2] - 2.0 * u[i, j, 2] + u[i - 1, j, 2]) * idx2
                d2vdy2 = (u[i, j + 1, 2] - 2.0 * u[i, j, 2] + u[i, j - 1, 2]) * idy2

                d4pdx4 = 0.0
                d4pdy4 = 0.0
                if 1 < i < imax - 2:
                    d4pdx4 = (u[i - 2, j, 0] - 4.0 * u[i - 1, j, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i + 1, j, 0] + u[i + 2, j, 0]) / dx4
                if 1 < j < jmax - 2:
                    d4pdy4 = (u[i, j - 2, 0] - 4.0 * u[i, j - 1, 0] + 6.0 * u[i, j, 0] - 4.0 * u[i, j + 1, 0] + u[i, j + 2, 0]) / dy4

                cont_rhs = s[i, j, 0] - rho * (dudx + dvdy) + artviscx[i, j] * d4pdx4 + artviscy[i, j] * d4pdy4
                xmom_rhs = -rho * (uvel * dudx + vvel * dudy) - dpdx + rmu * (d2udx2 + d2udy2) + s[i, j, 1]
                ymom_rhs = -rho * (uvel * dvdx + vvel * dvdy) - dpdy + rmu * (d2vdx2 + d2vdy2) + s[i, j, 2]

                u[i, j, 0] += dt[i, j] * beta2 * cont_rhs
                u[i, j, 1] += dt[i, j] * rhoinv * xmom_rhs
                u[i, j, 2] += dt[i, j] * rhoinv * ymom_rhs

    return u
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
        for k in range(neq):
            delta = np.abs((u[1:-1, 1:-1, k] - uold[1:-1, 1:-1, k]) / np.maximum(dt[1:-1, 1:-1], fsmall))
            res[k] = np.max(delta)
    else:
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                invdt = 1.0 / max(dt[i, j], fsmall)
                for k in range(neq):
                    res[k] = max(res[k], abs((u[i, j, k] - uold[i, j, k]) * invdt))

    if n == ninit:
        resinit = np.copy(res)
    res[:] = res[:] / np.maximum(resinit, fsmall)
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
        npts = max((imax - 2) * (jmax - 2), 1)
        diff = u[1:-1, 1:-1, :] - ummsArray[1:-1, 1:-1, :]
        rL1norm[:] = np.sum(np.abs(diff), axis=(0, 1)) / npts
        rL2norm[:] = np.sqrt(np.sum(diff * diff, axis=(0, 1)) / npts)
        rLinfnorm[:] = np.max(np.abs(diff), axis=(0, 1))




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

    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    if vectorize:
        uvel = uold[1:-1, 1:-1, 1]
        vvel = uold[1:-1, 1:-1, 2]

        dudx = (uold[2:, 1:-1, 1] - uold[:-2, 1:-1, 1]) * inv2dx
        dudy = (uold[1:-1, 2:, 1] - uold[1:-1, :-2, 1]) * inv2dy
        dvdx = (uold[2:, 1:-1, 2] - uold[:-2, 1:-1, 2]) * inv2dx
        dvdy = (uold[1:-1, 2:, 2] - uold[1:-1, :-2, 2]) * inv2dy

        d2udx2 = (uold[2:, 1:-1, 1] - 2.0 * uold[1:-1, 1:-1, 1] + uold[:-2, 1:-1, 1]) * idx2
        d2udy2 = (uold[1:-1, 2:, 1] - 2.0 * uold[1:-1, 1:-1, 1] + uold[1:-1, :-2, 1]) * idy2
        d2vdx2 = (uold[2:, 1:-1, 2] - 2.0 * uold[1:-1, 1:-1, 2] + uold[:-2, 1:-1, 2]) * idx2
        d2vdy2 = (uold[1:-1, 2:, 2] - 2.0 * uold[1:-1, 1:-1, 2] + uold[1:-1, :-2, 2]) * idy2

        u_rhs = -uvel * dudx - vvel * dudy + nu * (d2udx2 + d2udy2) + s[1:-1, 1:-1, 1] / rho
        v_rhs = -uvel * dvdx - vvel * dvdy + nu * (d2vdx2 + d2vdy2) + s[1:-1, 1:-1, 2] / rho

        u[1:-1, 1:-1, 1] = uold[1:-1, 1:-1, 1] + dt[1:-1, 1:-1] * u_rhs
        u[1:-1, 1:-1, 2] = uold[1:-1, 1:-1, 2] + dt[1:-1, 1:-1] * v_rhs
    else:
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                uvel = uold[i, j, 1]
                vvel = uold[i, j, 2]

                dudx = (uold[i + 1, j, 1] - uold[i - 1, j, 1]) * inv2dx
                dudy = (uold[i, j + 1, 1] - uold[i, j - 1, 1]) * inv2dy
                dvdx = (uold[i + 1, j, 2] - uold[i - 1, j, 2]) * inv2dx
                dvdy = (uold[i, j + 1, 2] - uold[i, j - 1, 2]) * inv2dy

                d2udx2 = (uold[i + 1, j, 1] - 2.0 * uold[i, j, 1] + uold[i - 1, j, 1]) * idx2
                d2udy2 = (uold[i, j + 1, 1] - 2.0 * uold[i, j, 1] + uold[i, j - 1, 1]) * idy2
                d2vdx2 = (uold[i + 1, j, 2] - 2.0 * uold[i, j, 2] + uold[i - 1, j, 2]) * idx2
                d2vdy2 = (uold[i, j + 1, 2] - 2.0 * uold[i, j, 2] + uold[i, j - 1, 2]) * idy2

                u_rhs = -uvel * dudx - vvel * dudy + nu * (d2udx2 + d2udy2) + s[i, j, 1] / rho
                v_rhs = -uvel * dvdx - vvel * dvdy + nu * (d2vdx2 + d2vdy2) + s[i, j, 2] / rho

                u[i, j, 1] = uold[i, j, 1] + dt[i, j] * u_rhs
                u[i, j, 2] = uold[i, j, 2] + dt[i, j] * v_rhs
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
    idx = 1.0 / (2.0 * dx)
    idy = 1.0 / (2.0 * dy)
    dx2 = dx * dx
    dy2 = dy * dy
    coeff = 1.0 / (2.0 * (dx2 + dy2))

    p = u[:, :, 0].copy()
    smass = np.zeros((imax, jmax)) if s is None else s[:, :, 0]

    if vectorize:
        for _ in range(iterations):
            pold = p.copy()

            divu = ((u[2:, 1:-1, 1] - u[:-2, 1:-1, 1]) * idx + (u[1:-1, 2:, 2] - u[1:-1, :-2, 2]) * idy)
            rhs = rho / np.maximum(dt[1:-1, 1:-1], 1.0e-20) * (divu - smass[1:-1, 1:-1] / rho)

            p[1:-1, 1:-1] = ((pold[2:, 1:-1] + pold[:-2, 1:-1]) * dy2 + (pold[1:-1, 2:] + pold[1:-1, :-2]) * dx2 - rhs * dx2 * dy2) * coeff

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

            if np.max(np.abs(p - pold)) < p_toler:
                break
    else:
        for _ in range(iterations):
            pold = p.copy()

            for i in range(1, imax - 1):
                for j in range(1, jmax - 1):
                    divu = (
                        (u[i + 1, j, 1] - u[i - 1, j, 1]) * idx
                        + (u[i, j + 1, 2] - u[i, j - 1, 2]) * idy
                    )
                    rhs = rho / max(dt[i, j], 1.0e-20) * (
                        divu - smass[i, j] / rho
                    )

                    p[i, j] = (
                        (pold[i + 1, j] + pold[i - 1, j]) * dy2
                        + (pold[i, j + 1] + pold[i, j - 1]) * dx2
                        - rhs * dx2 * dy2
                    ) * coeff

            # PPE uses zero-gradient Neumann pressure boundaries.
            for j in range(jmax):
                p[0, j] = p[1, j]
                p[imax - 1, j] = p[imax - 2, j]

            for i in range(imax):
                p[i, 0] = p[i, 1]
                p[i, jmax - 1] = p[i, jmax - 2]

            if np.max(np.abs(p - pold)) < p_toler:
                break

    u[:, :, 0] = p

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
    idx = 1.0 / (2.0 * dx)
    idy = 1.0 / (2.0 * dy)
    dx2 = dx * dx
    dy2 = dy * dy
    coeff = 1.0 / (2.0 * (dx2 + dy2))

    p = u[:, :, 0]
    smass = np.zeros((imax, jmax)) if s is None else s[:, :, 0]

    if vectorize:
        # SGS is inherently sequential; this branch keeps array-level RHS setup
        # while preserving forward/backward Gauss-Seidel pressure updates.
        for _ in range(iterations):
            p_prev = p.copy()
            divu = (
                (u[2:, 1:-1, 1] - u[:-2, 1:-1, 1]) * idx
                + (u[1:-1, 2:, 2] - u[1:-1, :-2, 2]) * idy
            )
            rhs = rho / np.maximum(dt[1:-1, 1:-1], 1.0e-20) * (
                divu - smass[1:-1, 1:-1] / rho
            )

            for i in range(1, imax - 1):
                for j in range(1, jmax - 1):
                    p[i, j] = (
                        (p[i + 1, j] + p[i - 1, j]) * dy2
                        + (p[i, j + 1] + p[i, j - 1]) * dx2
                        - rhs[i - 1, j - 1] * dx2 * dy2
                    ) * coeff

            for i in range(imax - 2, 0, -1):
                for j in range(jmax - 2, 0, -1):
                    p[i, j] = (
                        (p[i + 1, j] + p[i - 1, j]) * dy2
                        + (p[i, j + 1] + p[i, j - 1]) * dx2
                        - rhs[i - 1, j - 1] * dx2 * dy2
                    ) * coeff

            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

            if np.max(np.abs(p - p_prev)) < p_toler:
                break
    else:
        for _ in range(iterations):
            p_prev = p.copy()

            for i in range(1, imax - 1):
                for j in range(1, jmax - 1):
                    divu = ((u[i + 1, j, 1] - u[i - 1, j, 1]) * idx + (u[i, j + 1, 2] - u[i, j - 1, 2]) * idy)
                    rhs = rho / max(dt[i, j], 1.0e-20) * (divu - smass[i, j] / rho)
                    p[i, j] = ((p[i + 1, j] + p[i - 1, j]) * dy2 + (p[i, j + 1] + p[i, j - 1]) * dx2 - rhs * dx2 * dy2) * coeff

            for i in range(imax - 2, 0, -1):
                for j in range(jmax - 2, 0, -1):
                    divu = ((u[i + 1, j, 1] - u[i - 1, j, 1]) * idx + (u[i, j + 1, 2] - u[i, j - 1, 2]) * idy)
                    rhs = rho / max(dt[i, j], 1.0e-20) * (divu - smass[i, j] / rho)
                    p[i, j] = ((p[i + 1, j] + p[i - 1, j]) * dy2 + (p[i, j + 1] + p[i, j - 1]) * dx2 - rhs * dx2 * dy2) * coeff

            for j in range(jmax):
                p[0, j] = p[1, j]
                p[imax - 1, j] = p[imax - 2, j]

            for i in range(imax):
                p[i, 0] = p[i, 1]
                p[i, jmax - 1] = p[i, jmax - 2]

            if np.max(np.abs(p - p_prev)) < p_toler:
                break

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
    idx = 1.0 / (2.0 * dx)
    idy = 1.0 / (2.0 * dy)
    dx2 = dx * dx
    dy2 = dy * dy
    coeff = 1.0 / (2.0 * (dx2 + dy2))

    p = u[:, :, 0]
    for _ in range(iterations):
        p_prev = p.copy()

        for diag in range(2, imax + jmax - 3):
            for i in numba.prange(1, imax - 1):
                j = diag - i
                if 1 <= j < jmax - 1:
                    divu = ((u[i + 1, j, 1] - u[i - 1, j, 1]) * idx + (u[i, j + 1, 2] - u[i, j - 1, 2]) * idy)
                    smass = 0.0
                    if s is not None:
                        smass = s[i, j, 0]
                    rhs = rho / max(dt[i, j], 1.0e-20) * (divu - smass / rho)
                    p[i, j] = ((p[i + 1, j] + p[i - 1, j]) * dy2 + (p[i, j + 1] + p[i, j - 1]) * dx2 - rhs * dx2 * dy2) * coeff

        for diag in range(imax + jmax - 4, 1, -1):
            for i in numba.prange(1, imax - 1):
                j = diag - i
                if 1 <= j < jmax - 1:
                    divu = ((u[i + 1, j, 1] - u[i - 1, j, 1]) * idx + (u[i, j + 1, 2] - u[i, j - 1, 2]) * idy)
                    smass = 0.0
                    if s is not None:
                        smass = s[i, j, 0]
                    rhs = rho / max(dt[i, j], 1.0e-20) * (divu - smass / rho)
                    p[i, j] = ((p[i + 1, j] + p[i - 1, j]) * dy2 + (p[i, j + 1] + p[i, j - 1]) * dx2 - rhs * dx2 * dy2) * coeff

        for j in numba.prange(jmax):
            p[0, j] = p[1, j]
            p[imax - 1, j] = p[imax - 2, j]

        for i in numba.prange(imax):
            p[i, 0] = p[i, 1]
            p[i, jmax - 1] = p[i, jmax - 2]

        if np.max(np.abs(p - p_prev)) < p_toler:
            break

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

    inv2dx = 0.5 / dx
    inv2dy = 0.5 / dy

    if vectorize:
        dpdx = (u[2:, 1:-1, 0] - u[:-2, 1:-1, 0]) * inv2dx
        dpdy = (u[1:-1, 2:, 0] - u[1:-1, :-2, 0]) * inv2dy
        u[1:-1, 1:-1, 1] -= dt[1:-1, 1:-1] * dpdx / rho
        u[1:-1, 1:-1, 2] -= dt[1:-1, 1:-1] * dpdy / rho
    else:
        for i in range(1, imax - 1):
            for j in range(1, jmax - 1):
                dpdx = (u[i + 1, j, 0] - u[i - 1, j, 0]) * inv2dx
                dpdy = (u[i, j + 1, 0] - u[i, j - 1, 0]) * inv2dy
                u[i, j, 1] -= dt[i, j] * dpdx / rho
                u[i, j, 2] -= dt[i, j] * dpdy / rho
    return u