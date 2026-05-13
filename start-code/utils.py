import numpy as np

################################## Source ##################################

def compute_source_terms(s, imax, jmax, imms, xmax, xmin, ymax, ymin,
                         rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy):
    """
    Computes source terms for mass, x-momentum, and y-momentum.
    Assumes that s array and all global variables are already defined.
    """
    imax = s.shape[0]
    jmax = s.shape[1]

    for j in range(1, jmax - 1):
        for i in range(1, imax - 1):
            x = (xmax - xmin) * (i) / (imax - 1)
            y = (ymax - ymin) * (j) / (jmax - 1)
            s[i, j, 0] = imms * srcmms_mass(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy)
            s[i, j, 1] = imms * srcmms_xmtm(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy)
            s[i, j, 2] = imms * srcmms_ymtm(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy)
    return s

def srcmms_mass(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy):
    dudx = phix[1] * apx[1] * np.pi / rlength * np.cos(apx[1] * np.pi * x / rlength) + \
           phixy[1] * apxy[1] * np.pi * y / rlength**2 * np.cos(apxy[1] * np.pi * x * y / rlength**2)
    dvdy = -phiy[2] * apy[2] * np.pi / rlength * np.sin(apy[2] * np.pi * y / rlength) - \
           phixy[2] * apxy[2] * np.pi * x / rlength**2 * np.sin(apxy[2] * np.pi * x * y / rlength**2)
    return rho * (dudx + dvdy)

def srcmms_xmtm(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy):
    # Compute the required temporary variables for the x-momentum source term
    termx = phix[1] * np.sin(apx[1] * np.pi * x / rlength)
    termy = phiy[1] * np.cos(apy[1] * np.pi * y / rlength)
    termxy = phixy[1] * np.sin(apxy[1] * np.pi * x * y / rlength**2)
    uvel = phi0[1] + termx + termy + termxy

    termx = phix[2] * np.cos(apx[2] * np.pi * x / rlength)
    termy = phiy[2] * np.cos(apy[2] * np.pi * y / rlength)
    termxy = phixy[2] * np.cos(apxy[2] * np.pi * x * y / rlength**2)
    vvel = phi0[2] + termx + termy + termxy

    dudx = phix[1] * apx[1] * np.pi / rlength * np.cos(apx[1] * np.pi * x / rlength) + \
           phixy[1] * apxy[1] * np.pi * y / rlength**2 * np.cos(apxy[1] * np.pi * x * y / rlength**2)
    dudy = -phiy[1] * apy[1] * np.pi / rlength * np.sin(apy[1] * np.pi * y / rlength) + \
           phixy[1] * apxy[1] * np.pi * x / rlength**2 * np.cos(apxy[1] * np.pi * x * y / rlength**2)
    dpdx = -phix[0] * apx[0] * np.pi / rlength * np.sin(apx[0] * np.pi * x / rlength) + \
           phixy[0] * apxy[0] * np.pi * y / rlength**2 * np.cos(apxy[0] * np.pi * x * y / rlength**2)
    d2udx2 = -phix[1] * (apx[1] * np.pi / rlength)**2 * np.sin(apx[1] * np.pi * x / rlength) - \
             phixy[1] * (apxy[1] * np.pi * y / rlength**2)**2 * np.sin(apxy[1] * np.pi * x * y / rlength**2)
    d2udy2 = -phiy[1] * (apy[1] * np.pi / rlength)**2 * np.cos(apy[1] * np.pi * y / rlength) - \
             phixy[1] * (apxy[1] * np.pi * x / rlength**2)**2 * np.sin(apxy[1] * np.pi * x * y / rlength**2)

    return rho * uvel * dudx + rho * vvel * dudy + dpdx - rmu * (d2udx2 + d2udy2)

def srcmms_ymtm(x, y, rho, rmu, rlength, phi0, phix, phiy, phixy, apx, apy, apxy):
    termx = phix[1] * np.sin(apx[1] * np.pi * x / rlength)
    termy = phiy[1] * np.cos(apy[1] * np.pi * y / rlength)
    termxy = phixy[1] * np.sin(apxy[1] * np.pi * x * y / rlength**2)
    uvel = phi0[1] + termx + termy + termxy

    termx = phix[2] * np.cos(apx[2] * np.pi * x / rlength)
    termy = phiy[2] * np.cos(apy[2] * np.pi * y / rlength)
    termxy = phixy[2] * np.cos(apxy[2] * np.pi * x * y / rlength**2)
    vvel = phi0[2] + termx + termy + termxy

    dvdx = -phix[2] * apx[2] * np.pi / rlength * np.sin(apx[2] * np.pi * x / rlength) - \
           phixy[2] * apxy[2] * np.pi * y / rlength**2 * np.sin(apxy[2] * np.pi * x * y / rlength**2)
    dvdy = -phiy[2] * apy[2] * np.pi / rlength * np.sin(apy[2] * np.pi * y / rlength) - \
           phixy[2] * apxy[2] * np.pi * x / rlength**2 * np.sin(apxy[2] * np.pi * x * y / rlength**2)
    dpdy = phiy[0] * apy[0] * np.pi / rlength * np.cos(apy[0] * np.pi * y / rlength) + \
           phixy[0] * apxy[0] * np.pi * x / rlength**2 * np.cos(apxy[0] * np.pi * x * y / rlength**2)
    d2vdx2 = -phix[2] * (apx[2] * np.pi / rlength)**2 * np.cos(apx[2] * np.pi * x / rlength) - \
             phixy[2] * (apxy[2] * np.pi * y / rlength**2)**2 * np.cos(apxy[2] * np.pi * x * y / rlength**2)
    d2vdy2 = -phiy[2] * (apy[2] * np.pi / rlength)**2 * np.cos(apy[2] * np.pi * y / rlength) - \
             phixy[2] * (apxy[2] * np.pi * x / rlength**2)**2 * np.cos(apxy[2] * np.pi * x * y / rlength**2)

    return rho * uvel * dvdx + rho * vvel * dvdy + dpdy - rmu * (d2vdx2 + d2vdy2)

def umms(x, y, k, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy):
    """
    Returns the MMS exact solution.
    rpi, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy.
    Inputs: x, y, k
    Returns: ummstmp
    """
    # global rpi, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy

    # Calculate arguments and terms
    argx = apx[k] * np.pi * x / rlength
    argy = apy[k] * np.pi * y / rlength
    argxy = apxy[k] * np.pi * x * y / (rlength ** 2)
    
    termx = phix[k] * (fsinx[k] * np.sin(argx) + (1 - fsinx[k]) * np.cos(argx))
    termy = phiy[k] * (fsiny[k] * np.sin(argy) + (1 - fsiny[k]) * np.cos(argy))
    termxy = phixy[k] * (fsinxy[k] * np.sin(argxy) + (1 - fsinxy[k]) * np.cos(argxy))

    # Compute the MMS exact solution
    ummstmp = phi0[k] + termx + termy + termxy
    
    return ummstmp

def pressure_rescaling(u, imms, xmax, xmin, ymax, ymin, pinf, imax, jmax,
                       k, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy):
    """
    Rescales the pressure field based on a reference point.

    Parameters:
    - u: 3D numpy array representing the flow field [pressure, u-velocity, v-velocity]
    - imms: Integer indicating if a manufactured solution is used (1 if MMS is used)
    - xmax: Maximum x-coordinate
    - xmin: Minimum x-coordinate
    - ymax: Maximum y-coordinate
    - ymin: Minimum y-coordinate
    - pinf: Reference pressure
    - imax: Number of points in x direction
    - jmax: Number of points in y direction
    - umms: Callable function umms(x, y, 1) if MMS is used (optional)

    Modifies:
    - u: Updates the pressure values directly in the array.
    """
    # Set reference pressure point to the center of the cavity
    iref = (imax - 1) // 2
    jref = (jmax - 1) // 2

    # Calculate the reference pressure difference (deltap)
    if imms == 1:
        x = (xmax - xmin) * (iref) / (imax - 1)
        y = (ymax - ymin) * (jref) / (jmax - 1)
        deltap = u[iref, jref, 0] - umms(x, y, k, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy)  # Constant for MMS
    else:
        deltap = u[iref, jref, 0] - pinf  # Reference pressure

    # Rescale all pressure values in the flow field
    u[:, :, 0] -= deltap

    return u

import numpy as np


def output_file_headers(imms):
    """
    Sets up output files (history and solution) and writes headers.
    Uses global variables imms, fp1, and fp2.
    """
    
    # Open the 'history.dat' file in write mode and write headers
    fp1 = open('./history.txt', 'w')
    fp1.write('TITLE = "Cavity Iterative Residual History"\n')
    fp1.write('variables="Iteration" "Time(s)" "Res1" "Res2" "Res3"\n')

    # Open the 'cavity.dat' file in write mode and write headers
    fp2 = open('./cavity.txt', 'w')
    fp2.write('TITLE = "Cavity Field Data"\n')
    
    if imms == 1:
        # MMS case
        fp2.write('variables="x(m)" "y(m)" "p(N/m^2)" "u(m/s)" "v(m/s)" ')
        fp2.write('"p-exact" "u-exact" "v-exact" "DE-p" "DE-u" "DE-v"\n')
    elif imms == 0:
        # Standard case without MMS
        fp2.write('variables="x(m)" "y(m)" "p(N/m^2)" "u(m/s)" "v(m/s)"\n')
    else:
        print('ERROR! imms must equal 0 or 1!!!')
        return
    
    # Header for screen output
    print('Iter.    Time (s)   dt (s)      Continuity    x-Momentum    y-Momentum')

    return fp1, fp2


def write_output(n, resinit, rtime, xmax, xmin, ymax, ymin, imms, u, ummsArray, fp2):
    """
    Writes output and restart files.
    Uses global variables: imax, jmax, xmax, xmin, ymax, ymin, imms, u, ummsArray, fp2
    """

    imax = u.shape[0]
    jmax = u.shape[1]

    # Field output
    fp2.write(f'zone T="n={n}"\n')
    fp2.write(f'I= {imax} J= {jmax}\n')
    fp2.write('DATAPACKING=POINT\n')

    if imms == 1:
        for j in range(jmax):
            for i in range(imax):
                x = (xmax - xmin) * i / (imax - 1)
                y = (ymax - ymin) * j / (jmax - 1)
                fp2.write(f'{x:e} {y:e} {u[i, j, 0]:e} {u[i, j, 1]:e} {u[i, j, 2]:e} '
                          f'{ummsArray[i, j, 0]:e} {ummsArray[i, j, 1]:e} {ummsArray[i, j, 2]:e} '
                          f'{(u[i, j, 0] - ummsArray[i, j, 0]):e} {(u[i, j, 1] - ummsArray[i, j, 1]):e} '
                          f'{(u[i, j, 2] - ummsArray[i, j, 2]):e}\n')
    elif imms == 0:
        for j in range(jmax):
            for i in range(imax):
                x = (xmax - xmin) * i / (imax - 1)
                y = (ymax - ymin) * j / (jmax - 1)
                fp2.write(f'{x:e} {y:e} {u[i, j, 0]:e} {u[i, j, 1]:e} {u[i, j, 2]:e}\n')
    else:
        print("ERROR: imms must equal 0 or 1!")
        return

    # Restart file: overwrites every 'iterout' iteration
    with open('./restart.out', 'w') as fp3:
        fp3.write(f'{n} {rtime:e}\n')
        fp3.write(f'{resinit[0]:e} {resinit[1]:e} {resinit[2]:e}\n')
        for j in range(jmax):
            for i in range(imax):
                x = (xmax - xmin) * i / (imax - 1)
                y = (ymax - ymin) * j / (jmax - 1)
                fp3.write(f'{x:e} {y:e} {u[i, j, 0]:e} {u[i, j, 1]:e} {u[i, j, 2]:e}\n')


def initialize(ninit, rtime, resinit, irstr, neq, uinf, pinf, xmax, xmin, ymax, ymin, u, s, ummsArray,
               rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy):
    """
    Sets initial conditions for the cavity simulation.
    Uses global variables: irstr, neq, uinf, pinf, xmax, xmin, ymax, ymin
    Modifies: ninit, rtime, resinit, u, s
    """
    imax = u.shape[0]
    jmax = u.shape[1]
    
    if irstr == 0:
        # Starting from scratch
        ninit = 0
        rtime = 0
        resinit = np.ones(neq)
        
        # Initialize `u` and `s` arrays
        for i in range(imax):
            for j in range(jmax):
                u[i, j, 0] = pinf
                u[i, j, 1] = 0
                u[i, j, 2] = 0
                s[i, j, 0] = 0
                s[i, j, 1] = 0
                s[i, j, 2] = 0
            # Set lid velocity for the top boundary
            u[i, jmax - 1, 1] = uinf

    elif irstr == 1:
        # Restarting from a previous run
        try:
            with open('./restart.in', 'r') as fp4:
                # Read iteration number, time, and initial residuals
                ninit = int(fp4.readline().strip())
                rtime = float(fp4.readline().strip())
                resinit = np.array([float(x) for x in fp4.readline().split()])

                # Read values for `u` from file
                for j in range(jmax):
                    for i in range(imax):
                        read_vals = [float(x) for x in fp4.readline().split()]
                        x, y = read_vals[0], read_vals[1]
                        u[i, j, 0], u[i, j, 1], u[i, j, 2] = read_vals[2], read_vals[3], read_vals[4]

                ninit += 1
                print(f"Restarting at iteration {ninit}")

        except FileNotFoundError:
            print("Error opening restart file. Stopping.")
            return

    else:
        print("ERROR: irstr must equal 0 or 1!")
        return

    # Initialize the `ummsArray` with values from the `umms` function
    for j in range(jmax):
        for i in range(imax):
            for k in range(neq):
                x = (xmax - xmin) * (i) / (imax - 1)
                y = (ymax - ymin) * (j) / (jmax - 1)
                ummsArray[i, j, k] = umms(x, y, k, rlength, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy)
    
    return ninit, rtime, resinit, ummsArray

# rL1norm, rL2norm, and rLinfnorm now contain the computed norms
