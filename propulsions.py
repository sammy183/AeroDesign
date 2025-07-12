# -*- coding: utf-8 -*-
"""
Propulsions Module (no more globals, now integrated into the object oriented common.py structure)

Note: edited down for conciseness
"""

import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt
import time
from gekko import GEKKO
import multiprocessing
from functools import partial
import itertools
import numba
from numba import njit, jit

#ignoring the np RankWarning lol
# np.seterr(all='ignore')

lbfN = 4.44822
ftm = 0.3048
MPH_TO_MPS = 0.44704  # Conversion factor: 1 mph to m/s
def parse_propeller_data(prop_name):
    """
    prop_name in the form: 16x10E, 18x12E, 12x12, etc (no PER3_ and no .dat to make it easier for new users)
    Parses the provided PER3_16x10E.dat content to extract RPM, V (m/s), Thrust (N), Torque (N-m).
    Stores in PROP_DATA as {rpm: {'V': np.array, 'Thrust': np.array, 'Torque': np.array}}
    """    
    PROP_DATA = {}

    with open(f'Databases/PropDatabase/PER3_{prop_name}.dat', 'r') as f:
        data_content = f.read()

    current_rpm = None
    in_table = False
    table_lines = []
    
    for line in data_content.splitlines():
        line = line.strip()
        if line.startswith("PROP RPM ="):
            # Extract RPM
            current_rpm = int(line.split("=")[-1].strip())
            in_table = False
            table_lines = []
        elif line.startswith("V") and "J" in line and current_rpm is not None:
            # Start of table headers
            in_table = True
        elif in_table and line and not line.startswith("(") and len(line.split()) >= 10:
            # Parse data rows (ensure it's a data line with enough columns)
            parts = line.split()
            try:
                v_mph = float(parts[0])  # V in mph
                torque_nm = float(parts[9])  # Torque (N-m)
                thrust_n = float(parts[10])  # Thrust (N)
                v_mps = v_mph * MPH_TO_MPS  # Convert to m/s
                table_lines.append((v_mps, thrust_n, torque_nm))
            except (ValueError, IndexError):
                continue  # Skip malformed lines
        elif in_table and (line == "" or "PROP RPM" in line):
            # End of table for this RPM, store if data exists
            if current_rpm and table_lines:
                v_list, thrust_list, torque_list = zip(*sorted(table_lines))  # Sort by V for interp1d
                PROP_DATA[current_rpm] = {
                    'V': np.array(v_list),
                    'Thrust': np.array(thrust_list),
                    'Torque': np.array(torque_list)
                }
            in_table = False
    
    # Sort RPM keys for efficient lookup
    PROP_DATA['rpm_list'] = sorted(PROP_DATA.keys())
    
    # array based datastructure where each index corresponds to rpm_values[i] (or i+1*1000 RPM)
    # and in each index there is [[V values], [Thrust values], [Torque values]] at the indices, 0, 1, 2
    numba_prop_data = []
    for RPM in PROP_DATA['rpm_list']:
        datasection = np.array([PROP_DATA[RPM]['V'], 
                                PROP_DATA[RPM]['Thrust'], 
                                PROP_DATA[RPM]['Torque']])
        numba_prop_data.append(datasection)
        
    return(PROP_DATA, numba_prop_data)

def initialize_RPM_polynomials(PROP_DATA):
    """
    returns: rpm_values, thrust_polys, torque_polys, V_DOMAINS
    
    Creates polynomial approximations for thrust and torque that are compatible with GEKKO in the form of Thrust(V) for a fixed RPM
    Uses piecewise polynomials for different RPM ranges.
    """
    
    # Extract data for polynomial fitting
    rpm_values = sorted([rpm for rpm in PROP_DATA.keys() if isinstance(rpm, int)])
    
    # Create coefficient matrices for polynomial approximation
    # We'll use separate polynomials for different velocity ranges
    thrust_polys = {}
    torque_polys = {}
    
    V_DOMAINS = []
    for rpm in rpm_values:
        data = PROP_DATA[rpm]
        V_data = data['V']
        thrust_data = data['Thrust']
        torque_data = data['Torque']
                
        # Fit polynomials (degree 3-4 should be sufficient for most cases)
        thrust_poly = Polynomial.fit(V_data, thrust_data, deg=4)
        thrust_polys[rpm] = thrust_poly
        
        # Fit polynomial to torque data  
        torque_poly = Polynomial.fit(V_data, torque_data, deg=4)
        torque_polys[rpm] = torque_poly
    
        V_DOMAINS.append(torque_poly.domain[1])
    
    V_DOMAINS = np.array(V_DOMAINS)
    return rpm_values, thrust_polys, torque_polys, V_DOMAINS

#%% functions intended for PointDesign objects:
# Thrust and Torque functions from prop data!
def Thrust_scalar(self, RPM, V):
    '''Self is the PointDesign object!'''
    if not self.PROP_DATA or RPM < min(self.PROP_DATA['rpm_list']) or RPM > max(self.PROP_DATA['rpm_list']) or V < 0:
        # print('THIS IS THE PROBLEM!!!')
        return 0.0

    rpm_list = self.PROP_DATA['rpm_list']
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]

    thrust_values = []
    for rpm in closest_rpms:
        data = self.PROP_DATA[rpm]
        if len(data['V']) < 2:
            thrust_values.append(0.0)
            continue
        interpolator = interp1d(data['V'], data['Thrust'], kind='linear', bounds_error=False, fill_value=0.0)
        thrust_values.append(interpolator(V))

    if len(closest_rpms) == 1:
        return thrust_values[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * thrust_values[0] + weight * thrust_values[1]

# Vectorized version using np.vectorize
Thrust = np.vectorize(Thrust_scalar)

def Torque_scalar(self, RPM, V):        
    if not self.PROP_DATA or RPM < min(self.PROP_DATA['rpm_list']) or RPM > max(self.PROP_DATA['rpm_list']) or V < 0:
        return 0.0

    rpm_list = self.PROP_DATA['rpm_list']
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]

    torque_values = []
    for rpm in closest_rpms:
        data = self.PROP_DATA[rpm]
        if len(data['V']) < 2:
            torque_values.append(0.0)
            continue
        interpolator = interp1d(data['V'], data['Torque'], kind='linear', bounds_error=False, fill_value=0.0)
        torque_values.append(interpolator(V))

    if len(closest_rpms) == 1:
        return torque_values[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * torque_values[0] + weight * torque_values[1]

# Vectorized version using np.vectorize
Torque = np.vectorize(Torque_scalar)

#%% Scipy formulation for solving RPM, Q, T, etc at a specified velocity and runtime (t)
def ModelCalcs(self, velocity, t):
    '''
    Returns: T, P, Itot, RPM
    
    This formulation encounters problems when Itot could equal I0 (bc then Q just goes to 0!)
    Have a switching mechanism if Q = 0!
    
    To add: better checks for errors, better root finding mechanism (perhaps with constraints?)
    ability to switch methods if code is inaccurate
    
    Swapped Torque for power P!
    
    NOTE: I0 IS NOT SCALED PROPERLY IN THIS FORMULATION!!!
    '''

    methodtesting = 'lm'
    
    Itotguess = 100
    def equation(Itot):
        SOC = (self.CB*3.6 - Itot*t)/(self.CB*3.6) #maybe this shouldn't be Itot??
        Vsoc = -1.031*np.e**(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
        # Rint = 0.3208*np.e**(-24.14*SOC) + 0.04669
        Vin = (Vsoc - Itot*self.Rint)*self.ns #voltage out of the battery
        
        Imot = Itot/self.nmot
        RPM = (self.KV)*(Vin - Imot*self.Rm) # Voltage is the same between parallel, but current is split eventy between motors!
        Q2 = Torque(self, RPM, velocity) #total torque
        
        #**** should change I0 here to scale with Vin! (it's giving weird data for some reason)
        Q1 = (Imot - self.I0)*(1/(self.KV*2*np.pi/60)) #only a fraction of total current goes to each motor!
        residual = np.abs(Q1 - Q2)
        return(residual)
        
    res = opt.root(equation, Itotguess, method=methodtesting) # should consider adding some constraints: Itot > 0, SOC > 0, etc and using a try statement to deal with conditions where SOC is too small (aka infeasible)
    Itot = res.x[0]
    # SOC = (self.CB*3.6 - Itot*t)/(self.CB*3.6)
    # Vsoc = -1.031*np.e**(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
    # Vin = (Vsoc - Itot*self.Rint)*self.ns #voltage out of the battery
    
    if Itot <= self.I0*self.nmot:
        '''Switch to alternate formulation'''
        RPMguess = max(self.PROP_DATA['rpm_list'])-500
        # RPM guess matters a LOT more than you'd think. 
        # Best guess for the top of the plot is within 2000 RPM underneath the limit (CANNOT GO OVER THE LIMIT)
        # i.e. if RPMmax = 16000, best guesses are [14000, 15900]
        
        def equation(RPM):
            Q = self.nmot*Torque(self, RPM, velocity)
            Itot = Q*(self.KV*2*np.pi/60) + self.I0   #CAN'T SCALE I0 IN THIS FORMULATION!!!
            
            SOC = (self.CB*3.6 - Itot*t)/(self.CB*3.6) #maybe this shouldn't be Itot??
            Vsoc = -1.031*np.e**(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
            # Rint = 0.3208*np.e**(-24.14*SOC) + 0.04669
            
            Vin = (Vsoc - Itot*self.Rint)*self.ns
            
            Imot = Itot/self.nmot
            RPMnew = self.KV*(Vin - Imot*self.Rm)
            residual = np.abs(RPMnew - RPM)
            return residual
        
        res = opt.root(equation, RPMguess, method=methodtesting) #somehow RPMguess + 5000 fixed the Xenova + 18x12E issue???
        RPM = res.x[0]

        T = self.nmot*Thrust(self, RPM, velocity)
        Q = self.nmot*Torque(self, RPM, velocity)  #torque for all motors
        Itot = Q*(self.KV*2*np.pi/60) + self.I0   #I0 should depend on V, determine that
        Imot = Itot/self.nmot
        SOC = (self.CB*3.6 - Itot*t)/(self.CB*3.6) #maybe this shouldn't be Itot??
        Vsoc = -1.031*np.e**(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
        Vin = (Vsoc - Itot*self.Rint)*self.ns
        P = Imot*Vin
        
        if SOC < (1-self.ds) or Itot < 0 or T <= 0.0 or Q < 0:
            return(np.zeros(4))
        else:
            return(T, P, Itot, RPM)
        
    SOC = (self.CB*3.6 - Itot*t)/(self.CB*3.6)
    Vsoc = -1.031*np.e**(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
    Vin = (Vsoc - Itot*self.Rint)*self.ns #voltage out of the battery
    Imot = Itot/self.nmot
    RPM = (self.KV)*(Vin - Imot*self.Rm)
    T = self.nmot*Thrust(self, RPM, velocity)
    Q = (Itot - self.I0)*(1/(self.KV*2*np.pi/60))        
    P = Imot*Vin

    if SOC < (1-self.ds) or Q <= 0.0 or Itot < 0 or T <= 0.0:
        return(np.zeros(4))
    else:
        return(T, P, Itot, RPM)

#%% RPM poly fitting
def RPMPolyFits(self, V):
    #*********** convert the following into a vectorized operation with a mask (just make sure THRUST_POLYS can take multiple RPMs at once!)
    #or better yet ues the data direct from PROP_DATA so there's no error from the TORQUE_POLYs
    torque = []
    thrust = []
    rpm_use = []
    for rpm in self.RPM_VALUES:
        ThrustPolyUse = self.THRUST_POLYS[rpm]
        TorquePolyUse = self.TORQUE_POLYS[rpm]
        if TorquePolyUse.domain[0] > V or TorquePolyUse.domain[1] < V: #note: domain for torque_polys and thrust_polys is the same!!!
            continue
        else:
            torque.append(TorquePolyUse(V))
            thrust.append(ThrustPolyUse(V))
            rpm_use.append(rpm)

    rpm_use = np.array(rpm_use)
    torque = np.array(torque)
    thrust = np.array(thrust)
            
    # need to do study on the degrees!!!
    if len(rpm_use) >= 2:
        Torque_RPM_Poly = Polynomial.fit(rpm_use, torque, deg=4)   # so does this impact the original object or a copied version?
        Thrust_RPM_Poly = Polynomial.fit(rpm_use, thrust, deg=4)
    else:
        raise ValueError('V is outside data range')
    
    self.Thrust_RPM_Poly = Thrust_RPM_Poly
    self.Torque_RPM_Poly = Torque_RPM_Poly

#%% Primary data functions NOTE: probably want to rewrite all using self
def GekkoRuntime(self, V, verbose = False):
    '''
    Goal of this func is to find the maximum runtime t at a given cruise velocity 
    
    NOTE: I've learned that I0*(Vin/V_I0) is NECESSARY for some of the props to give good results
    NOTE: I've now learned that the I0*(Vin/V_I0 formulation is causing some problems???
    '''
    RPMPolyFits(self, V)
    lowerRPM, upperRPM = self.Torque_RPM_Poly.domain
    
    m = GEKKO(remote=False)
    RPM =       m.Var((upperRPM+lowerRPM)/2)

    t =         m.Var(100.0) # could potentially improve this via prechecking of some kind?
    Vin =       m.Var()
    
    Q =         m.Intermediate(self.nmot*self.Torque_RPM_Poly(RPM))  #torque for all motors # if I multiply by 1.35 the NTM propdrive 2024 setup matches better (but others match worse!)
    Itot =      m.Intermediate(Q*(self.KV*2*np.pi/60) + self.I0)                  #I0 should depend on V, determine that (let's just do I0 = I0*(Vactual/VofI0measurement))
    SOC =       m.Intermediate((self.CB*3.6 - Itot*t)/(self.CB*3.6)) 
    Vsoc =      -1.031*m.exp(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
    
    Imot =      Itot/self.nmot
    
    m.Equation([Itot > self.I0,
                SOC == (1-self.ds),
                Q > 0, 
                
                # Can add these restrictions to ensure physical values
                RPM < upperRPM,
                RPM > lowerRPM,
                
                # moved these eqns here bc it runs faster
                Vin == (Vsoc - Itot*self.Rint)*self.ns,
                RPM == self.KV*(Vin - Imot*self.Rm),
                ])

    m.options.SOLVER = 1   #1 = apopt, 3 = ipopt
    
    try:
        m.solve(disp=verbose)
    except:
        try:
            RPM = m.Var((upperRPM+lowerRPM)/2 - 2000)
            m.solve(disp=verbose)
        except:
            raise ValueError('Solution Failed')
    
    # Thrust was missing self.nmot!! Now this matches with the original formulation
    T_maxds = self.nmot*self.Thrust_RPM_Poly(RPM.value[0])
    
    return([t.value[0], T_maxds])


def PlotRuntimes(self, n, verbose = False, showthrust = True):
    '''
    TO BE OPTIMIZED LATER WITH SCIPY CONFIG OR MULTIPROCESSING OR NUMBA!!
    
    Also get rid of 
    
    n is number of velocities considered (essentially more --> greater accuracy, less --> faster execution)
    '''
    Vs = np.linspace(0, self.V_DOMAINS[-1], n)
    runtimes = []
    V_plot = []
    T_maxds = []
    
    print('\nGetting runtimes....')
    
    for velocity in tqdm(Vs):
        # right now this uses GekkoRuntime which uses a fixed velocity and the TorqueRPM(V) polys. 
        # However, this seems inefficient and maybe I should pivot to scipy minimize
        try:
            data = GekkoRuntime(self, velocity, verbose = verbose)
        except ValueError as e:
            if verbose:
                print(e)
            continue
        
        runtimes.append(data[0])
        V_plot.append(velocity)
        T_maxds.append(data[1])
    
    V_plot = np.array(V_plot)
    runtimes = np.array(runtimes)
    T_maxds = np.array(T_maxds)
    
    # plotting
    fig, ax1 = plt.subplots(figsize=(6,4), dpi = 1000)

    ax1.plot(V_plot/ftm, runtimes, color='k')#, marker='o')
    ax1.set_ylabel('Runtime (s)')

    if showthrust:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Thrust (lbf) at {(1- self.ds)*100:.0f}% battery', color='#cc0000')
        D = 0.5*self.rho*(V_plot**2)*self.Sw*self.CD
        indx = np.argmin(np.abs(T_maxds - D))
        ax2.plot(V_plot/ftm, T_maxds/lbfN, '#cc0000', label = 'Thrust')
        ax2.plot(V_plot/ftm, D/lbfN, 'blue', label = 'Drag')
        ax2.plot(V_plot[indx]/ftm, T_maxds[indx]/lbfN, 'o', 
                color='k', markersize=7, zorder = 5, 
                label=f'Cruise = {V_plot[indx]/ftm:.2f} fps,\nat T = D = {T_maxds[indx]/lbfN:.2f} lbs')
        ax2.legend(loc = 'center left')
        ax2.minorticks_on()
    
    ax1.grid()
    ax1.minorticks_on()
    ax1.set_xlabel('Cruise Velocity (fps)')
    
    if self.nmot > 1:
        s = 's'
    else:
        s = ''
    plt.title(f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; \n'
              f'{self.nmot} APC {self.prop_name} propeller{s}; '
              f'\n{self.ds*100:.0f}% discharge of an {self.ns}S {self.CB} mAh LiPo')
    plt.show()
    
    
# misc based on gekko runtime

def getGekkoRuntimeMin(self, Vmax):
    '''benchmarks for opt.minimize algorithms on this problem:
        nelder-mead,    time: 4.349444500010577 s
        powell,         time: 4.11858470000152 s
        cg,             time: 1.370953899997403 s
        bfgs,           time: 1.1993925999995554 s
        l-bfgs-b,       time: 0.7182616000063717 s
        tnc,            time: 1.2165635999990627 s
        cobyla,         time: 3.0351095000078203 s
        cobyqa,         time: 2.028681299998425 s
        slsqp,          time: 6.496950100001413 s
        trust-constr,   time: 1.422567400004482 s
        
        best is l-bfgs-b
        
        PROBLEM SOLVED: use a smaller Vmax initial value to prompt better convergence!'''
    self.PROP_DATA, self.NUMBA_PROP_DATA= parse_propeller_data(self.prop_name)
    initialize_RPM_polynomials(self.PROP_DATA)
    
    def f(V):
        '''I think something is fucking up wrt to the object oriented implementation'''
        try:
            RPMPolyFits(self, V[0]) # need to do this to define the Torque_RPM_Polys for GekkoRuntime
            t, _ = GekkoRuntime(self, V[0], verbose=False)
            return(t)
        except:
            return(1e10)
    # use BFGS to optimize the V part
    
    res = opt.minimize(f, Vmax/4, method = 'l-bfgs-b')
    tmin = res.fun
    V_tmin = res.x[0]
    
    return(tmin, V_tmin)

#%% Coefficient matrix based gekko
def getCoefs(V, RPM, data, degree = 3):
    degree_x = degree
    degree_y = degree

    # Generate grid points for evaluation
    x = V
    y = RPM
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten the grid points and the corresponding values
    X_flat = x_grid.flatten()
    Y_flat = y_grid.flatten()
    Q_flat = data.flatten()

    # Generate Vandermonde matrix
    V = np.polynomial.polynomial.polyvander2d(X_flat, Y_flat, [degree_x, degree_y])
    
    # Calculate polynomial coefficients using least squares
    coeffs, _, _, _ = np.linalg.lstsq(V, Q_flat, rcond=None)

    # Reshape the coefficients into a 2D matrix
    coeffs_matrix = np.reshape(coeffs, (degree_y + 1, degree_x + 1))
    return(coeffs_matrix)

def getValue(V, RPM, coef_matrix):
    '''Mostly shorthand, could take out if it slows down the code'''
    return(np.polynomial.polynomial.polyval2d(V, RPM, coef_matrix))

def ThrustDataCollect(self, indx):
    '''Generates a coefficient matrix for Thrust in N for the selected RPM, V square of data 
    (given by indx, which denotes the max RPM value to take data from)'''
    Vlim = self.PROP_DATA[self.RPM_VALUES[indx]]['V'].max()    
    
    # alter the number of elements here to change the fit, 
    # lower means worse fit but slightly faster
    Vmain = np.linspace(0, Vlim, 1000) 
    
    # Ts is the Thrust data generated via the THRUST_POLYS global variable at known RPM values
    Ts = np.array(self.THRUST_POLYS[self.RPM_VALUES[indx]](Vmain))
    Ts[Ts < 0.0] = 0.0 # ensuring all = 0!
    
    # collect Thrust data
    for rpm in self.RPM_VALUES[indx+1:]:        
        Tuse = self.THRUST_POLYS[rpm](Vmain)
        Tuse[Tuse < 0.0] = 0.0
        Ts = np.vstack((Ts, Tuse))
    
    # generate coefficient matrix
    newrpm = np.array(self.RPM_VALUES[indx:])
    coeffmatrix = getCoefs(Vmain, newrpm, Ts)
    
    return(coeffmatrix)

def TorqueDataCollect(self, indx):
    '''Same as ThrustDataCollect but for Torque in N*m'''
    Vlim = self.PROP_DATA[self.RPM_VALUES[indx]]['V'].max()    
    
    Vmain = np.linspace(0, Vlim, 1000)
    Vs = np.array(Vmain)
    Ts = np.array(self.TORQUE_POLYS[self.RPM_VALUES[indx]](Vmain))
    Ts[Ts < 0.0] = 0.0
    
    for rpm in self.RPM_VALUES[indx+1:]:        
        Tuse = self.TORQUE_POLYS[rpm](Vmain)
        Tuse[Tuse < 0.0] = 0.0
        Vs = np.vstack((Vs, Vmain))
        Ts = np.vstack((Ts, Tuse))
                
    newrpm = np.array(self.RPM_VALUES[indx:])    
    coeffmatrix = getCoefs(Vmain, newrpm, Ts)
    
    return(coeffmatrix)

def GekkoVmax(self, index, discharge, Tlimit = True, verbose = False, t_in = 0.0):
    ''' ds is discharge (I needed a new name bc of the global variable in use as well; def streamline this later)
    
        Provides the maximum speed attainable at the selected discharge value
        for the given setup in one of the RPM-V rectangle of data
        
        Iterate through all RPM-V squares of data (usually 15 max per prop) and get
        the actual maximum speed!
        
        Each rectangle of data will have a greater Vlim (V_DOMAINS[index]) and a greater RPMmin
        
        Returns Vmax (m/s), runtime (s)
        
        NOTE:   could probably change the RPM-RPMnew residual formulation to just RPM bc it solves alltogether
                check accuracy in future
        NOTE:   the torque data wasn't able to be fit to my standards; be extra careful about results where the Vmax is very close to the
                data rectangle boundary
        '''
    
    TorqueCoeffs = TorqueDataCollect(self, index)
    ThrustCoeffs = ThrustDataCollect(self, index)

    m = GEKKO(remote=False)
    RPM =       m.Var((self.RPM_VALUES[index+1]+self.RPM_VALUES[index])/2)
    t =         m.Var(t_in)
    Vin =       m.Var()
    V =         m.Var(self.V_DOMAINS[index-1])
    
    #
    Q = m.Intermediate(self.nmot*getValue(V, RPM, TorqueCoeffs))    
    Itot =      m.Intermediate(Q*(self.KV*2*np.pi/60) + self.I0)                  #I0 should depend on V, determine that
    SOC =       m.Intermediate((self.CB*3.6 - Itot*t)/(self.CB*3.6)) 
    Vsoc =      -1.031*m.exp(-35*SOC) + 3.685 + 0.2156*SOC - 0.1178*SOC**2 + 0.3201*(SOC**3)
    
    Imot =      Itot/self.nmot
    
    T =         m.Intermediate(self.nmot*getValue(V, RPM, ThrustCoeffs))
    D =         m.Intermediate(0.5*self.rho*(V**2)*self.CD*self.Sw)
    
    if Tlimit:
        m.Equation([T > D])

    m.Equation([Itot > self.I0, #took away new I0 formulation for now! *(Vin/self.V_I0)
                SOC == (1.0-discharge),
                Q > 0, 
                
                # moved these eqns here bc it runs faster
                Vin == (Vsoc - Itot*self.Rint)*self.ns,
                RPM == self.KV*(Vin - Imot*self.Rm),

                # ensures the data doesn't exceed the range of the coef matrix
                RPM < self.RPM_VALUES[-1],
                RPM > self.RPM_VALUES[index],
                V <= self.V_DOMAINS[index],
                V > 0,
                                
                T > 0.0,
                ])    
    m.Maximize(V)
    m.options.SOLVER = 1   #1 = apopt, 3 = ipopt
    
    try:
        m.solve(disp=False)
    except:
        raise ValueError('Solution Failed')
    
    if verbose:
        print('')
        print(f"{'Itot:':10} {Itot.value}")
        print(f"{'SOC:':10} {SOC.value}")
        print(f"{'Vin:':10} {Vin.value}")
        print(f"{'Runtime':10} {t.value}")    
        print(f'{'Velocity':10} {V.value}')
        
        # checking the results with the more directly interpolated data!
        print(f'{"Thrust (T) from gekko model":30} {T.value[0]}')
        print(f'{"Thrust (T) from data":30} {self.nmot*Thrust(self, RPM.value[0], V.value[0])}') # uh oh that's significantly different.... (now it's better)
        print(f"{'Torque (Q) from gekko model':30} {Q.value[0]}")
        print(f'{"Torque (Q) from data":30} {self.nmot*Torque(self, RPM.value[0], V.value[0])}') # uh oh that's significantly different.... (now it's better)
    
    return(V.value[0], t.value[0])


def GekkoVmaxForData(self, discharge, verbose = False, Tlimit = False):
    Vmax = 0 
    runtime = 0
    
    for i in range(len(self.RPM_VALUES)):
        try:
            Vspec, tspec = GekkoVmax(self, i, discharge, Tlimit = Tlimit, verbose = False)
            if Vspec > Vmax:
                Vmax = Vspec
                runtime = tspec
        except:
            continue
        
    return(Vmax, runtime)

def VmaxLean(self, discharge, t_in = 0.0, Tlimit=False):
    Vmax = 0 
    for i in range(len(self.RPM_VALUES)):
        try:
            Vspec, tspec = GekkoVmax(self, i, discharge, t_in = t_in, 
                                     Tlimit = Tlimit, verbose = False)
            # print(i, self.RPM_VALUES[i], 'new', Vspec, tspec)
            if Vspec > Vmax:
                Vmax = Vspec
        except:
            if i > 0:
                break
    return(Vmax)

def GetPointDesignData(self, m):
    '''Now updates Vinf and t to get better axes'''    
    
    print('\nFinding maximum velocities (please wait ~10s)')
    
    start = time.perf_counter()
    VmaxTDend, t_Vmax = GekkoVmaxForData(self, self.ds, Tlimit=True)
    VmaxReal = VmaxLean(self, 0.0, Tlimit=True)
    Vmax = self.PROP_DATA[self.RPM_VALUES[-1]]['V'].max()
    
    Vnew = np.linspace(0.0, VmaxReal + 20*ftm, m)
    tmin, V_tmin = getGekkoRuntimeMin(self, Vmax) # I'm not understanding how this tmin is accurate but the Vmax isn't???
    tnew = np.linspace(0.0, t_Vmax + 30, m)
    end = time.perf_counter()
    print(f'Maximum velocity = {VmaxReal/ftm:.2f} fps at runtime = 0.0s')
    print(f'Minimum runtime = {tmin:.2f}s at velocity = {V_tmin/ftm:.2f} fps')
    print(f'Maximum runtime = {t_Vmax:.2f}s at velocity = {VmaxTDend/ftm:.2f} fps')
    print(f'Time Taken: {end-start:.2f}s\n')

    print('Starting data collection')
    rpm_list = np.array(self.RPM_VALUES)

    T = np.zeros((m, m))
    Itot = np.zeros((m, m))
    P = np.zeros((m, m))
    RPM = np.zeros((m, m))
    
    for i, velocity in enumerate(tqdm(Vnew)):
        if velocity > VmaxReal + 30*ftm:
            break
        else:
            for j, tspec in enumerate(tnew):
                # NOT RETURNING CORRECT VALUES FOR SOME CASES!!!
                Tspec, Pspec, Itotspec, RPMspec = ModelCalcsNumba(velocity, tspec, rpm_list, 
                                                                      self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                                                      self.KV, self.Rm, self.nmot, self.I0, self.ds)
                if Tspec == 0:
                    # switch to slower but more reliablel Scipy formulation
                    Tspec, Pspec, Itotspec, RPMspec = ModelCalcs(self, velocity, tspec)
                
                if Tspec == 0:
                    if tspec > tmin:
                        break
                    else:
                        continue
                else:
                    T[i, j] = Tspec
                    P[i, j] = Pspec
                    Itot[i, j] = Itotspec
                    RPM[i, j] = RPMspec
    Vnew = Vnew/ftm
    T = T/lbfN
    
    self.PointDesignData = [Vnew, tnew, T, P, Itot, RPM]
    return(Vnew, tnew, T, P, Itot, RPM)

def plotMosaic(self, grade = 15, colormap='viridis',
               
               Tlimit = True, Ilimit = np.inf, RPMlimit = np.inf):
    '''Vinf in ft/s, t in s, T in lbf, P in W, Itot in A, RPM in RPM'''
    
    Vinf, t, T, P, Itot, RPM = self.PointDesignData
    
    fig = plt.figure(layout="constrained", figsize=(15,10))
    ax_dict = fig.subplot_mosaic("AB;CD")
    
    for letter in ['A', 'B', 'C', 'D']:
        ax_dict[letter].grid()
        ax_dict[letter].minorticks_on()
        if letter in ['C', 'D']:
            ax_dict[letter].set_xlabel('Battery Runtime (s)')
            
    X, Y = np.meshgrid(t, Vinf) 
        
    # Thrust plot (top left)
    upper = T.max()
    Talt = T[T > 0.0]
    Tmin = Talt.min()
    Tplot = ax_dict['A'].contourf(X, Y, T, cmap = colormap, levels = np.linspace(Tmin, upper, grade))
    fig.colorbar(Tplot, ticks = np.linspace(Tmin, upper, 11), pad = 0.025, shrink = 1.0, spacing = 'uniform', label= 'Thrust (lbf)', ax=ax_dict['A'])
    ax_dict['A'].set_ylabel('Freestream Velocity (fps)')
    
    inlinelabel = True
    # adding the limit lines if relevant 
    if RPMlimit < RPM.max():
        # line on main plot
        def RPMlimitFmt(value):
            return(f'{value:.0f} RPM')
        RPMlimitline = ax_dict['A'].contour(X, Y, RPM, colors = 'black', levels = np.array([RPMlimit]))
        ax_dict['A'].clabel(RPMlimitline, inline = inlinelabel, fmt=RPMlimitFmt)
        
        # line on secondary plot
        RPMline = ax_dict['C'].contour(X, Y, RPM, colors = 'black', levels = np.array([RPMlimit]))
        ax_dict['C'].clabel(RPMline, inline = inlinelabel, fmt = RPMlimitFmt)

    if self.Pmax < P.max():
        # line on main plot
        def PlimitFmt(value):
            return(f'{value:.0f}W')
        Plimitline = ax_dict['A'].contour(X, Y, P, colors = 'red', levels = np.array([self.Pmax]))
        ax_dict['A'].clabel(Plimitline, inline = inlinelabel, fmt = PlimitFmt)
        
        # line on secondary plot
        Pline = ax_dict['B'].contour(X, Y, P, colors = 'red', levels = np.array([self.Pmax]))
        ax_dict['B'].clabel(Pline,  inline = inlinelabel, fmt = PlimitFmt)
        
    
    if Ilimit < Itot.max():
        # line on main plot
        def IlimitFmt(value):
            return(f'{value:.0f}A')
        Ilimitline = ax_dict['A'].contour(X, Y, Itot, colors = 'darkviolet', levels = np.array([Ilimit]))
        ax_dict['A'].clabel(Ilimitline, inline = inlinelabel, fmt = IlimitFmt)
        
        Itotline = ax_dict['D'].contour(X, Y, Itot, colors = 'darkviolet', levels = np.array([Ilimit]))
        ax_dict['D'].clabel(Itotline, inline = inlinelabel, fmt = IlimitFmt)

    
    if Tlimit:
        D = 0.5*self.rho*self.CD*self.Sw*((Y*ftm)**2)/lbfN
        Dlimit = T-D
    
        def TlimitFmt(value):
            return "T = D"
        # adding on every plot for context
        for letter in ['A', 'B', 'C', 'D']:
            Tlimitline = ax_dict[letter].contour(X, Y, Dlimit, colors = 'orange', levels = np.array([0.000001]))
            ax_dict[letter].clabel(Tlimitline, inline = True, fmt = TlimitFmt)    
    
    # elec power plot (top right)
    Palt = P[P != 0]   #very good for removing all the violation keys, and finding the max, but flattens the array
    lower = Palt.min()        #finds the minimum Score discounting violation trips
    upper = P.max()
    Pplot = ax_dict['B'].contourf(X, Y, P, cmap = colormap, levels = np.linspace(lower, upper, grade))
    fig.colorbar(Pplot, ticks = np.linspace(lower, upper, 11), pad = 0.025, shrink = 1.0, spacing = 'uniform', label= 'Power per motor (W)', ax=ax_dict['B'])
    
    # RPM plot
    RPMalt = RPM[RPM != 0]   #very good for removing all the violation keys, and finding the max, but flattens the array
    lower = RPMalt.min()         #finds the minimum Score discounting violation trips
    upper = RPM.max()
    RPMplot = ax_dict['C'].contourf(X, Y, RPM, cmap = colormap, levels = np.linspace(lower, upper, grade))
    fig.colorbar(RPMplot, ticks = np.linspace(lower, upper, 11), pad = 0.025, shrink = 1.0, spacing = 'uniform', label= 'RPM', ax=ax_dict['C'])
    ax_dict['C'].set_ylabel('Freestream Velocity (fps)')
    
    # current plot
    Itotalt = Itot[Itot != 0]   #very good for removing all the violation keys, and finding the max, but flattens the array
    lower = Itotalt.min()         #finds the minimum Score discounting violation trips
    upper = Itot.max()
    Itotplot = ax_dict['D'].contourf(X, Y, Itot, cmap = colormap, levels = np.linspace(lower, upper, grade))
    fig.colorbar(Itotplot, ticks = np.linspace(lower, upper, 11), pad = 0.025, shrink = 1.0, spacing = 'uniform', label= 'Current (A)', ax=ax_dict['D'])

    if self.nmot > 1:
        s = 's'
    else:
        s = ''
    fig.suptitle(f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; {self.ns}S {self.CB} mAh battery; {self.nmot} APC {self.prop_name} propeller{s}; {self.ds*100:.0f}% battery discharge')
    plt.show()

#%% NUMBA BASED FUNCTIONS
@njit(fastmath=True)
def TorqueNumba(RPM, V, rpm_list, numba_prop_data):
    # numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    # with the structure [[Vvalues], [Thrustvalues], [TorqueValues]] for each index
    # data[0] = V values, data[1] = Thrust, data[2] = Torque!!!
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or V < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    torques = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        if data[0].size < 2:
            torques.append(0.0)
            continue
        torques.append(np.interp(V, data[0], data[2]))
        
    torques = np.array(torques)
    
    if len(closest_rpms) == 1:
        return torques[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * torques[0] + weight * torques[1]

@njit(fastmath=True)
def ThrustNumba(RPM, V, rpm_list, numba_prop_data):
    # numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    # with the structure [[Vvalues], [Thrustvalues], [TorqueValues]] for each index
    # data[0] = V values, data[1] = Thrust, data[2] = Torque!!!
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or V < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    thrusts = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        if data[0].size < 2:
            thrusts.append(0.0)
            continue
        thrusts.append(np.interp(V, data[0], data[1]))
        
    thrusts = np.array(thrusts)
    
    if len(closest_rpms) == 1:
        return thrusts[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * thrusts[0] + weight * thrusts[1]

@njit(fastmath=True)
def ModelCalcsNumba(velocity, t, rpm_list, numba_prop_data, CB, 
                        ns, Rint, KV, Rm, nmot, I0, ds):
    '''
    Optimized Numba-compatible version solving for T, P, Itot, RPM.
    Implements bisection for root finding on Itot, with fallback to alternate formulation.
    '''
    def residual_primary(Itot):
        if Itot < 0:
            return 1e10
        SOC = (CB*3.6 - Itot*t)/(CB*3.6)
        if SOC < 0 or SOC > 1:
            return 1e10
        Vsoc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
        Vin = ns * Vsoc - Itot * Rint * ns
        if Vin < 0:
            return 1e10
        RPM = KV * Vin - (KV * Rm / nmot) * Itot
        if RPM < 0 or RPM > rpm_list[-1]:
            return 1e10
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot_calc = Q * (KV * np.pi / 30) + I0
        return Itot - Itot_calc

    # Bisection for Itot (primary formulation)
    low = 0.0
    high = 400.0  # Adjust as needed
    tol = 1e-3
    max_iter = 100
    Itot = 100 #initial guess
    for _ in range(max_iter):
        mid = (low + high) / 2
        res_mid = residual_primary(mid)
        if abs(res_mid) < tol:
            Itot = mid
            break
        if res_mid > 0:
            high = mid
        else:
            low = mid
    else:
        # If not converged, try alternate or return zeros
        Itot = -1.0
    
    if Itot > 0 and Itot > I0 * nmot:
        # Compute values for primary case
        SOC = (CB*3.6 - Itot*t)/(CB*3.6)
        Vsoc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
        Vin = ns * Vsoc - Itot * Rint * ns
        RPM = KV * Vin - (KV * Rm / nmot) * Itot
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        T = nmot * ThrustNumba(RPM, velocity, rpm_list, numba_prop_data)
        P = (Itot / nmot) * Vin
        # this SOC check doesn't quite make sense!
        if SOC < (1 - ds) or Q <= 0 or T <= 0:
            return 0.0, 0.0, 0.0, 0.0
        return T, P, Itot, RPM
    
    # Alternate formulation (when Itot <= I0 * nmot or primary failed)
    def residual_alt(RPM):
        if RPM < 0 or RPM > rpm_list[-1]:
            return 1e10
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot_alt = Q * (KV * np.pi / 30) + I0
        SOC = 1.0 - (Itot_alt * t) / (CB * 3.6)
        if SOC < 0 or SOC > 1:
            return 1e10
        Vsoc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
        Vin = ns * Vsoc - Itot_alt * Rint * ns
        Imot = Itot_alt / nmot
        RPM_new = KV * (Vin - Imot * Rm)
        return RPM_new - RPM

    # Bisection for RPM in alternate
    low = rpm_list[0]
    high = rpm_list[-1]
    RPM = -1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        res_mid = residual_alt(mid)
        if abs(res_mid) < tol:
            RPM = mid
            break
        if res_mid > 0:
            high = mid
        else:
            low = mid
    else:
        return 0.0, 0.0, 0.0, 0.0
    
    if RPM > 0:
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot = Q * (KV * np.pi / 30) + I0
        SOC = 1.0 - (Itot * t) / (CB * 3.6)
        Vsoc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3
        Vin = ns * Vsoc - Itot * Rint * ns
        T = nmot * ThrustNumba(RPM, velocity, rpm_list, numba_prop_data)
        P = (Itot / nmot) * Vin
        if SOC < (1 - ds) or Itot < 0 or T <= 0:
            return 0.0, 0.0, 0.0, 0.0
        return T, P, Itot, RPM

    return 0.0, 0.0, 0.0, 0.0

#%% Pareto front for Static Thrust vs Cruise Speed for selected motor, battery
# use multiprocessing as default
#%% OLD VERSION WITHOUT NUMBA SPEEDUP Pareto front of velocity vs static thrust for a given motor + battery and all APC propellers!
# def process_propeller_thrust_cruise(propname, self, verbose = False):
#     try:
#         #NOTE: this was fucking up the results for a while bc I didn't redefine RPM_VALUES, etc
#         self.PROP_DATA, self.NUMBA_PROP_DATA = parse_propeller_data(propname)
#         self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS = initialize_RPM_polynomials(self.PROP_DATA)
        
#         # Get static thrust (V=0, t=0)
#         T_static, P_static, I_static, RPM_static = ModelCalcs(self, 0.0, 0.0)
        
#         if T_static <= 0:
#             return None
                        
#         # Get cruise speed (t=0, Tlimit=True)
#         V_cruise = VmaxLean(self, 0.0, Tlimit=True)
        
#         if V_cruise <= 0:
#             return None
        
#         if verbose:
#             name = propname.replace('PER3_', '').replace('.dat', '')
#             print(f'{name}: Static T = {T_static/lbfN:.2f} lbf, Cruise V = {V_cruise/ftm:.2f} fps')
            
#     except Exception as e:
#         print('here')
#         if verbose:
#             print(f'Failed for {propname}: {e}')
#         return None
    
#     return(propname, T_static/lbfN, V_cruise/ftm, P_static, I_static, RPM_static)

# def PlotTCPareto_mp(self, verbose=False):
#     '''
#     Plots the Pareto front of static thrust vs cruise speed for provided propellers.
    
#     Parameters:
#     proplist : list, optional
#         List of propeller filenames. If None, uses all props in database within diameter bounds.
#     lb : float
#         Lower bound for propeller diameter (inches)
#     ub : float  
#         Upper bound for propeller diameter (inches)
#     verbose : bool
#         Print detailed information during calculation
        
#     Returns:
#     pareto_data : list of tuples
#         List of (propname, static_thrust_lbf, cruise_speed_fps) for Pareto optimal props
#     '''
    
#     # Get propeller list if not provided
#     if self.proplist is None:
#         with open("Databases/PropDatabase/proplist.txt", "r") as data_file:
#             all_props = data_file.readlines()
        
#         self.proplist = []
#         for item in all_props:
#             if item == 'filelist.txt\n' or 'EP(CD)' in item or 'E(CD)' in item:
#                 continue
            
#             # Filter by diameter
#             try:
#                 test = item.replace('PER3_', '').split('x')[0]
#                 if float(test) < self.lb or float(test) > self.ub:
#                     continue
                
#                 propname_corrected = item.replace('PER3_', '').replace('.dat', '').replace('\n', '')
#                 self.proplist.append(propname_corrected)
#             except:
#                 continue
#     print(f'Analyzing {len(self.proplist)} propellers for Pareto front...')
    
#     # Calculate static thrust and cruise speed for each prop
#     prop_data = []
    
#     with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
#         process_func = partial(process_propeller_thrust_cruise, self=self)
#         results = list(tqdm(pool.imap(process_func, self.proplist)))
        
#     prop_data = [r for r in results if r is not None]
    
#     # print(len(prop_data))
    
#     if not prop_data:
#         print('No valid propeller data found!')
#         return []
    
#     # Find Pareto front
#     pareto_data = []
#     prop_data.sort(key=lambda x: x[1], reverse=True)  # Sort by static thrust descending
    
#     # gather 'violation points' and 'pareto points'
#     current_violations = []
#     power_violations = []
    
#     max_cruise_speed = 0
#     for prop in prop_data:
#         propname, static_thrust, cruise_speed, power, current, RPM = prop
#         if current > self.Ilimit:
#             current_violations.append(prop)
#         elif power > self.Pmax:
#             power_violations.append(prop)
#         elif cruise_speed > max_cruise_speed:
#             pareto_data.append(prop)
#             max_cruise_speed = cruise_speed
    
#     # Sort Pareto front by cruise speed for plotting
#     pareto_data.sort(key=lambda x: x[2])
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    
#     # Plot all props as light points
#     all_static = [p[1] for p in prop_data]
#     all_cruise = [p[2] for p in prop_data]
#     ax.scatter(all_cruise, all_static, alpha=0.8, s=20, 
#                color='lightgray', label='All Props')
    
#     # highlight violation points
#     current_static = [p[1] for p in current_violations]
#     current_cruise = [p[2] for p in current_violations]
#     ax.scatter(current_cruise, current_static, s=30, color='red', 
#                label=f'I > {self.Ilimit:.0f} A', zorder=2, marker='x')
    
#     power_static = [p[1] for p in power_violations]
#     power_cruise = [p[2] for p in power_violations]
#     ax.scatter(power_cruise, power_static, s=30, color='orange', 
#                label=f'Pmax > {self.Pmax:.0f} W', zorder=2, marker='^')

#     # Plot Pareto front
#     pareto_static = [p[1] for p in pareto_data]
#     pareto_cruise = [p[2] for p in pareto_data]
#     ax.plot(pareto_cruise, pareto_static, 'ro-', linewidth=2, markersize=8, label='Pareto Front')
    
#     # Annotate Pareto points
#     for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(pareto_data):
#         name = propname.replace('PER3_', '').replace('.dat', '')
#         ax.annotate(name, (cruise_speed, static_thrust), 
#                    xytext=(5, 5), textcoords='offset points',
#                    fontsize=8, ha='left', va='bottom',
#                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
#     # annotate all external datapoints if provided a set number of props 
#     if self.AnnotateAll:
#         for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(current_violations):
#             name = propname.replace('PER3_', '').replace('.dat', '')
#             ax.annotate(f'{name}, {current:.1f}A', (cruise_speed, static_thrust), 
#                        xytext=(5, 5), textcoords='offset points',
#                        fontsize=8, ha='left', va='bottom',
#                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
#         for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(power_violations):
#             name = propname.replace('PER3_', '').replace('.dat', '')
#             ax.annotate(f'{name}, {power:.1f}W', (cruise_speed, static_thrust), 
#                        xytext=(5, 5), textcoords='offset points',
#                        fontsize=8, ha='left', va='bottom',
#                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
#     ax.set_xlabel('Cruise Speed (fps)')
#     ax.set_ylabel('Static Thrust (lbf)')
#     ax.grid(True, alpha=0.3)
#     ax.minorticks_on()
#     ax.legend()
    
#     # Title with setup info
#     if self.nmot > 1:
#         s = 's'
#     else:
#         s = ''
        
#     plt.title(f'Pareto Front: Static Thrust vs Cruise Speed\n'
#               f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; {self.ns}S {self.CB} mAh battery;\n'
#               f'100% Battery; Cd = {self.CD}, Sw = {self.Sw:.3f} ft²')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print results
#     print(f'\nPareto Front Results ({len(pareto_data)} propellers):')
#     print(f'{"Propeller":<20} {"Static Thrust (lbf)":<18} {"Cruise Speed (fps)":<18}')
#     print('-' * 60)
    
#     for propname, static_thrust, cruise_speed, power, current, RPM in pareto_data:
#         name = propname.replace('PER3_', '').replace('.dat', '')
#         print(f'{name:<20} {static_thrust:<18.2f} {cruise_speed:<18.2f}')
        
#     return pareto_data

#%% numba version of velocity vs thrust pareto front!
@njit(fastmath=True)
def VmaxLeanNumba(velocity_max, rpm_list, numba_prop_data, CB, ns, Rint, KV, Rm, nmot, I0, ds, rho, CD, Sw, t_in = 0.0):
    """
    Numba-optimized version of VmaxLean that finds maximum velocity where T >= D
    without using GEKKO. Uses bisection method for root finding.
    """
    def thrust_minus_drag_residual(V):
        if V <= 0:
            return -1e10
        
        # Calculate drag at this velocity
        D = 0.5 * rho * (V**2) * CD * Sw
        
        # Find maximum thrust available at this velocity (t=0, full battery)
        T, P, Itot, RPM_actual = ModelCalcsNumba(V, t_in, rpm_list, numba_prop_data, 
                                             CB, ns, Rint, KV, Rm, nmot, I0, ds)
        return T - D
    
    # Bisection method to find maximum velocity where T >= D
    low = 1e-3
    high = velocity_max
    tol = 1e-3  # 0.1 m/s tolerance
    max_iter = 100
    
    # Check if solution exists
    if thrust_minus_drag_residual(high) < 0:
        # Even at max velocity, thrust < drag, so find the highest velocity with positive thrust
        for i in range(max_iter):
            mid = (low + high) / 2
            if thrust_minus_drag_residual(mid) >= 0:
                low = mid
            else:
                high = mid
            if (high - low) < tol:
                break
        return low
    
    # Normal bisection for T = D
    for i in range(max_iter):
        mid = (low + high) / 2
        residual = thrust_minus_drag_residual(mid)
        
        if abs(residual) < 1.0:  # 1N tolerance for thrust-drag balance
            return mid
        
        if residual > 0:
            low = mid
        else:
            high = mid
            
        if (high - low) < tol:
            break
    
    return (low + high) / 2

def process_propeller_thrust_cruise_optimized(propname, self, verbose=False):
    """
    Optimized version using numba-compatible functions and avoiding GEKKO.
    """
    try:
        # Parse propeller data
        self.PROP_DATA, self.NUMBA_PROP_DATA = parse_propeller_data(propname)
        self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS = initialize_RPM_polynomials(self.PROP_DATA)
        
        # Convert to numpy arrays for numba compatibility
        rpm_list = np.array(self.RPM_VALUES, dtype=np.float64)
        
        # Get static thrust (V=0, t=0) using optimized function
        T_static, P_static, I_static, RPM_static = ModelCalcsNumba(
            0.0, 0.0, rpm_list, self.NUMBA_PROP_DATA, 
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds
        )
        
        # Fallback to scipy version if numba version fails
        if T_static <= 0:
            T_static, P_static, I_static, RPM_static = ModelCalcs(self, 0.0, 0.0)
        
        if T_static <= 0:
            return None
        
        # Get maximum velocity from propeller data
        velocity_max = self.V_DOMAINS[-1]
        
        # Get cruise speed using numba-optimized function
        V_cruise = VmaxLeanNumba(
            velocity_max, rpm_list, self.NUMBA_PROP_DATA,
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds,
            self.rho, self.CD, self.Sw
        )
                
        # if V_cruise <= 0.1:
            # fallback to gekko version
            # V_cruise = VmaxLean(self, self.ds, Tlimit=True)
        if V_cruise <= 0.1:
            return None
        
        if verbose:
            name = propname.replace('PER3_', '').replace('.dat', '')
            print(f'{name}: Static T = {T_static/lbfN:.2f} lbf, Cruise V = {V_cruise/ftm:.2f} fps')
            
        return (propname, T_static/lbfN, V_cruise/ftm, P_static, I_static, RPM_static)
        
    except Exception as e:
        if verbose:
            print(f'Failed for {propname}: {e}')
        return None

def PlotTCPareto_mp(self, verbose=False):
    """
    Optimized version of PlotTCPareto_mp using the numba bisection based methods for 
    Vmax and ModelCalcs
    """
    if self.proplist is None:
        with open("Databases/PropDatabase/proplist.txt", "r") as data_file:
            all_props = data_file.readlines()
        
        self.proplist = []
        for item in all_props:
            if item == 'filelist.txt\n' or 'EP(CD)' in item or 'E(CD)' in item:
                continue
            
            # Filter by diameter
            try:
                test = item.replace('PER3_', '').split('x')[0]
                if float(test) < self.lb or float(test) > self.ub:
                    continue
                
                propname_corrected = item.replace('PER3_', '').replace('.dat', '').replace('\n', '')
                self.proplist.append(propname_corrected)
            except:
                continue
    
    print(f'Analyzing {len(self.proplist)} propellers for Pareto front (optimized)...')
    
    # Calculate static thrust and cruise speed for each prop using optimized function
    prop_data = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        process_func = partial(process_propeller_thrust_cruise_optimized, self=self)
        results = list(tqdm(pool.imap(process_func, self.proplist)))
        
    prop_data = [r for r in results if r is not None]
    
    if not prop_data:
        print('No valid propeller data found!')
        return []
    
    # Find Pareto front
    pareto_data = []
    prop_data.sort(key=lambda x: x[1], reverse=True)  # Sort by static thrust descending
    
    # gather 'violation points' and 'pareto points'
    current_violations = []
    power_violations = []
    
    max_cruise_speed = 0
    for prop in prop_data:
        propname, static_thrust, cruise_speed, power, current, RPM = prop
        if current > self.Ilimit:
            current_violations.append(prop)
        elif power > self.Pmax:
            power_violations.append(prop)
        elif cruise_speed > max_cruise_speed:
            pareto_data.append(prop)
            max_cruise_speed = cruise_speed
    
    # Sort Pareto front by cruise speed for plotting
    pareto_data.sort(key=lambda x: x[2])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
    
    # Plot all props as light points
    all_static = [p[1] for p in prop_data]
    all_cruise = [p[2] for p in prop_data]
    ax.scatter(all_cruise, all_static, alpha=0.8, s=20, 
               color='lightgray', label='All Props')
    
    # highlight violation points
    current_static = [p[1] for p in current_violations]
    current_cruise = [p[2] for p in current_violations]
    ax.scatter(current_cruise, current_static, s=30, color='red', 
               label=f'I > {self.Ilimit:.0f} A', zorder=2, marker='x')
    
    power_static = [p[1] for p in power_violations]
    power_cruise = [p[2] for p in power_violations]
    ax.scatter(power_cruise, power_static, s=30, color='orange', 
               label=f'Pmax > {self.Pmax:.0f} W', zorder=2, marker='^')

    # Plot Pareto front
    pareto_static = [p[1] for p in pareto_data]
    pareto_cruise = [p[2] for p in pareto_data]
    ax.plot(pareto_cruise, pareto_static, 'ro-', linewidth=2, markersize=8, label='Pareto Front')
    
    # Annotate Pareto points
    for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(pareto_data):
        name = propname.replace('PER3_', '').replace('.dat', '')
        ax.annotate(name, (cruise_speed, static_thrust), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # annotate all external datapoints if provided a set number of props 
    if self.AnnotateAll:
        for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(current_violations):
            name = propname.replace('PER3_', '').replace('.dat', '')
            ax.annotate(f'{name}, {current:.1f}A', (cruise_speed, static_thrust), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
        for i, (propname, static_thrust, cruise_speed, power, current, RPM) in enumerate(power_violations):
            name = propname.replace('PER3_', '').replace('.dat', '')
            ax.annotate(f'{name}, {power:.1f}W', (cruise_speed, static_thrust), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
    ax.set_xlabel('Cruise Speed (fps)')
    ax.set_ylabel('Static Thrust (lbf)')
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.legend()
    
    # Title with setup info
    if self.nmot > 1:
        s = 's'
    else:
        s = ''
        
    plt.title(f'Pareto Front: Static Thrust vs Cruise Speed (Optimized)\n'
              f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; {self.ns}S {self.CB} mAh battery;\n'
              f'100% Battery; Cd = {self.CD}, Sw = {self.Sw:.3f} ft²')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f'\nPareto Front Results ({len(pareto_data)} propellers):')
    print(f'{"Propeller":<20} {"Static Thrust (lbf)":<18} {"Cruise Speed (fps)":<18}')
    print('-' * 60)
    
    for propname, static_thrust, cruise_speed, power, current, RPM in pareto_data:
        name = propname.replace('PER3_', '').replace('.dat', '')
        print(f'{name:<20} {static_thrust:<18.2f} {cruise_speed:<18.2f}')
        
    return pareto_data

#%% OLD Takeoff Pareto Front Functions (single motor + battery, all propellers for now)
# def calculate_ground_roll_optimized(self):
#     """
#     Optimized ground roll calculation using simplified physics model.
#     """
    
#     try:
#         # Use average values approach for faster calculation
#         # Sample thrust at key velocities
#         V_samples = np.linspace(0.1, self.Vlof, 10)
#         thrust_samples = []
        
#         for V in V_samples:
#             T_N, _, _, _ = ModelCalcs(self, V, 0.0)
#             if T_N <= 0:
#                 return None
#             thrust_samples.append(T_N)
        
#         # Average thrust during ground roll
#         T_avg = np.mean(thrust_samples)
        
#         # Average aerodynamic properties during ground roll
#         V_avg = self.Vlof / 2  # Approximate average velocity
#         q_avg = 0.5 * self.rho * V_avg**2
        
#         # Approximate lift coefficient during ground roll
#         CL_avg = self.CLto  # takeoff lift coefficient (0 deg aoa + high lift devices)
        
#         # Drag coefficient
#         CD_avg = self.CD0 + (CL_avg**2) / (np.pi * self.AR * self.e)
        
#         # Average forces
#         L_avg = q_avg * self.Sw * CL_avg
#         D_avg = q_avg * self.Sw * CD_avg
#         N_avg = max(0, self.MGTOW - L_avg)
#         F_friction_avg = self.mufric * N_avg #mufric is the wheel rolling resistance (defaults to 0..04)
        
#         # Net force and acceleration
#         F_net_avg = T_avg - D_avg - F_friction_avg
        
#         if F_net_avg <= 0:
#             return None  # Cannot accelerate
        
#         a_avg = F_net_avg / (self.MGTOW/9.81) # g = 9.81 for now
        
#         # Kinematic equation: V² = V₀² + 2as, where V₀ = 0
#         # Solving for s: s = V² / (2a)
#         distance_m = (self.Vlof**2) / (2 * a_avg)
#         distance_ft = distance_m / ftm
#         return distance_ft
        
#     except Exception:
#         return None

# def process_takeoff_cruise(propname, self, verbose = False):
#     try:
#         # Parse propeller data
#         self.PROP_DATA, self.NUMBA_PROP_DATA = parse_propeller_data(propname)
#         self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS = initialize_RPM_polynomials(self.PROP_DATA)
        
#         # Optimized ground roll calculation
#         takeoff_distance_ft = calculate_ground_roll_optimized(self)
#         if takeoff_distance_ft is None or takeoff_distance_ft <= 0.0: # tloflimit:
#             return(None)
        
#         # Get cruise speed at t=0 (full battery)
#         V_cruise = VmaxLean(self, 0.0, t_in=0.0, Tlimit=True)
        
#         if V_cruise <= 0:
#             return(None)
        
#         # # Get static thrust and power for current/power limit checking
#         T_static, P_static, I_static, RPM_static = ModelCalcs(self, 0.0, 0.0)
        
#         if T_static <= 0:
#             return(None)
            
#         return(propname, V_cruise/ftm, takeoff_distance_ft, 
#                P_static, I_static, RPM_static)
        
#         if verbose:
#             name = propname.replace('PER3_', '').replace('.dat', '')
#             print(f'{name}: Cruise V = {V_cruise/ftm:.2f} fps, '
#                   f'Takeoff dist = {takeoff_distance_ft:.0f} ft')
            
#     except Exception as e:
#         if verbose:
#             print(f'Failed for {propname}: {e}')
#         return(None)
    
# def plotTakeoffParetoFront(self, verbose=False):
#     '''
#     Plots the Pareto front of cruise speed vs takeoff distance for provided propellers.
#     '''    
#     if self.proplist is None:
#         with open("Databases/PropDatabase/proplist.txt", "r") as data_file:
#             all_props = data_file.readlines()
        
#         self.proplist = []
#         for item in all_props:
#             if item == 'filelist.txt\n' or 'EP(CD)' in item or 'E(CD)' in item:
#                 continue
            
#             # Filter by diameter
#             try:
#                 test = item.replace('PER3_', '').split('x')[0]
#                 if float(test) < self.lb or float(test) > self.ub:
#                     continue
                
#                 propname_corrected = item.replace('PER3_', '').replace('.dat', '').replace('\n', '')
#                 self.proplist.append(propname_corrected)
#             except:
#                 continue
#     print(f'Analyzing {len(self.proplist)} propellers for takeoff Pareto front...')
            
#     # Calculate propeller performance data
#     prop_data = []
    
#     with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
#         process_func = partial(process_takeoff_cruise, self=self)
#         results = list(tqdm(pool.imap(process_func, self.proplist)))
        
#     prop_data = [r for r in results if r is not None]
        
    
#     if not prop_data:
#         print('No valid propeller data found!')
#         return []
    
#     # Categorize propellers by violations
#     pareto_data = []
#     current_violations = []
#     power_violations = []
#     takeoff_violations = []
    
#     # Sort by cruise speed for Pareto front identification
#     prop_data.sort(key=lambda x: x[1], reverse=True)
    
#     min_takeoff_distance = float('inf')
#     for prop in prop_data:
#         propname, cruise_speed, takeoff_distance, power, current, RPM = prop
        
#         # Check violations
#         if current > self.Ilimit:
#             current_violations.append(prop)
#         elif power > self.Pmax:
#             power_violations.append(prop)
#         elif takeoff_distance > self.xloflimit:
#             takeoff_violations.append(prop)
#         elif takeoff_distance < min_takeoff_distance:
#             # This prop is Pareto optimal (better takeoff distance)
#             pareto_data.append(prop)
#             min_takeoff_distance = takeoff_distance
    
#     # Sort Pareto front by cruise speed for plotting
#     pareto_data.sort(key=lambda x: x[1])
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)
    
#     # Plot all valid props as light points
#     all_cruise = [p[1] for p in prop_data]
#     all_takeoff = [p[2] for p in prop_data]
#     ax.scatter(all_cruise, all_takeoff, alpha=0.8, s=20, color='lightgray', 
#                label='All Props', zorder=1)
    
#     # Highlight violation points
#     if current_violations:
#         current_cruise = [p[1] for p in current_violations]
#         current_takeoff = [p[2] for p in current_violations]
#         ax.scatter(current_cruise, current_takeoff, s=30, color='red', 
#                    label=f'I > {self.Ilimit}A', zorder=2, marker='x')
    
#     if power_violations:
#         power_cruise = [p[1] for p in power_violations]
#         power_takeoff = [p[2] for p in power_violations]
#         ax.scatter(power_cruise, power_takeoff, s=30, color='orange', 
#                    label=f'P > {self.Pmax}W', zorder=2, marker='^')
    
#     if takeoff_violations:
#         takeoff_cruise = [p[1] for p in takeoff_violations]
#         takeoff_takeoff = [p[2] for p in takeoff_violations]
#         ax.scatter(takeoff_cruise, takeoff_takeoff, s=30, color='purple', 
#                    label=f'TO > {self.xloflimit}ft', zorder=2, marker='s')
    
#     # Plot Pareto front
#     if pareto_data:
#         pareto_cruise = [p[1] for p in pareto_data]
#         pareto_takeoff = [p[2] for p in pareto_data]
#         ax.plot(pareto_cruise, pareto_takeoff, 'go-', linewidth=3, markersize=10, 
#                 label='Pareto Front', zorder=4)
        
#         # Annotate Pareto points
#         for propname, cruise_speed, takeoff_distance, power, current, RPM in pareto_data:
#             name = propname.replace('PER3_', '').replace('.dat', '')
#             ax.annotate(name, (cruise_speed, takeoff_distance), 
#                        xytext=(5, 5), textcoords='offset points',
#                        fontsize=9, ha='left', va='bottom', weight='bold',
#                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', 
#                                 alpha=0.8, edgecolor='darkgreen'))
    
#     # Add limit lines
#     ax.axhline(y=self.xloflimit, color='purple', linestyle='--', alpha=0.7,
#                    label=f'Takeoff Limit ({self.xloflimit} ft)')
    
#     ax.set_xlabel('Cruise Speed (fps)', fontsize=12)
#     ax.set_ylabel('Takeoff Distance (ft)', fontsize=12)
    
#     # plot scaling
#     test = np.array(all_takeoff)
#     # first get distance between minimum and maximum:
#     # minmax = test.max() - test.min()
#     # if len(pareto_takeoff) > 0:
#     #     ax.set_ylim([0, min(pareto_takeoff)*1.5])
    
#     if test.min() > self.xloflimit:
#         ax.set_ylim([0, test.min()*1.5])
#     else:
#         ax.set_ylim([0, self.xloflimit*1.5])
#     ax.grid(True, alpha=0.3)
#     ax.minorticks_on()
#     ax.legend(fontsize=10)
    
#     # Invert y-axis so shorter takeoff distances are "better" (higher on plot)
#     ax.invert_yaxis()
    
#     # Title with setup info
#     if self.nmot > 1:
#         s = 's'
#     else:
#         s = ''
#     plt.title(f'Pareto Front: Cruise Speed vs Takeoff Distance\n'
#               f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; {self.ns}S {self.CB} mAh battery; '
#               f'100% Battery\n'
#               f'MGTOW = {self.MGTOW/lbfN:.2f} lbs, Sw = {self.Sw/ftm/ftm:.2f} ft², AR = {self.AR}, CL_max = {self.CLmax}, CD0 = {self.CD0}, e = {self.e}'
#               , fontsize=11)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print results
#     print(f'\nPareto Front Results ({len(pareto_data)} propellers):')
#     print(f'{"Propeller":<20} {"Cruise Speed (fps)":<18} {"Takeoff Dist (ft)":<18}')
#     print('-' * 60)
    
#     for propname, cruise_speed, takeoff_distance, power, current, RPM in pareto_data:
#         name = propname.replace('PER3_', '').replace('.dat', '')
#         print(f'{name:<20} {cruise_speed:<18.2f} {takeoff_distance:<18.0f}')
    
#     print('\nViolation Summary:')
#     print(f'Current violations (I > {self.Ilimit}A): {len(current_violations)}')
#     print(f'Power violations (P > {self.Pmax}W): {len(power_violations)}')
#     print(f'Takeoff violations (TO > {self.xloflimit}ft): {len(takeoff_violations)}')
        
#     return pareto_data

#%% Takeoff Pareto Front with numba
@njit(fastmath=True)
def calculate_ground_roll_numba(velocity_lof, rpm_list, numba_prop_data, CB, ns, Rint, 
                               KV, Rm, nmot, I0, ds, rho, Sw, CLto, CD0, AR, e, 
                               MGTOW, mufric):
    """
    Numba-optimized ground roll calculation using simplified physics model.
    """
    # Sample thrust at key velocities during ground roll
    V_samples = np.linspace(0.1, velocity_lof, 10)
    thrust_samples = np.zeros(len(V_samples))
    
    valid_samples = 0
    for i, V in enumerate(V_samples):
        T_N, _, _, _ = ModelCalcsNumba(V, 0.0, rpm_list, numba_prop_data, 
                                         CB, ns, Rint, KV, Rm, nmot, I0, ds)
        if T_N > 0:
            thrust_samples[i] = T_N
            valid_samples += 1
        else:
            return -1.0  # Invalid configuration
    
    if valid_samples == 0:
        return -1.0
    
    # Average thrust during ground roll (alternatively, use thrust at 1/sqrt(2)*Vlof as raymer says!!)
    T_avg = np.mean(thrust_samples)
    
    # Average aerodynamic properties during ground roll
    V_avg = velocity_lof / 2  # Approximate average velocity # CHANGE TO 1/sqrt(2)*Vlof soon
    q_avg = 0.5 * rho * V_avg**2
    
    # Drag coefficient
    CD_avg = CD0 + (CLto**2) / (np.pi * AR * e)
    
    # Average forces
    L_avg = q_avg * Sw * CLto
    D_avg = q_avg * Sw * CD_avg
    N_avg = max(0.0, MGTOW - L_avg)
    F_friction_avg = mufric * N_avg
    
    # Net force and acceleration
    F_net_avg = T_avg - D_avg - F_friction_avg
    
    if F_net_avg <= 0:
        return -1.0  # Cannot accelerate
    
    g = 9.81
    a_avg = F_net_avg / (MGTOW / g)
    
    # Kinematic equation: V² = V₀² + 2as, where V₀ = 0
    # Solving for s: s = V² / (2a)
    distance_m = (velocity_lof**2) / (2 * a_avg)
    return distance_m

def process_takeoff_cruise_optimized(propname, self, verbose=False):
    """
    Optimized version using numba-compatible functions.
    """
    try:
        # Parse propeller data
        self.PROP_DATA, self.NUMBA_PROP_DATA = parse_propeller_data(propname)
        self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS = initialize_RPM_polynomials(self.PROP_DATA)
        
        # Convert to numpy arrays for numba compatibility
        rpm_list = np.array(self.RPM_VALUES, dtype=np.float64)
        
        # Calculate takeoff distance using numba-optimized function
        takeoff_distance_m = calculate_ground_roll_numba(
            self.Vlof, rpm_list, self.NUMBA_PROP_DATA,
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds,
            self.rho, self.Sw, self.CLto, self.CD0, self.AR, self.e, 
            self.MGTOW, self.mufric
        )
        
        if takeoff_distance_m <= 0.0:
            return None
            
        takeoff_distance_ft = takeoff_distance_m / ftm
        
        # Get cruise speed using numba-optimized function
        velocity_max = self.V_DOMAINS[-1]
        V_cruise = VmaxLeanNumba(
            velocity_max, rpm_list, self.NUMBA_PROP_DATA,
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds,
            self.rho, self.CD, self.Sw
        )
        
        if V_cruise <= 0.1:
            return None
        
        # Get static thrust and power for current/power limit checking using optimized function
        T_static, P_static, I_static, RPM_static = ModelCalcsNumba(
            0.0, 0.0, rpm_list, self.NUMBA_PROP_DATA, 
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds
        )
        
        # Fallback to scipy version if numba version fails
        if T_static <= 0:
            T_static, P_static, I_static, RPM_static = ModelCalcs(self, 0.0, 0.0)
        
        if T_static <= 0:
            return None
            
        if verbose:
            name = propname.replace('PER3_', '').replace('.dat', '')
            print(f'{name}: Cruise V = {V_cruise/ftm:.2f} fps, '
                  f'Takeoff dist = {takeoff_distance_ft:.0f} ft')
            
        return (propname, V_cruise/ftm, takeoff_distance_ft, 
               P_static, I_static, RPM_static)
        
    except Exception as e:
        if verbose:
            print(f'Failed for {propname}: {e}')
        return None

def plotTakeoffParetoFrontNumba(self, verbose=False):
    '''
    Optimized version of plotTakeoffParetoFront using numba optimizations.
    '''    
    if self.proplist is None:
        with open("Databases/PropDatabase/proplist.txt", "r") as data_file:
            all_props = data_file.readlines()
        
        self.proplist = []
        for item in all_props:
            if item == 'filelist.txt\n' or 'EP(CD)' in item or 'E(CD)' in item:
                continue
            
            # Filter by diameter
            try:
                test = item.replace('PER3_', '').split('x')[0]
                if float(test) < self.lb or float(test) > self.ub:
                    continue
                
                propname_corrected = item.replace('PER3_', '').replace('.dat', '').replace('\n', '')
                self.proplist.append(propname_corrected)
            except:
                continue
    
    print(f'Analyzing {len(self.proplist)} propellers for takeoff Pareto front...')
            
    # Calculate propeller performance data using optimized function
    prop_data = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        process_func = partial(process_takeoff_cruise_optimized, self=self)
        results = list(tqdm(pool.imap(process_func, self.proplist)))
        
    prop_data = [r for r in results if r is not None]
    
    if not prop_data:
        print('No valid propeller data found!')
        return []
    
    # Categorize propellers by violations
    pareto_data = []
    current_violations = []
    power_violations = []
    takeoff_violations = []
    
    # Sort by cruise speed for Pareto front identification
    prop_data.sort(key=lambda x: x[1], reverse=True)
    
    min_takeoff_distance = float('inf')
    for prop in prop_data:
        propname, cruise_speed, takeoff_distance, power, current, RPM = prop
        
        # Check violations
        if current > self.Ilimit:
            current_violations.append(prop)
        elif power > self.Pmax:
            power_violations.append(prop)
        elif takeoff_distance > self.xloflimit:
            takeoff_violations.append(prop)
        elif takeoff_distance < min_takeoff_distance:
            # This prop is Pareto optimal (better takeoff distance)
            pareto_data.append(prop)
            min_takeoff_distance = takeoff_distance
    
    # Sort Pareto front by cruise speed for plotting
    pareto_data.sort(key=lambda x: x[1])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)
    
    # Plot all valid props as light points
    all_cruise = [p[1] for p in prop_data]
    all_takeoff = [p[2] for p in prop_data]
    ax.scatter(all_cruise, all_takeoff, alpha=0.8, s=20, color='lightgray', 
               label='All Props', zorder=1)
    
    # Highlight violation points
    if current_violations:
        current_cruise = [p[1] for p in current_violations]
        current_takeoff = [p[2] for p in current_violations]
        ax.scatter(current_cruise, current_takeoff, s=30, color='red', 
                   label=f'I > {self.Ilimit}A', zorder=2, marker='x')
    
    if power_violations:
        power_cruise = [p[1] for p in power_violations]
        power_takeoff = [p[2] for p in power_violations]
        ax.scatter(power_cruise, power_takeoff, s=30, color='orange', 
                   label=f'P > {self.Pmax}W', zorder=2, marker='^')
    
    if takeoff_violations:
        takeoff_cruise = [p[1] for p in takeoff_violations]
        takeoff_takeoff = [p[2] for p in takeoff_violations]
        ax.scatter(takeoff_cruise, takeoff_takeoff, s=30, color='purple', 
                   label=f'TO > {self.xloflimit}ft', zorder=2, marker='s')
    
    # Plot Pareto front
    if pareto_data:
        pareto_cruise = [p[1] for p in pareto_data]
        pareto_takeoff = [p[2] for p in pareto_data]
        ax.plot(pareto_cruise, pareto_takeoff, 'go-', linewidth=3, markersize=10, 
                label='Pareto Front', zorder=4)
        
        # Annotate Pareto points
        for propname, cruise_speed, takeoff_distance, power, current, RPM in pareto_data:
            name = propname.replace('PER3_', '').replace('.dat', '')
            ax.annotate(name, (cruise_speed, takeoff_distance), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left', va='bottom', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', 
                                alpha=0.8, edgecolor='darkgreen'))
    
    # Add limit lines
    ax.axhline(y=self.xloflimit, color='purple', linestyle='--', alpha=0.7,
                   label=f'Takeoff Limit ({self.xloflimit} ft)')
    
    ax.set_xlabel('Cruise Speed (fps)', fontsize=12)
    ax.set_ylabel('Takeoff Distance (ft)', fontsize=12)
    
    # plot scaling
    test = np.array(all_takeoff)
    
    if test.min() > self.xloflimit:
        ax.set_ylim([0, test.min()*1.5])
    else:
        ax.set_ylim([0, self.xloflimit*1.5])
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.legend(fontsize=10)
    
    # Invert y-axis so shorter takeoff distances are "better" (higher on plot)
    ax.invert_yaxis()
    
    # Title with setup info
    if self.nmot > 1:
        s = 's'
    else:
        s = ''
    plt.title(f'Pareto Front: Cruise Speed vs Takeoff Distance (Optimized)\n'
              f'{self.nmot} {self.motor_manufacturer} {self.motor_name} motor{s}; {self.ns}S {self.CB} mAh battery; '
              f'100% Battery\n'
              f'MGTOW = {self.MGTOW/lbfN:.2f} lbs, Sw = {self.Sw/ftm/ftm:.2f} ft², AR = {self.AR}, CL_max = {self.CLmax}, CD0 = {self.CD0}, e = {self.e}'
              , fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f'\nPareto Front Results ({len(pareto_data)} propellers):')
    print(f'{"Propeller":<20} {"Cruise Speed (fps)":<18} {"Takeoff Dist (ft)":<18}')
    print('-' * 60)
    
    for propname, cruise_speed, takeoff_distance, power, current, RPM in pareto_data:
        name = propname.replace('PER3_', '').replace('.dat', '')
        print(f'{name:<20} {cruise_speed:<18.2f} {takeoff_distance:<18.0f}')
    
    print('\nViolation Summary:')
    print(f'Current violations (I > {self.Ilimit}A): {len(current_violations)}')
    print(f'Power violations (P > {self.Pmax}W): {len(power_violations)}')
    print(f'Takeoff violations (TO > {self.xloflimit}ft): {len(takeoff_violations)}')
        
    return pareto_data


#%% MGTOW vs Cruise velocity pareto front
from itertools import cycle

def switcher(options):
    option_cycler = cycle(options)
    def switch_option():
        return next(option_cycler)
    return switch_option

def MGTOWinnerfunc_fast(self, MGTOWinit):
    """
    Fast version of MGTOWinnerfunc using scipy optimization instead of GEKKO.
    Uses analytical relationships and scipy.optimize for much better performance.
    
    Old func so it returns options instead of modifying the PointDesign object
    """
    g = 9.81  # m/s^2
    takeofflimit_m = self.xloflimit * ftm  # Convert to meters
    # Cache for repeated calculations
    cache = {}
    
    def objective_and_constraints(x):
        """
        x = [MGTOW, RPM]
        Returns: (-MGTOW_N, constraints)
        """
        key = tuple(x)
        if key in cache:
            return cache[key]
        
        MGTOW, RPM = x
        
        # Basic bounds checking
        if (MGTOW <= 0 or RPM <= self.RPM_VALUES[0] or RPM >= self.RPM_VALUES[-1]):
            result = 1e10, np.array([1e10] * 8)  # Large penalty
            cache[key] = result
            return result
        
        try:
            # Calculate aerodynamic properties
            Vstall = np.sqrt((2 * MGTOW) / (self.rho * self.Sw * self.CLmax))
            Vlof = 1.15 * Vstall
            V = Vlof*(1/np.sqrt(2)) # calculating values at 70% of distance
            
            # Propulsion calculations
            Q = self.nmot * Torque(self, RPM, V)
            
            if Q <= 0:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result
                
            Itot = Q * (self.KV * 2 * np.pi / 60) + self.I0
            if Itot <= self.I0:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result
                
            # Battery calculations (using t=1.0 for consistency)
            t = 1.0
            SOC = (self.CB * 3.6 - Itot * t) / (self.CB * 3.6)
            if SOC <= 0 or SOC >= 1:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result
                
            Vsoc = (-1.031 * np.exp(-35 * SOC) + 3.685 + 
                   0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * (SOC**3))
            Vin = (Vsoc - Itot * self.Rint) * self.ns
            
            if Vin <= 0:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result
                
            Imot = Itot / self.nmot
            RPM_calc = self.KV * (Vin - Imot * self.Rm)
            
            T = self.nmot * Thrust(self, RPM, V)
            if T <= 0:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result
            
            # Ground roll calculation using Raymer approximation
            Ka = (self.rho / (2 * self.MGTOW / self.Sw)) * (self.mufric * self.CLto - self.CD0 - (self.CLto**2) / (np.pi * self.AR * self.e))
            Kt = (T / MGTOW) - self.mufric
            
            if (Kt + Ka * (Vlof**2)) / (Kt + Ka * (1**2)) <= 0.0:
                result = 1e10, np.array([1e10] * 8)
                cache[key] = result
                return result

            xlof = (1 / (2 * g * Ka)) * np.log((Kt + Ka * (Vlof**2)) / (Kt + Ka * (1**2)))
            
            # Objective: maximize MGTOW (minimize negative MGTOW)
            objective = -MGTOW
            
            # Constraints (all should be <= 0)
            constraints = np.array([
                abs(RPM_calc - RPM) - 1e-6,  # RPM consistency
                -Q,  # Q > 0
                -(Itot - self.I0),  # Itot > I0
                -T,  # T > 0
                xlof - takeofflimit_m,  # xlof <= takeoff limit
                RPM - self.RPM_VALUES[-1],  # RPM <= RPM_max
                self.RPM_VALUES[0] - RPM,  # RPM >= RPM_min
                -xlof, #xlof > 0!! this fixes the problems
                
            ])
            
            result = objective, constraints
            cache[key] = result
            return result
            
        except Exception:
            result = 1e10, np.array([1e10] * 8)
            cache[key] = result
            return result
    
    def constraint_func(x):
        _, constraints = objective_and_constraints(x)
        return constraints
    
    def objective_func(x):
        obj, _ = objective_and_constraints(x)
        return obj
    
    # Initial guess - improved to be within bounds
    MGTOW_init = MGTOWinit * lbfN # NOTE: restricted to 60 lbs maximum for now; change when eventually applying to different aircraft
    RPM_init = max(self.RPM_VALUES[0] + 1, min((self.RPM_VALUES[0] + self.RPM_VALUES[-1]) / 2, self.RPM_VALUES[-1] - 1))
    x0 = np.array([MGTOW_init, RPM_init])
    
    # Bounds - tightened for better convergence
    bounds = [
        (max(1.0 * lbfN, MGTOW_init * 0.5), min(100.0 * lbfN, MGTOW_init * 2.0)),  # MGTOW bounds
        (self.RPM_VALUES[0] + 10, self.RPM_VALUES[-1] - 10),  # RPM bounds
    ]
    
    # Constraint definition for scipy
    constraints = {'type': 'ineq', 'fun': lambda x: -constraint_func(x)}
    
    try:
        # Use SLSQP with reduced tolerance for faster convergence
        result = opt.minimize(objective_func, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'ftol': 1e-4, 'disp': False, 'maxiter': 50})
        if result.success and result.fun < 1e9:
            MGTOW_N, RPM_final = result.x
            return MGTOW_N / lbfN
        else:
            return 0.0
            
    except Exception:
        return 0.0
    
def process_MGTOW_cruise(propname, self, motor):
    '''
    moved process_propeller outside of the MotorsMGTOWParetoPlot_fast function
    and you need to reapply InitializeSetup here so the motor changes 
    get applied to the prop data calcs!
        
    '''
    self.Motor(motor, self.nmot) # initialize motor parameters using the PointDesign function
    
    try:
        # Parse propeller data
        self.PROP_DATA, self.NUMBA_PROP_DATA = parse_propeller_data(propname)
        self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS = initialize_RPM_polynomials(self.PROP_DATA)
        
        
        # Get static thrust and power for current/power limit checking
        # old function:
        # T_static, P_static, I_static, RPM_static = ModelCalcs(self, 0.0, 0.0)
        
        rpm_list = np.array(self.RPM_VALUES)
        T_static, P_static, I_static, RPM_static = ModelCalcsNumba(0.0, 0.0, rpm_list, self.NUMBA_PROP_DATA, 
                                                                   self.CB, self.ns, self.Rint, self.KV, 
                                                                   self.Rm, self.nmot, self.I0, self.ds)
        # print('thrust', T_static / lbfN)
        if T_static <= 0:
            return None
        
        if self.SkipInvalid:
            if P_static > self.Pmax or I_static > self.Ilimit:
                return None
            
        # Get cruise speed at t=0 (full battery)
        # old function:
        # V_cruise = VmaxLean(self, 0.0, t_in = 0.0, Tlimit = True)
        
        # convergence can be +- 8 fps off!! Not sure what to do about that though
        velocity_max = self.V_DOMAINS[-1]
        V_cruise = VmaxLeanNumba(
            velocity_max, rpm_list, self.NUMBA_PROP_DATA,
            self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds,
            self.rho, self.CD, self.Sw
        )
        # print('cruise', V_cruise/ftm)
        
        if V_cruise <= 0:
            return None

        # NOTE: this breaks for certain props with diameters < 10, they all return 60 lbs MGTOW (not possible)
        # Use fast MGTOW calculation
        MGTOW = MGTOWinnerfunc_fast(self, 30.0) # returns in LBS!!!
        
        # check for if V_cruise is < Vstall given by mgtow
        Vlof = np.sqrt((2 * (MGTOW*lbfN)) / (self.rho * self.Sw * self.CLmax))*1.15
        if V_cruise < Vlof or MGTOW <= 0.0:
            return None
    
        return (propname, V_cruise/ftm, MGTOW, P_static, I_static, RPM_static)
        
    except Exception as e:
        print(e)
        return None

def plotMultiMotorMGTOWPareto(self, verbose = False, AllPareto = False):
    """
    Fast version of MotorsMGTOWParetoPlot using optimized MGTOW calculation.
    Expected to be ~10x faster than the original GEKKO-based version.
    """
    # ... existing code for propeller list processing ...
    if self.proplist is None:
        with open("Databases/PropDatabase/proplist.txt", "r") as data_file:
            all_props = data_file.readlines()
        
        self.proplist = []
        for item in all_props:
            if item == 'filelist.txt\n' or 'EP(CD)' in item or 'E(CD)' in item:
                continue
            
            # Filter by diameter
            try:
                test = item.replace('PER3_', '').split('x')[0]
                if float(test) < self.lb or float(test) > self.ub:
                    continue
                
                propname_corrected = item.replace('PER3_', '').replace('.dat', '').replace('\n', '')
                self.proplist.append(propname_corrected)
            except:
                continue
            
    df = pd.read_csv('Databases/Motors.csv')

    if self.motorlist == 'all':
        self.motorlist = list(df['Name'])
        self.nmots = np.ones(len(self.motorlist))
    
    print(f'\nAnalyzing {len(self.motorlist)} motors, {len(self.proplist)} propellers, ({len(self.motorlist)*len(self.proplist)} point designs) for MGTOW Pareto front (using {multiprocessing.cpu_count()-1} processers)')
    
    # to avoid color repetition
    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0, 1, len(self.motorlist))) # for allpareto = True
    
    motor_color = {}
    colorindex = -1 # this mechanism of switching colors for each motor is very convoluted, but it works
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)
    # ax.set_prop_cycle('color', colors)
    
    cumulative_pareto_data = []
    cumulative_prop_data = []
    
    # Parallel processing setup
    for i, motor in enumerate(self.motorlist):
        self.Motor(motor, self.nmots[i]) # initialize motor parameters using the PointDesign function
        
        print(f'\nRunning {self.nmots[i]:.0f} {self.motor_manufacturer} {motor}; total list {(i)/len(self.motorlist)*100:.0f}% complete')
        
        # Use multiprocessing for parallel execution
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
            # Partial function for parallel processing
            process_func = partial(process_MGTOW_cruise, self = self, motor = motor)
            results = list(tqdm(pool.imap(process_func, self.proplist)))

        prop_data = [r for r in results if r is not None]
        
        if not prop_data:
            print('No valid propeller data found!')
            continue
            
        # ... rest of the plotting code remains the same ...
        # Categorize propellers by violations
        pareto_data = []
        current_violations = []
        power_violations = []
        
        # Sort by cruise speed for Pareto front identification
        prop_data.sort(key=lambda x: x[1], reverse=True)
                
        max_MGTOW = float(0.0)
        for prop in prop_data:
            propname, cruise_speed, MGTOW, power, current, RPM = prop
            # Check violations
            if current > self.Ilimit:
                current_violations.append(prop)
            elif power > self.Pmax:
                power_violations.append(prop)
            elif MGTOW > max_MGTOW:
                # This prop is Pareto optimal (better MGTOW!)
                pareto_data.append(prop)
                max_MGTOW = MGTOW
        
        # Sort Pareto front by cruise speed for plotting
        pareto_data.sort(key=lambda x: x[1])
        cumulative_prop_data.append([pareto_data, motor, self.nmots[i]])
                
        # Highlight violation points
        if i == 0:
            if current_violations:
                current_cruise = [p[1] for p in current_violations]
                current_takeoff = [p[2] for p in current_violations]
                ax.scatter(current_cruise, current_takeoff, s=30, color='red', 
                           label=f'I > {self.Ilimit} A', zorder=2, marker='x')
            
            if power_violations:
                power_cruise = [p[1] for p in power_violations]
                power_takeoff = [p[2] for p in power_violations]
                ax.scatter(power_cruise, power_takeoff, s=30, color='orange', 
                           label='Power limited', zorder=2, marker='^')
                
            # Plot all valid props as light points
            all_cruise = [p[1] for p in prop_data]
            all_takeoff = [p[2] for p in prop_data]
            ax.scatter(all_cruise, all_takeoff, alpha=1.0, s=20, 
                       label='All Props', color = 'lightgray', zorder=1)
        else:
            if current_violations:
                current_cruise = [p[1] for p in current_violations]
                current_takeoff = [p[2] for p in current_violations]
                ax.scatter(current_cruise, current_takeoff, s=30, color='red', 
                           zorder=2, marker='x')
            
            if power_violations:
                power_cruise = [p[1] for p in power_violations]
                power_takeoff = [p[2] for p in power_violations]
                ax.scatter(power_cruise, power_takeoff, s=30, color='orange', 
                           zorder=2, marker='^')
            
            # Plot all valid props as light points
            all_cruise = [p[1] for p in prop_data]
            all_takeoff = [p[2] for p in prop_data]
            ax.scatter(all_cruise, all_takeoff, alpha=1.0, s=20, 
                       color = 'lightgray', zorder=1)
            
        # Old version plotted the pareto front for every motor, new version plots it for ALL combined
        # Plot Pareto front 
        if pareto_data and AllPareto:
            colorindex += 1
            pareto_cruise = [p[1] for p in pareto_data]
            pareto_takeoff = [p[2] for p in pareto_data]
            ax.plot(pareto_cruise, pareto_takeoff, 'o-', linewidth=3, color=colors[colorindex], markersize=10, 
                    label=f'{self.nmots[i]:.0f} {self.motor_manufacturer} {motor}', zorder=4)
            
            # Annotate Pareto points
            for propname, cruise_speed, MGTOW, power, current, RPM in pareto_data:
                # name = propname.replace('PER3_', '').replace('.dat', '')
                ax.annotate(propname, (cruise_speed, MGTOW), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, ha='left', va='bottom', weight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', edgecolor = colors[colorindex],
                                     facecolor = 'white', alpha=0.6))
                    
            print(f'\nPareto Front Results ({len(pareto_data)} propellers):')
            print(f'{"Propeller":<20} {"Cruise Speed (fps)":<18} {"MGTOW (lbs)":<18}')
            print('-' * 60)
        
            for propname, cruise_speed, MGTOW, power, current, RPM in pareto_data:
                name = propname.replace('PER3_', '').replace('.dat', '')
                print(f'{name:<20} {cruise_speed:<18.2f} {MGTOW:<18.0f}')
        
            print('\nViolation Summary:')
            print(f'Current violations (I > {self.Ilimit}A): {len(current_violations)}')
            print(f'Power violations (P > {self.Pmax}W): {len(power_violations)}')
    
    print('\nData gathering complete: plotting!')
    
    cumulative_prop_data.sort(key=lambda x: x[0][0][1]) # sorting prop_data by cruise velocity, but the data's grouped so that full sorting is kinda impossible
    # cumulative pareto_data gathering
    if AllPareto == False:
        ################# DO THIS BETTER IN THE FUTURE ####################
        # this part is a lot harder bc the data can't be nicely ordered when preserving the motor part (maybe zip it differently?)
        # lowk a class might help, add a class that has prop data
        max_cruise = float(0.0)
        max_MGTOW = float(0.0)
        for data in cumulative_prop_data: # [[[(pareto_data 1), (pareto data 2)], motorname, nmot], [[[(propdata 1), (propdata2)], motorname, nmot]]]
            #data is [[(prop 1), (prop2)], motorname, nmot]
            motorname_specific = data[1]
            nmot_specific = data[2]
            for propdata in data[0]: 
                propname, cruise_speed, MGTOW, power, current, RPM = propdata
                # Check violations
                if MGTOW > max_MGTOW or cruise_speed > max_cruise:
                    # This prop is Pareto optimal (better MGTOW!)
                    cumulative_pareto_data.append([propdata, motorname_specific, nmot_specific])
                    max_MGTOW = MGTOW
                    max_cruise = cruise_speed
        
        cumulative_pareto_data.sort(key = lambda x: x[0][1], reverse=True) # re-sorts by cruise speed, but accurately this time because the data is packaged better
        # prune cumulative_pareto_data at the end ig??? To get rid of the ones that really aren't good. Feels VERY inefficient tho
        max_MGTOW = float(0.0)
        final_pareto_data = []
        for i, [propdata, motorname, nmot] in enumerate(cumulative_pareto_data):
            propname, cruise_speed, MGTOW, power, current, RPM = propdata
            if MGTOW > max_MGTOW:
                final_pareto_data.append([propdata, motorname, nmot])
                max_MGTOW = MGTOW
                
        colors = colormap(np.linspace(0, 1, len(final_pareto_data)))
        # gather overall data
        pareto_cruise = [p[0][1] for p in final_pareto_data]
        pareto_takeoff = [p[0][2] for p in final_pareto_data]
        
        # plots all pareto data lines in the same color
        ax.plot(pareto_cruise, pareto_takeoff, '-', color='red', linewidth=2.5, zorder=4)
        
        # then I want to plot the points in the specific colors (and annotate them correctly!)
        for propdata, motorname, nmot in final_pareto_data:
            # only change point color and label when the motor is lacking from the existing labels 
            if f'{nmot:.0f} {motorname}' not in ax.get_legend_handles_labels()[1]:
                colorindex += 1
                motor_color[f'{nmot:.0f} {motorname}'] = colors[colorindex]
                ax.plot(propdata[1], propdata[2], 'o-', color = colors[colorindex], 
                        label =  f'{nmot:.0f} {motorname}', markersize = 10, zorder = 5)
            else:
                ax.plot(propdata[1], propdata[2], 'o-', color = motor_color[f'{nmot:.0f} {motorname}'], 
                        markersize = 10, zorder = 5)
            
            ax.annotate(propdata[0], (propdata[1], propdata[2]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left', va='bottom', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor = 'white', alpha=0.6), 
                       zorder = 6)
            
        print(f'\nPareto Front Results ({len(final_pareto_data)} propellers):')
        print(f'{"Propeller":<20} {"Cruise Speed (fps)":<18} {"MGTOW (lbs)":<18} {"Motor":<18}')
        print('-' * 75)

        for [propname, cruise_speed, MGTOW, power, current, RPM], motorname, nmot in final_pareto_data:
            print(f'{propname:<20} {cruise_speed:<18.2f} {MGTOW:<18.0f} {nmot:.0f} {motorname:<18}')
    
    ax.set_xlabel('Cruise Speed (fps)', fontsize=12)
    ax.set_ylabel('Maximum Takeoff Weight (lbs)', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    plt.title(f'Pareto Fronts: Cruise Speed vs MGTOW for Max Ground Roll = {self.xloflimit} ft\n'
              f'{self.ns}S {self.CB} mAh battery; 100% Battery; Imax = {self.Ilimit} A \n'
              f'Sw = {self.Sw/ftm/ftm:.2f} ft², CL_max = {self.CLmax}, CLto = {self.CLto}, CD0 = {self.CD0}, e = {self.e}, AR = {self.AR}', 
              fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return pareto_data    


#%% modelcalcs for mission simulations (where you use a cumulative Itot to get the State Of Charge)
@njit(fastmath=True)
def ModelCalcsExternalSOC(velocity, SOC, rpm_list, numba_prop_data, CB, 
                          ns, Rint, KV, Rm, nmot, I0, ds):
    '''
    Optimized Numba-compatible version solving for T, P, Itot, RPM.
    Implements bisection for root finding on Itot, with fallback to alternate formulation.
    
    since SOC was calculated externally based on the integral of the current used in the mission,
    SOC, Vsoc are no longer variables and the system reduces to:
        Vin = ns*Vsoc - Itot*Rint*ns
        RPM = KV*Vin - ((KV*Rm)/nmot)*Itot
        Q = nmot*TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot = Q*(KV * pi/30) + I0 
    with Vin, Itot, RPM, Q as variables!
    '''
    Vsoc = 3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3 # fixed
        
    def residual_primary(Itot):
        Vin = ns * Vsoc - Itot * Rint * ns
        if Vin < 0:
            return 1e10
        RPM = KV * Vin - (KV * Rm / nmot) * Itot
        if RPM < 0 or RPM > rpm_list[-1]:
            return 1e10
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot_calc = Q * (KV * np.pi / 30) + I0
        return Itot - Itot_calc

    # Bisection for Itot (primary formulation)
    low = 0.0
    high = 400.0  # Adjust as needed
    tol = 1e-3
    max_iter = 100
    Itot = 100 #initial guess
    for _ in range(max_iter):
        mid = (low + high) / 2
        res_mid = residual_primary(mid)
        if abs(res_mid) < tol:
            Itot = mid
            break
        if res_mid > 0:
            high = mid
        else:
            low = mid
    else:
        # If not converged, try alternate or return zeros
        Itot = -1.0
        
    if Itot > 0 and Itot > I0 * nmot:
        # Compute values for primary case
        Vin = ns * Vsoc - Itot * Rint * ns
        RPM = KV * Vin - (KV * Rm / nmot) * Itot
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        T = nmot * ThrustNumba(RPM, velocity, rpm_list, numba_prop_data)
        P = (Itot / nmot) * Vin
        if Q <= 0 or T <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return T, P, Itot, RPM, Q
    
    # Alternate formulation (when Itot <= I0 * nmot or primary failed)
    def residual_alt(RPM):
        if RPM < 0 or RPM > rpm_list[-1]:
            return 1e10
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot_alt = Q * (KV * np.pi / 30) + I0
        Vin = ns * Vsoc - Itot_alt * Rint * ns
        Imot = Itot_alt / nmot
        RPM_new = KV * (Vin - Imot * Rm)
        return RPM_new - RPM

    # Bisection for RPM in alternate
    low = rpm_list[0]
    high = rpm_list[-1]
    RPM = -1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        res_mid = residual_alt(mid)
        if abs(res_mid) < tol:
            RPM = mid
            break
        if res_mid > 0:
            high = mid
        else:
            low = mid
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    if RPM > 0:
        Q = nmot * TorqueNumba(RPM, velocity, rpm_list, numba_prop_data)
        Itot = Q * (KV * np.pi / 30) + I0
        Vin = ns * Vsoc - Itot * Rint * ns
        T = nmot * ThrustNumba(RPM, velocity, rpm_list, numba_prop_data)
        P = (Itot / nmot) * Vin
        if Itot < 0 or T <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return T, P, Itot, RPM, Q

    return 0.0, 0.0, 0.0, 0.0, 0.0
