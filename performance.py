# -*- coding: utf-8 -*-
"""

File dedicated to performance functions

@author: Sammy Nassau
"""

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid, cumulative_trapezoid, quad
import scipy.integrate as sp 
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit, njit
# import numba

import propulsions

lbfN = 4.44822
ftm = 0.3048

#%% Takeoff simulation using initial value problem methods
def SimulateTakeoff(self, texpect = 20, results = True, plot = False):
    '''
    h0 = distance of fuselage centerline from ground (meters)
    b = wingspan (m)
    t expect is the expected takeoff time in s
    
    Currently no accounting for takeoff rotation!!
    
    Using: https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1080&context=mae_facpub
    For more accurate ground effect predictions for wings with LINEAR TAPER
    
    '''
    print('\nSimulating Ground Roll...')
    h = self.h0
    # to account for ground effect (phillips + hunsaker correction) at SMALL aoa
    delta_D = 1 - 0.157*(self.taper**0.775 - 0.373)*(self.AR**0.417 - 1.27)
    CDige_oge = 1 - delta_D*(np.e**(-4.74*(h/self.b)**0.814)) - ((h/self.b)**2)*(np.e**(-3.88*(h/self.b)**0.758))
    
    # at SMALL aerodynamic angles of attack
    delta_L = 1 - 2.25*(self.taper**0.00273 - 0.997)*(self.AR**0.717 + 13.6)
    CLige_oge = 1 + delta_L*(288*((h/self.b)**0.787)*(np.e**(-9.14*((h/self.b)**0.327))))/(self.AR**0.882)
    
    # Pre-rotation!
    M = self.MGTOW/9.81 # mass in kg
    
    def T_Dtakeoff(t, V):
        T, _, _, _ = propulsions.ModelCalcs(self, V, t) 
        D = 0.5*self.rho*(V**2)*self.Sw*self.CD*CDige_oge
        L = 0.5*self.rho*(V**2)*self.Sw*self.CLto*CLige_oge
        F = self.mufric*(self.MGTOW-L)
        return((T - D - F)/M)
    
    V0 = [0]
    tspan= [0, texpect]
    
    sol = solve_ivp(T_Dtakeoff, tspan, V0, method = 'RK45', dense_output = True, t_eval=np.linspace(0, texpect, 100000)) #might need to change this if time becomes an issue
    ts = sol.t
    Vs = sol.y[0]
    
    diffs = np.abs(Vs - (4/5)*self.Vlof) # assume rotation at 4/5 Vlof (BIG BIG APPROX)
    rotate = np.argsort(diffs)[:1][0]
    # trotate = ts[rotate]
    xrotate = trapezoid(Vs[:rotate], x=ts[:rotate])
    
    t1 = ts[:rotate]
    V1 = Vs[:rotate]
    d1 = cumulative_trapezoid(V1, x=t1)
    d1 = np.insert(d1, 0, 0.0)
    a1 = Vs[1:rotate]/ts[1:rotate]
    
    # ROTATION phase
    CLnew = self.CLmax-0.2 #HUGEEEE assumption, this would be changing continuously
    
    # updated using the high aoa formulations from Phillips and Hunsaker
    Beta_D = 1 + 0.0361*(CLnew**1.21)/((self.AR**1.19)*((h/self.b)**1.51))
    CDige_oge = (1 - delta_D*(np.e**(-4.74*(h/self.b)**0.814)) - ((h/self.b)**2)*(np.e**(-3.88*(h/self.b)**0.758)))*Beta_D
    Beta_L = 1 + 0.269*(CLnew**1.45)
    CLige_oge = (1 + (delta_L*(288*(h/self.b)**0.787)*(np.e**(-9.14*((h/self.b)**0.327))))/(self.AR**0.882))/Beta_L
    
    # assume that the takeoff occurs at 10 deg aoa, and adjust the thrust vector accordingly
    aoa = 10 #deg
    def T_Dtakeoff(t, V):
        T, _, _, _ = propulsions.ModelCalcs(self, V, t)
        T *= np.cos(aoa*(np.pi/180))
        D = 0.5*self.rho*(V**2)*self.Sw*self.CD*CDige_oge
        L = 0.5*self.rho*(V**2)*self.Sw*self.CLto*CLige_oge + T*np.sin(aoa*(np.pi/180))
        F = self.mufric*(self.MGTOW-L)
        return((T - D - F)/M)
    
    V0 = [V1[-1]]
    tspan= [t1[-1], texpect]
    
    sol = solve_ivp(T_Dtakeoff, tspan, V0, method = 'RK45', dense_output = True, t_eval=np.linspace(t1[-1], texpect, 100000)) #might need to change this if time becomes an issue
    ts = sol.t
    Vs = sol.y[0]
    
    diffs = np.abs(Vs - self.Vlof) # assume rotation at 2/3 Vlof (BIG BIG APPROX)
    liftoff = np.argsort(diffs)[:1][0]
    tlof = ts[liftoff]
    
    t2 = ts[:liftoff]
    V2 = Vs[:liftoff]
    d2 = cumulative_trapezoid(V2, x=t2) + d1[-1]
    d2 = np.insert(d2, 0, d1[-1])
    a2 = Vs[:liftoff]/ts[:liftoff]
    xlof = trapezoid(Vs[:liftoff], x=ts[:liftoff])
    
    t_tot = np.append(t1, t2)
    V_tot = np.append(V1, V2)
    d_tot = np.append(d1, d2)
    a_tot = np.append(a1, a2)
    xlof_tot = xrotate + xlof
    tlof_tot = tlof
    if plot:
        fig, ax1 = plt.subplots(figsize = (6, 4), dpi = 1000)
        
        p1 = ax1.plot(t_tot, V_tot/ftm, color = 'black')
        ax1.set_ylabel('Velocity (ft/s)')
        
        ax2 = ax1.twinx()
        p2 = ax2.plot(t_tot, d_tot/ftm, color = 'red')
        ax2.set_ylabel('Distance (ft)')
        
        ax3 = ax1.twinx()
        p3 = ax3.plot(t_tot[1:], a_tot/ftm, color = 'blue')
        ax3.set_ylabel(r'Acceleration (ft/s$^2$)')
        ax3.spines['right'].set_position(('outward', 60))

        plt.xlabel('Time (s)')
        
        ax1.plot([t2[0], t2[0]], ax1.get_ylim(), '--', color = 'orange', label = 'Rotation')
        
        ax1.yaxis.label.set_color(p1[0].get_color())
        ax2.yaxis.label.set_color(p2[0].get_color())
        ax3.yaxis.label.set_color(p3[0].get_color())

        ax1.grid()
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()
        ax1.legend(loc = 'upper center')
        plt.annotate('Rotation', (1, 1))
        ax1.set_xlabel('Time (s)')
        plt.title(f'Ground Roll\nVlof = {self.Vlof/ftm:.1f} fps, Vrotate = {V1[-1]/ftm:.1f} ft/s\nDistance = {xlof_tot/ftm:.1f} ft, Time = {tlof_tot:.2f} s')
        plt.show()
    
    if results:
        print('\nGround Roll Calculations:\n'
              f'{"Distance":10} {xlof_tot/ftm:.4f} ft\n'
              f'{"Time":10} {tlof_tot:.4f} s')
    return(tlof_tot, xlof_tot)

#%% next is mission simulation, use an object for each lap to simplify

# def DragCalcs(self):
#     '''
#     V in m/s
#     Assumes Re > Recutoff, so not great for foam planes'''
#     #via raymer eqn
#     e = 1.78*(1-0.045*(AR**0.68))-0.64

#     #via wind tunnel testing! 
#     CDfstatic = 0.01890 #for everything except the wing!
    
#     #horner eqn for Form Factor
#     FFwing = 1+(2*t)+(60*(t**4))
    
#     Swet_wing = (Swet_ratio*(c/ftm))*ftm*ftm #to get Swet_wing in m!! holy fuck it's been so wrong for a while....
    
#     Re = (rho*V*c)/muair
    
#     #using np where for Re to allow for vectorization of CD
#     #Schlichting Compressible (only works for Re > 10^5 ofc!!)
#     CfW = np.where(Re < 1e5, 0.0070467, 0.455/((np.log10(Re))**2.58))
    
#     CDf_W = (FFwing*Swet_wing*CfW)/Sw
    
#     CDf = CDf_W + CDfstatic 
#     CDi = (CL**2)/(np.pi*AR*e)
#     CD = CDf + CDi
#     return(CD)



# how to best structure this???? Do I create a new object or just add methods for the PointDesign one!
# class MissionSim(PointDesign):
#     '''For DBF mission simulations: covers number of laps, etc'''
#     def __init__(self):
#         pass
    
#     def ThreeLapSim(self, time_limit):
#         # first simulate takeoff
#         tlof_tot, xlof_tot = SimulateTakeoff(self, texpect = 20, results = False, plot = False)
        
        # then simultate 3 laps but record the t vs V data for each segment so the plot can record them correctly
        # ideally you could also get Thrust and battery charge across the entire time as well though.....
        
        
        # I should reformat this problem
        
        
#%% Problem formulation:

# aircraft forces for 4 dof simulation (not full 6DOF):
    # T, D, L, W
# angles 
    # climb (theta), angle of attack (aoa), bank angle (phi)
# dependancies:
    # velocity, mission time, past history of current (Itot) to determine current state of charge (SOC)
    # have ModelCalcsNumba(velocity, t, rpm_list (fixed from start), numba_prop_data, CB, ns, Rint, KV, Rm, nmot, I0, ds) which returns T, P, Itot, RPM if solvable
    # D can be determined using a predetermined CL at 10 deg aoa and CL at 0 deg aoa plus other aircraft characteristics
    # L via the aforementioned CLs
    # W is fixed throughout (but could depend on some angle stuff)

# inputs: 
    # CL0, CL10, CLmax, CLtakeoff, CD0lift, Sw, MGTOW, AR
    # rho, muair (atmospheric)
    # propeller, battery, motor (already implemented)

# of these the hardest part is accurately providing CL, CD, CLtakeoff at the beginning! For initial point designs you don't often know them that well

# g = 9.81

# texpect = 100

# def testlinearsim(self, plot = True):
#     M = self.MGTOW/g # mass in kg
#     # this works perfectly!! But the stepping is really inefficient tbh, could try to numbafy it?
    
#     # # alternative with manual stepping
#     n = 10000
#     ts = np.linspace(0, 500, n)
#     dt = ts[2]-ts[1]
    
#     V = np.zeros(n+1)
#     V[0] = 15.0 #initial V (m/s)
#     a = np.zeros(n)
#     T = np.zeros(n)
#     P = np.zeros(n)
#     Itot = np.zeros(n)
#     RPM = np.zeros(n)
#     D = np.zeros(n)
#     SOC = np.zeros(n+1)
#     SOC[0] = 1.0 #initial state of charge
    
#     # propulsions.ModelCalcs(self, V[i], t)
#     # what if I want to account for changing Itot???
#     for i, t in enumerate(tqdm(ts)):
#         if SOC[i] < (1-self.ds):
#             endindex = i
#             print(f'\nSOC exceeds discharge at {t:.2f}s of runtime')
#             break
#         T[i], P[i], Itot[i], RPM[i] = propulsions.ModelCalcsExternalSOC(V[i], t, SOC[i], self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds) #max usable ds is 0.9999 (1.0 breaks the curve!)
#         # determining velocity change        
        
#         if i > 1000 and i < 2000:
#             CD0 = 0.32
#         else:
#             CD0 = self.CD0
            
#         D[i] = 0.5*self.rho*(V[i]**2)*self.Sw*CD0
#         a[i] = (T[i]-D[i])/M
#         V[i+1] = V[i] + a[i]*dt
        
#         # determining acceleration change
#         SOC[i+1] = (self.CB*3.6 - trapezoid(Itot[:i], x=ts[:i]))/(self.CB*3.6)
    
#     ts = ts[:endindex]
#     V = V[:endindex]
#     a = a[:endindex]
#     RPM = RPM[:endindex]
#     P = P[:endindex]
#     Itot = Itot[:endindex]
#     T = T[:endindex]
#     D = D[:endindex]
#     SOC = SOC[:endindex]
    
#     plt.figure()
#     plt.plot(ts, V, label='velocity')
#     plt.plot(ts, a, label='accel')
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.plot(ts, RPM)
#     plt.title('rpm')
#     plt.show()
    
#     plt.figure()
#     plt.plot(ts, P)
#     plt.title('power')
#     plt.show()
    
#     plt.figure()
#     plt.plot(ts, Itot)
#     plt.title('current')
#     plt.show()
    
#     plt.figure()
#     plt.plot(ts, T, label='Thrust')
#     plt.plot(ts, D, label='Drag')
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.plot(ts, SOC*100)
#     plt.title('State of Charge (%)')
#     plt.show()
    
#     # t = 10
#     # thing = propulsions.ModelCalcsNumba(25.0, t, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, self.KV, self.Rm, self.nmot, self.I0, 1.0) #ds = 1.0 --> SOC can be anything > 0 
#     # print(thing)
    
#     # thing2 = propulsions.ModelCalcs(self, 25.0, t)
#     # print(thing2)
    
#     # t2 = ts[:liftoff]
#     # V2 = Vs[:liftoff]
#     # d2 = cumulative_trapezoid(V2, x=t2) + d1[-1]
#     # d2 = np.insert(d2, 0, d1[-1])
#     # a2 = Vs[:liftoff]/ts[:liftoff]
#     # xlof = trapezoid(Vs[:liftoff], x=ts[:liftoff])

#%% The real functions start here!
@njit(fastmath=True) #DOESN'T WORK BC OF THE TRAPEZOID USAGE!!!, NOW IT DOES!!
def Takeoff(aoa, texpect, h0, taper, AR, b, MGTOW, rho, Sw, CDtoPreR, CLtoPreR, CDtoPostR, CLtoPostR, CLmax,  
            mass, mufric, Vlof, rpm_list, NUMBA_PROP_DATA, CB, ns, Rint, KV, Rm, nmot, I0, ds, n = 1000, plot = False, results = False):
    '''
    
    Several approximations in this code:
        - the phillips + hunsaker correction is valid for ground affect impact for very small UAVs
        - ***rotation occurs instantaneously at 4/5 of liftoff speed (itself 1.15*Vstall)
        - g = 9.81 m/s^2
        
    '''    
    h = h0
    # to account for ground effect (phillips + hunsaker correction) at SMALL aoa
    delta_D = 1 - 0.157*(taper**0.775 - 0.373)*(AR**0.417 - 1.27)
    CDige_oge = 1 - delta_D*(np.e**(-4.74*(h/b)**0.814)) - ((h/b)**2)*(np.e**(-3.88*(h/b)**0.758))
    
    # at SMALL aerodynamic angles of attack
    delta_L = 1 - 2.25*(taper**0.00273 - 0.997)*(AR**0.717 + 13.6)
    CLige_oge = 1 + delta_L*(288*((h/b)**0.787)*(np.e**(-9.14*((h/b)**0.327))))/(AR**0.882)
    
    
    # timestepping from mission start to rotate velocity assumption
    ts = np.linspace(0.0, texpect, n) # start the mission at 0.0 s
    dt = ts[2]-ts[1]
    
    x = np.zeros(n+1)
    V = np.zeros(n+1)
    SOC = np.zeros(n+1)
    V[0] = 0.0 #initial V (m/s)
    SOC[0] = 1.0 #initial state of charge

    a = np.zeros(n)
    T = np.zeros(n)
    P = np.zeros(n)
    Itot = np.zeros(n)
    Q = np.zeros(n)
    RPM = np.zeros(n)
    D = np.zeros(n)
    L = np.zeros(n)
    
    for i, t in enumerate(ts):
        if SOC[i] < (1-ds):
            endindex = i
            break
        elif V[i] > (4/5)*Vlof: # assume rotation at 4/5 Vlof (BIG BIG APPROX)
            endindex = i
            break
        
        T[i], P[i], Itot[i], RPM[i], Q[i] = propulsions.ModelCalcsExternalSOC(V[i], SOC[i], rpm_list, NUMBA_PROP_DATA, CB, ns, Rint, KV, Rm, nmot, I0, ds)         
        D[i] = 0.5*rho*(V[i]**2)*Sw*CDtoPreR*CDige_oge
        L[i] = 0.5*rho*(V[i]**2)*Sw*CLtoPreR*CLige_oge
        F = mufric*(MGTOW-L[i])
        
        a[i] = (T[i] - D[i] - F)/mass
        
        # PROBLEM: this is derivative based
        V[i+1] = V[i] + a[i]*dt
        x[i+1] = x[i] + V[i]*dt + 0.5*a[i]*(dt**2)
        
        # This is integral based though?
        # SOC[i+1] = (CB*3.6 - trapezoid(Itot[:i], x=ts[:i]))/(CB*3.6)
        SOC[i+1] = SOC[i] - (Itot[i]*dt)/(CB*3.6)
    
    # trim arrays
    a = a[:endindex]
    V = V[:endindex]
    x = x[:endindex]
    ts = ts[:endindex]
    T = T[:endindex]
    D = D[:endindex]
    SOC = SOC[:endindex]
    Itot = Itot[:endindex]
    RPM = RPM[:endindex]
    P = P[:endindex]
    Q = Q[:endindex]    
    
    # now simulate post rotation (please don't mention my awful naming conventions)
    
    # updated using the high aoa formulations from Phillips and Hunsaker
    Beta_D = 1 + 0.0361*(CLtoPostR**1.21)/((AR**1.19)*((h/b)**1.51))
    CDige_oge = (1 - delta_D*(np.e**(-4.74*(h/b)**0.814)) - ((h/b)**2)*(np.e**(-3.88*(h/b)**0.758)))*Beta_D
    Beta_L = 1 + 0.269*(CLtoPostR**1.45)
    CLige_oge = (1 + (delta_L*(288*(h/b)**0.787)*(np.e**(-9.14*((h/b)**0.327))))/(AR**0.882))/Beta_L

    ts2 = np.linspace(ts[-1], texpect, n) # start the mission at 0.0 s
    dt2 = ts[2]-ts[1]
    
    x2 = np.zeros(n+1)
    V2 = np.zeros(n+1)
    SOC2 = np.zeros(n+1)

    a2 = np.zeros(n)
    T2 = np.zeros(n)
    P2 = np.zeros(n)
    Itot2 = np.zeros(n)
    Q2 = np.zeros(n)
    RPM2 = np.zeros(n)
    D2 = np.zeros(n)
    L2 = np.zeros(n)
    
    V2[0] = V[-1]      # initial V (m/s)
    SOC2[0] = SOC[-1]  # initial state of charge
    x2[0] = x[-1]
    Itot2[0] = Itot[-1] # also need to do this so SOC doesn't reset to 100!    
        
    for i, t in enumerate(ts2):
        if SOC2[i] < (1-ds):
            endindex = i
            break
        elif V2[i] > Vlof: # assume rotation at 4/5 Vlof (BIG BIG APPROX)
            endindex = i
            break
        
        T2[i], P2[i], Itot2[i], RPM2[i], Q2[i] = propulsions.ModelCalcsExternalSOC(V2[i], SOC2[i], rpm_list, NUMBA_PROP_DATA, 
                                                 CB, ns, Rint, KV, Rm, nmot, I0, ds)         
        D2[i] = 0.5*rho*(V2[i]**2)*Sw*CDtoPreR*CDige_oge
        L2[i] = 0.5*rho*(V2[i]**2)*Sw*CLtoPreR*CLige_oge
        F = mufric*(MGTOW-L2[i])
        
        a2[i] = (T2[i] - D2[i] - F)/mass
        
        V2[i+1] = V2[i] + a2[i]*dt2
        x2[i+1] = x2[i] + V2[i]*dt2 + 0.5*a2[i]*(dt2**2)
        SOC2[i+1] = SOC2[i] - (Itot2[i]*dt2)/(CB*3.6)
    
    # trim arrays
    a2 = a2[:endindex]
    V2 = V2[:endindex]
    x2 = x2[:endindex]
    ts2 = ts2[:endindex]
    T2 = T2[:endindex]
    D2 = D2[:endindex]
    SOC2 = SOC2[:endindex]
    Itot2 = Itot2[:endindex]
    RPM2 = RPM2[:endindex]
    P2 = P2[:endindex]
    Q2 = Q2[:endindex]
    
    
    # combine (this is so badly done, there MUST be a better way)
    a = np.append(a, a2)
    V = np.append(V, V2)
    x = np.append(x, x2)
    ts = np.append(ts, ts2)
    T = np.append(T, T2)
    D = np.append(D, D2)
    SOC = np.append(SOC, SOC2)
    Itot = np.append(Itot, Itot2)
    RPM = np.append(RPM, RPM2)
    P = np.append(P, P2)
    Q = np.append(Q, Q2)
    
    return(a, V, x, ts, T, D, SOC, Itot, RPM, P, Q)

#%% Cruise
@njit(fastmath=True)
def Cruise(segment_distance, V_initial, t_initial, SOC_initial, x_initial, CL0, CD0, 
           Sw, rho, mass, ds, rpm_list, NUMBA_PROP_DATA, CB, ns, Rint, KV, Rm, nmot, I0, tend = 500, n = 1000):
    '''
    Segment distance in m
    Initial velocity in m/s
    initial time in s
    initial SOC in %/100 
    CL0, CD0 aerodynamic coefs at cruise conditions (usually 0 aoa)
    rho = air density kg/m3
    mass = plane mass in kg
    
    max usable ds is 0.9999 (1.0 breaks the curve!)
    '''
    
    ts = np.linspace(t_initial, tend, n)
    dt = ts[2]-ts[1]
    
    x = np.zeros(n+1)
    V = np.zeros(n+1)
    a = np.zeros(n)
    T = np.zeros(n)
    P = np.zeros(n)
    Itot = np.zeros(n)
    Q = np.zeros(n)
    RPM = np.zeros(n)
    D = np.zeros(n)
    
    SOC = np.zeros(n+1)
    SOC[0] = SOC_initial #initial state of charge
    V[0] = V_initial #initial V (m/s)
    x[0] = x_initial
    
    # propulsions.ModelCalcs(self, V[i], t)
    # what if I want to account for changing Itot???
    # realistically I should aim to use RK45 or something 
    for i, t in enumerate(ts):
        if SOC[i] < (1-ds):
            endindex = i
            break
        elif x[i]-x_initial > segment_distance:
            endindex = i
            break
        
        T[i], P[i], Itot[i], RPM[i], Q[i] = propulsions.ModelCalcsExternalSOC(V[i], SOC[i], rpm_list, NUMBA_PROP_DATA, 
                                                                              CB, ns, Rint, KV, Rm, nmot, I0, ds) 
        D[i] = 0.5*rho*(V[i]**2)*Sw*CD0
        a[i] = (T[i]-D[i])/mass
        
        # PROBLEM: this is derivative method
        V[i+1] = V[i] + a[i]*dt
        x[i+1] = x[i] + V[i]*dt + 0.5*a[i]*(dt**2)
        
        # determining acceleration change
        SOC[i+1] = SOC[i] - (Itot[i]*dt)/(CB*3.6)
    
    a = a[:endindex]
    V = V[:endindex]
    x = x[:endindex]
    ts = ts[:endindex]
    T = T[:endindex]
    D = D[:endindex]
    SOC = SOC[:endindex]
    Itot = Itot[:endindex]
    RPM = RPM[:endindex]
    P = P[:endindex]
    Q = Q[:endindex]
    
    return(a, V, x, ts, T, D, SOC, Itot, RPM, P, Q)

#%% Climb (essentially cruise but with a theta modifier)



#%% Turn (with bank angle implementation)