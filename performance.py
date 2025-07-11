# -*- coding: utf-8 -*-
"""

File dedicated to performance functions

@author: Sammy Nassau
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid, cumulative_trapezoid 
import scipy.integrate as sp 
import matplotlib.pyplot as plt
from tqdm import tqdm

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
class MissionSim:
    def __init__(self):
        pass
    

