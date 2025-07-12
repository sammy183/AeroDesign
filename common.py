# -*- coding: utf-8 -*-
"""
10:51 PM 7/4/2025

Object Oriented Version of Prop Functions

PointDesign is for more computationally intense approaches to analyze interesting designs

Defined by the following variables:
Battery:        
    ds      discharge (% at end)                <--- user defined; 0.85 typically, could go higher and incur LiPo damage
    batt_name (determines database values)      <--- user defined
    
    ns      number of battery cells in series   <--- database value
    CB      battery capacity                    <--- database value
    Rint    internal battery resistance (Ohm)   <--- database value
    
Motor:
    nmot    number of motors                    <--- user defined
    motor_name (determines database values)     <--- user defined, string matching database naming
    
    kv      motor const (RPM/Volts)             <--- database value
    I0      no load current (A)                 <--- database value
    Rm      motor resistance (Ohm)              <--- database value
    Pmax    max continuous motor power (W)      <--- database value

Propeller:
    propname                                    <--- user defined, links to APC props database
    
Basic aerodynamics:
    CD      3D drag coefficient                 <--- user defined
    Sw      wing area (m^2)                     <--- user defined (for now, soon to be integrated with the geometry)
    rho     air density (kg/m^3)                <--- user defined (soon defined by atmospheric condition (inspired by OpPoint))

(SOON: integrate UIUC data and J/CT/CP/CQ coefficient based formulations!)
(SOON: link PROPID or QPROP)
(SOON: batt_W + *batt_Imax, *V_I0 + mag poles (?))


    
DesignStudy is intended to sweep databases to identify most promising candidate selections (add wing optimization, etc)

@author: NASSAS
"""

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from itertools import cycle
import multiprocessing
from functools import partial
from scipy.interpolate import interp1d
import scipy.optimize as opt
from gekko import GEKKO
import inspect

# other modules in the package
import propulsions
import performance

lbfN = 4.44822
ftm = 0.3048


class PointDesign:
    def __init__(self):
        print('Point Design Initiated!')#\nPlease define a battery, motor, prop, and aerodynamic parameters')
        self.battery = False
        self.motor = False
        self.prop = False
        self.parameters = False
    
    def Battery(self, batt_name, discharge):
        '''
        batt_name will be manufacturer_#cellS_capacity
        i.e. Gaoneng_8S_3300, MaxAmps_12S_2000
        '''
        self.batt_name = batt_name
        self.ds = discharge
        df = pd.read_csv('Databases/Batteries.csv')
        batt_data = df.loc[df['Name'] == self.batt_name]
        self.ns = batt_data['Cell Count'].values[0]
        self.CB = batt_data['Capacity (mAh)'].values[0]
        self.Rint = batt_data['Resistance (Ohm)'].values[0]
        self.battery = True
        # soon add weight, max continuous current, etc
        
    def Motor(self, motor_name, nmot):
        '''
        Motor name will be the designation without the manufacturer name
        i.e. C-4120/30 for the Cobra C-4120/30 motors
        Follow the nomenclature from eCalc for consistency 
        (AND IMPLEMENT A SEARCH THAT MAKES THE NAME CASE INSENSITIVE)
        
        NOTE: make sure to save the csv as a UTF-8 deliminated file (and perhaps change to excel workbook soon)
        '''
        self.motor_name = motor_name
        self.nmot = nmot
        df = pd.read_csv('Databases/Motors.csv')
        try:
            motor_data = df.loc[df['Name'] == motor_name]
            self.motor_manufacturer = motor_data['Manufacturer'].values[0]
            self.KV = motor_data['KV'].values[0]
            self.I0 = motor_data['I0 (A)'].values[0]
            self.V_I0 = motor_data['V_I0 (V)'].values[0]
            self.Rm = motor_data['Rm (Ohm)'].values[0]
            self.Pmax = motor_data['Pmax (W)'].values[0]
            self.motor = True
        except:
            raise KeyError('Motor name not recognized, please call .MotorOptions()')
        # soon add V_I0
    
    def Prop(self, prop_name):
        '''
        prop_name is a string with the form 16x10E, 12x12 (no PER3_ or .dat for useability'''
        self.prop_name = prop_name
        self.PROP_DATA, self.NUMBA_PROP_DATA = propulsions.parse_propeller_data(prop_name)
        self.RPM_VALUES, self.THRUST_POLYS, self.TORQUE_POLYS, self.V_DOMAINS  = propulsions.initialize_RPM_polynomials(self.PROP_DATA)
        self.rpm_list = np.array(self.RPM_VALUES)
        self.prop = True
        
    def Parameters(self, rho, MGTOW, Sw, AR, CLmax, CLto, CL, CD, CD0, e, b, h0, taper, takeoff_surface):
        '''
        rho     air density (kg/m^3)
        MGTOW   aircraft gross weight in N
        Sw      wing area (m^2)
        AR      aspect ratio
        CLmax   maximum lift coefficient 
        CL      lift coefficient at 0 deg AoA
        CD      drag coefficient at 0 deg AoA
        CLto    CL at 0 deg aoa when taking off (with flaps, etc)
        CD0     zero lift drag
        e       oswald eff
        
        soon move rho somewhere else when integrating with altitude
        soon move Sw to geom definition category
        where to put mufric for takeoff analysis?
        '''
        self.CD = CD
        self.Sw = Sw
        self.rho = rho
        self.CLmax = CLmax
        self.CL = CL #CL at 0 deg AoA
        self.CLto = CLto
        self.CD0 = CD0 
        self.e = e
        self.AR = AR
        self.MGTOW = MGTOW
        Vstall = np.sqrt((2*self.MGTOW)/(self.rho*self.Sw*self.CLmax))
        self.Vlof = 1.15 * Vstall  # Safety factor of 1.15 (raymer says at least 1.1)
        
        self.b = b          # wingspan (m)
        self.h0 = h0        # height from FRL to ground (m) when on the ground
        self.taper = taper  # taper ratio where 1.0 is rectangle!
        
        if takeoff_surface == 'dry concrete':
            self.mufric = 0.04
        elif takeoff_surface == 'wet concrete':
            self.mufric = 0.05
        elif takeoff_surface == 'icy concrete':
            self.mufric = 0.02
        elif takeoff_surface == 'hard turf':
            self.mufric = 0.05
        elif takeoff_surface == 'firm dirt':
            self.mufric = 0.04
        elif takeoff_surface == 'soft turf':
            self.mufric = 0.07
        elif takeoff_surface == 'wet grass':
            self.mufric = 0.08
        else:
            print('\nTakeoff surface not recognized\nOptions are: dry concrete, wet concrete, icy concrete\n\t\t\thard turf, firm dirt, soft turf, wet grass')
        self.parameters = True
                
    def ViewSetup(self):
        print(f'Battery Specs: {self.batt_name}, {self.ns}S, {self.CB} mAh, {self.Rint} Ohm\n'
              f'Motor Specs:   {self.nmot:.0f} {self.motor_name}, {self.KV} KV, I0 = {self.I0} A, {self.Rm} Ohm, Max Power = {self.Pmax} W\n'
              f'Aerodynamics:  CL = {self.CL}, CD = {self.CD}, CLto = {self.CLto}, CLmax = {self.CLmax}, CD0 = {self.CD0}, e = {self.e}\n'
              f'Geometry:      Sw = {self.Sw:.4f} m^2, AR = {self.AR} \n'
              f'Extras (WIP):  rho = {self.rho} kg/m^3')
    
    def MotorOptions(self):
        df = pd.read_csv('Databases/Motors.csv')
        print('\nMotor Options:')
        print(df[['Manufacturer', 'Name', 'KV', 'Pmax (W)']])
        
    def BatteryOptions(self):
        df = pd.read_csv('Databases/Batteries.csv')
        print('\nBattery Options:')
        print(df[['Name', 'Cell Count', 'Capacity (mAh)']])
    
    def PropellerOptions(self):
        with open('Databases/PropDatabase/proplist.txt', 'r') as f:
            data_content = f.read()
        propellers = data_content.replace('PER3_', '')
        propellers = propellers.replace('.dat', '').split('\n')
        print('\nPropeller Options:')
        for a,b,c,d, e, f in zip(propellers[::6],propellers[1::6],propellers[2::6], propellers[3::6], propellers[4::6], propellers[5::6]):
            print(f'{a:14} {b:14} {c:14} {d:14} {e:14} {f:14}')
            
    def CheckVariables(self, proptest = True, motortest = True):
        if self.battery == False:
            print('Please add a battery to PointDesign')
            return False
        elif self.motor == False and motortest:
            print('Please add a motor to PointDesign')
            return False
        elif self.prop == False and proptest:
            print('Please add a propeller to PointDesign')
            return False
        elif self.parameters == False:
            print('Please define PointDesign parameters')
            return False
        else:
            return True

    # Propulsion Functions
    def Runtimes(self, n, verbose = False, showthrust = True):
        '''
        TO BE OPTIMIZED LATER WITH SCIPY CONFIG OR MULTIPROCESSING
        
        n is number of velocities considered (essentially more --> greater accuracy, less --> faster execution)
        '''
        if self.CheckVariables():
            propulsions.PlotRuntimes(self, n, verbose = False, showthrust = showthrust)       
    
    def PointDesignData(self, n, Ilimit = np.inf, grade = 15):
        '''
        inputs n (iter), Ilimit for current limits, grade (contours), 
        plots Thrust, RPM, Power per motor, and Current as functions of cruise velocity and battery runtime

        n is number of indices, Ilimit is the current limit in A
        
        Multiprocessing is significantly slower than the normal version 
        (for n = 300, it takes 689s instead of 149s (but I kept it in the propulsions module bc I'm proud of the work)
        '''
        if self.CheckVariables():
            propulsions.GetPointDesignData(self, n)
            propulsions.plotMosaic(self, Ilimit = Ilimit, grade = grade)
            
            
    def ThrustCruisePareto(self, proplist=None, lb=0.0, ub=1000.0, 
                        Ilimit = 100, verbose=False, AnnotateAll = False):
        '''
        Plots the pareto front of static thrust vs cruise speed
        for all (or selected) propellers 
        
        multiprocessing
        '''
        if self.CheckVariables(proptest=False):
            self.Ilimit = Ilimit
            self.lb = lb
            self.ub = ub
            self.AnnotateAll = AnnotateAll
            self.proplist = proplist
            propulsions.PlotTCPareto_mp(self, verbose = verbose)
        
    def TakeoffCruisePareto(self, proplist = None, lb = 0.0, ub = 1000.0,
                            Ilimit = 100, xloflimit = 150,
                            verbose = False, AnnotateAll = False):
        '''
        xloflimit in ft, Ilimit in A, mufric as takeoff friction coef
        
        no multiprocessing (yet)
        '''
        if self.CheckVariables(proptest = False):
            self.Ilimit = Ilimit
            self.lb = lb
            self.ub = ub
            self.AnnotateAll = AnnotateAll
            self.proplist = proplist
            self.xloflimit = xloflimit
            propulsions.plotTakeoffParetoFrontNumba(self, verbose = verbose)
        
    def MGTOWCruisePareto(self, motorlist = None, nmots = None, proplist = None, 
                          lb = 0.0, ub = 1000.0, Ilimit = 100, xloflimit = 150, 
                          AnnotateAll = False, SkipInvalid = False, 
                          AllPareto = False):
        '''
        xloflimit in ft, Ilimit in A, mufric as takeoff friction coef
        
        can pass a list of motors and the corresponding number of them to test, or pass none
        to test all with 1 motor
        
        FUTURE: replace mufric with 'wet grass' or 'concrete', etc
        '''
        if self.CheckVariables(proptest=False, motortest=False):
            self.Ilimit = Ilimit
            self.lb = lb 
            self.ub = ub
            self.xloflimit = xloflimit
            self.AnnotateAll = AnnotateAll
            self.SkipInvalid = SkipInvalid
            
            # if motorlist is undefined and a motor is given, 
            #       run the preset motor and nmot
            # if motorlist is undefined and no motors are given:
            #       run all motors as singles
            # if motorlist is defined but number of motors is not:
            #       run all motors as singles and note
            # if motorlist is defined and nmot is defined:
            #       run as given
            if motorlist == None and self.CheckVariables(proptest = False):
                self.motorlist = [self.motor_name]
                self.nmots = [self.nmot]
            elif motorlist == None or motorlist == 'All' or motorlist == 'all':
                self.motorlist = 'all'
                self.nmots = nmots
            elif motorlist != None and nmots == None:
                self.motorlist = motorlist 
                self.nmots = np.ones(len(self.motorlist))
                print('\nNumber of motors undefined; running all motors as single')
            else:
                self.motorlist = motorlist
                self.nmots = nmots
            
            self.proplist = proplist
            propulsions.plotMultiMotorMGTOWPareto(self, verbose = False, AllPareto = AllPareto)

                
    def testVmax(self):
        print(propulsions.VmaxLean(self, 0.0, t_in = 0.0, Tlimit = True))
        
    def testMGTOW(self):
        self.xloflimit = 60
        self.Ilimit = 105
        print(propulsions.MGTOWinnerfunc_fast(self, 30.0))

        
    # performance functions
    def DetailedTakeoff(self, plot = True):
        '''
        b is wingspan in m
        h0 is height from fuselage centerline to ground (before takeoff)
        taper is the taper ratio'''
        if self.CheckVariables():
            performance.SimulateTakeoff(self, plot = plot, results = True)
    
    def PrepMissionSim(self, CDtoPreR, CDtoPostR, CLtoPreR, CLtoPostR):
        print('\nMission Simulation Initialized!')
        
        # acceleration, velocity, position, time
        self.a_track = []
        self.V_track = []
        self.x_track = []
        self.t_track = []
        
        # forces
        self.T_track = []
        self.D_track = []
        
        # propulsion data
        self.SOC_track = []
        self.Itot_track = []
        self.RPM_track = []
        self.Q_track = []
        self.P_track = []
        
        self.datatrack = [self.a_track, 
                        self.V_track,
                        self.x_track,
                        self.t_track,
                        self.T_track,
                        self.D_track,
                        self.SOC_track,
                        self.Itot_track,
                        self.RPM_track,
                        self.Q_track,
                        self.P_track]
        
        self.mass = self.MGTOW/9.81 
        
        self.CDtoPreR = CDtoPreR # drag coef for takeoff pre-rotation (so 0 aoa with high lift devices)
        self.CDtoPostR = CDtoPostR # drag coef for takeoff post-rotation (so ~10 deg aoa with high lift devices)
        self.CLtoPreR = CLtoPreR
        self.CLtoPostR = CLtoPostR
        
        #velocity for stall and lof are already implemented!
    
    def updatedata(self, newdata):
        '''
        All data from the takeoff, cruise, climb, turn functions is in the format:
            avalues, Vvalues, xvalues, tvalues, Tvalues, Dvalues, SOCvalues, Itotvalues, RPMvalues,
            Qvalues, Pvalues (note: P is per motor!)'''
        for i in range(len(self.datatrack)):
            self.datatrack[i].append(newdata[i])
            
    def DBF_Lap(self):
        '''
        
        Typical DBF lap: 
        500 ft straight, 180 deg turn, 500 ft straight (or less or more!), 360 deg turn, 500 ft straight, 180 deg turn, 500 ft straight 
        (and you're back over the start line!)
         
        First lap is slightly different bc you don't have the initial 500 ft straight, you go straight from the climb to the turn typically
        
        '''
        
        texpect = 100
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)
        
        # 180 turn (IMPLEMENT)
        
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)
        
        # 360 turn
        
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)

        # 180 turn
        
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)
        
        # lap end
    
    def DBF_ThreeLaps(self, aoa_rotation = 10, climb_altitude = 100*ftm):
        #### NOTE: FIND A BETTER WAY TO ANTICIPATE THE END OF THE SEGMENT (so it'll work beyond dbf!!!)
        
        print(f'\nSimulating Takeoff with {aoa_rotation} deg of rotation') # print statements here bc f-strings haven't been implemented in numba yet!
        texpect = 50
        self.segment_index = 0
        data = performance.Takeoff(aoa_rotation, texpect, self.h0, self.taper, self.AR, self.b, self.MGTOW, self.rho, self.Sw, 
                       self.CDtoPreR, self.CLtoPreR, self.CDtoPostR, self.CLtoPostR, self.CLmax,  
                       self.mass, self.mufric, self.Vlof, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, 
                       self.Rint, self.KV, self.Rm, self.nmot, self.I0, self.ds, n = 1000, plot = False, results = False)
        self.updatedata(data)
        
        # print(f'Simulating Climb to {climb_altitude/ftm} ft') 
        ########## IMPLEMENT THIS FUNCTION ################
        # (NOTE: then you have to go forward 500 ft combined horizontally so maybe add a super short cruise segment)
        
        
        # print('Simulating 180 deg turn')
        ########## IMPLEMENT THIS FUNCTION ################
        
        texpect = 50
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)

        # print('Simulating 360 deg turn')
        ########## IMPLEMENT THIS FUNCTION ################

        texpect = 50
        segment_distance = 500*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)
        
        # print('Simulating 180 deg turn')
        
        
        texpect = 50
        segment_distance = 1000*ftm # 200 ft converted to m
        self.segment_index += 1
        print(f'Simulating {segment_distance/ftm:.1f} ft linear')
        data = performance.Cruise(segment_distance, self.V_track[self.segment_index-1][-1], self.t_track[self.segment_index-1][-1], self.SOC_track[self.segment_index-1][-1], 
                                  self.x_track[self.segment_index-1][-1], self.CL, self.CD, self.Sw, self.rho, self.mass, self.ds, self.rpm_list, self.NUMBA_PROP_DATA, self.CB, self.ns, self.Rint, 
                                  self.KV, self.Rm, self.nmot, self.I0, tend = texpect, n = 1000)
        self.updatedata(data)
        
        
        # for now this is just 4 cruise segments
        self.DBF_Lap()


        labels = ['Takeoff', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 1000 ft', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 500 ft', 'Cruise 500 ft']

        plt.figure(figsize=(6,4), dpi = 1000)
        for i in range(self.segment_index+1):
            plt.plot(self.t_track[i], self.V_track[i]/ftm, label = labels[i])
        plt.legend()
        plt.title('velocity plot (fps)')
        plt.show() 
        
        
        # plt.figure(figsize=(6,4), dpi = 1000)
        # plt.plot(self.t_track[0], self.D_track[0]/lbfN, label = 'Drag, Takeoff')
        # plt.plot(self.t_track[1], self.D_track[1]/lbfN, label = 'Drag, Cruise')
        # plt.plot(self.t_track[0], self.T_track[0]/lbfN, label = 'Thrust, Takeoff')
        # plt.plot(self.t_track[1], self.T_track[1]/lbfN, label = 'Thrust, Cruise')
        # plt.legend()
        # plt.title('Forces')
        # plt.show() 
        
        plt.figure(figsize=(6,4), dpi = 1000)
        for i in range(self.segment_index+1):
            plt.plot(self.t_track[i], self.SOC_track[i]*100, label = labels[i])
        plt.legend()
        plt.title('SOC')
        plt.show() 
        
        plt.figure(figsize=(6,4), dpi = 1000)
        for i in range(self.segment_index+1):
            plt.plot(self.t_track[i], self.Itot_track[i], label = labels[i])
        plt.legend()
        plt.title('Current (Itot)')
        plt.show() 
        
        plt.figure(figsize=(6,4), dpi = 1000)
        for i in range(self.segment_index+1):
            plt.plot(self.t_track[i], self.x_track[i]/ftm, label = labels[i])
        plt.legend()
        plt.title('Distance (ft)')
        plt.show() 
        
    
            
        
        