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
        self.prop = True
        
    def Parameters(self, rho, MGTOW, Sw, AR, CLmax, CLto, CL, CD, CD0, e):
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
                            Ilimit = 100, xloflimit = 150, mufric = 0.04,
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
            self.mufric = mufric
            propulsions.plotTakeoffParetoFrontNumba(self, verbose = verbose)
        
    def MGTOWCruisePareto(self, motorlist = None, nmots = None, proplist = None, 
                          lb = 0.0, ub = 1000.0, Ilimit = 100, xloflimit = 150, 
                          mufric = 0.04, AnnotateAll = False, SkipInvalid = False, 
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
            self.mufric = mufric
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
        self.mufric = 0.04
        self.xloflimit = 60
        self.Ilimit = 105
        print(propulsions.MGTOWinnerfunc_fast(self, 30.0))

        
    # performance functions
    def DetailedTakeoff(self, b, h0, taper, mufric = 0.04, plot = True):
        '''
        b is wingspan in m
        h0 is height from fuselage centerline to ground (before takeoff)
        taper is the taper ratio'''
        if self.CheckVariables():
            self.h0 = h0 
            self.b = b
            self.taper = taper
            self.mufric = mufric
            performance.SimulateTakeoff(self, plot = plot, results = True)
            
        
    
            
        
        