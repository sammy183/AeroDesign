# -*- coding: utf-8 -*-
"""
Necessary python libraries installed:
    numpy, pandas, scipy, matplotlib, tqdm, gekko, multiprocessing, functools, itertools
    

Documentation:
    
PointDesign object:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Initialize characteristics with:
    .Battery(batt_name, discharge frac)
    .Motor(motor_name, number of motors)
    .Prop(prop_name)
    .Parameters(rho (density), MGTOW (Max Gross Takeoff Weight in N), 
                Sw (wing area in m2), AR (aspect ratio), 
                CLmax (max lift coeff.), CLto (CL during takeoff), 
                CL (CL at 0 deg AoA), CD (Drag coeff.), 
                CD0 (Zero lift/parasitic drag coeff.), e (Oswald efficiency))

Functions:
    .Runtimes(iters, verbose=False)
    .PointDesignData(iters, Ilimit=100)
    .ThrustCruisePareto(proplist=[Optional, if not put it will run all APC props within bounds],
                     lb = lowerbound_int, ub = upperbount_int, Ilimit = Current_limit in A, 
                     verbose=False, AnnotateAll = False)
    .TakeoffCruisePareto(proplist = [optional], lb = 0.0, ub = 1000.0,
                        Ilimit = 100, xloflimit = 150 (takeoff limit in ft!), mufric = 0.04 (takeoff friction (see Raymer for values, 0.08 on wet grass, typically ~0.04 for concrete)),
                        verbose = False, AnnotateAll = False)
    .MGTOWCruisePareto(motorlist = [optional, otherwise will do all in database], nmots = [optional, each one matches with a motor in motorlist, if none provided it defaults to 1 motor each],
                       proplist = [opitonal], lb = 0.0, ub = 1000.0, Ilimit = 100, xloflimit = 150, 
                       mufric = 0.04, AnnotateAll = False, SkipInvalid = False (this skips all invalid options bc the plot can get very cluttered!))
    
To view your current setup and potential propulsion options:
    .ViewSetup()
    .MotorOptions()
    .BatteryOptions()
    .PropellerOptions()

TO BE IMPLEMENTED:
    - Mission Model DBF Course velocity plot + more developed takeoff predictions
    - Energy Maneuverability plots for point designs
    - Obj/Constraint based DesignSweep objects for wing + propulsion optimization (wingspan, chord, sweep, taper, twist (maybe))
    - Prop stall + better predictions via UIUC experimental data
    - ESC/Wire resistance 
    - Scaling I0 by voltage 
    - Database expansion for more options!
    - Ecalc interface (or maybe not bc that's lowk illegal)
Known problems:
    - for the MGTOW pareto front plot, propellers with diameters < 10 are giving enormously incorrect information 
      (such as MGTOW = 60 is possible on a 6x4E propeller with xlof < 100 ft)
    - 

@author: Sammy Nassau, Documentation Lead for class of 2025. 7/6/2025
        Enjoy and I hope this helps :)
"""

import time
import multiprocessing

# my modules
import common as ad

lbfN = 4.44822
ftm = 0.3048

#%% Basic setup and usage demonstration based on Juggernaut characteristics
start = time.perf_counter()
rho = 1.12
Sw = 9.0*ftm*ftm #9.0 ft2 then converted to M
CLmax = 1.6
CL = 0.44
CLto = 0.9
CD = 0.055 #0.045
CD0 = 0.03
e = 0.8
AR = 4.0 

b = 6.0*ftm 
h0 = 1.778 #m
taper = 1.0

# MGTOW = 35.0*lbfN #30 lbf then converted to N 
MGTOW = 15*lbfN

# takeoff coefs
CDtoPreR = 0.07     # CD at aoa when on landing gear + high lift devices
CDtoPostR = 0.3     # CD at rotation aoa + high lift devices
CLtoPreR = 0.9      # CL at aoa when on landing gear + high lift devices
CLtoPostR = 1.7     # CD at rotation aoa + high lift devices

CLturn = 1.7221         # with flaps at ~ 10 deg aoa
CDturn = 0.324996513016 # with flaps at ~ 10 deg aoa

nmax = 3 #g

# theoretical without flaps
# CL10 = 1.1 
# CD10 = 0.09
# for the classic Cobra + 16x10E combo, the plot runtimes looks about accurate but the model calcs seems 
# to be overpredicting the results; examine why! (also no measure of ESC/wire resistance....)

#%% PointDesign setup 
if __name__ == '__main__':
    # call objects here so multiprocessing can work!
    multiprocessing.freeze_support()

# to initialize a setup:
    Design = ad.PointDesign() 
    Design.Battery('Gaoneng_8S_3300', 0.85)
    Design.Motor('C-4130/20', 2)
    Design.Prop('12x12E')
    Design.Parameters(rho, MGTOW, Sw, AR, CLmax, 
                      CLto, CL, CD, CD0, e, 
                      b, h0, taper, 'dry concrete', nmax)

# to check setup and options:
    # Design.ViewSetup()
    # Design.MotorOptions()
    # Design.BatteryOptions()
    # Design.PropellerOptions()

#%% Analysis functions (uncomment to run!):
    
# Runtimes plots the runtime for a specified design vs the freestream velocity. 
# The input number, n determines the number of iterations (more n --> nicer plot and more accurate but increased computational time)
    # Design.Runtimes(100, verbose=False)
    
# PointDesignData plots the thrust, RPM, current, and electric power for freestream velocity vs battery runtime
    # Design.PointDesignData(200, Ilimit = 100, grade = 15)

# ThrustCruisePareto plots the Pareto front of Static Thrust vs Cruise Velocity for a given PointDesign 
# and either a sepecified propeller/list of propellers or all propellers in the APC database. 
# For specified propellers use proplist = ['propname', 'propname', 'etc']
    # Design.ThrustCruisePareto(proplist = None, Ilimit = 50, AnnotateAll = False)
    
# TakeoffCruisePareto plots the Pareto front of takeoff distance vs cruise velocity for a given PointDesign.
    # Design.TakeoffCruisePareto(proplist = None, Ilimit = 105, xloflimit = 65)

    
# MGTOWCruisePareto plots the pareto front of MGTOW vs cruise velocity for a given PointDesign
# This function can be applied to multiple motors at once, either by specifiying in motorlist and nmots lists or 
# by setting motorlist, nmots to None and running all motor + propeller combos in the database for one motor
 
# example with specified motors + nmots:
    # s = time.perf_counter()
    # Design.MGTOWCruisePareto(motorlist = ['C-4120/30', 'A-5025-310', 'HKII-4525-370'], nmots = [2, 1, 1], 
    #                          proplist = ['10x55MR','16x10E', '18x12E', '14x14', '12x12', '15x10E', '16x16', '20x10E'],
    #                          # lb = 10.0, ub = 25.0,
    #                          Ilimit = 105, xloflimit = 60,
    #                          SkipInvalid = False, AllPareto = False)
    # e = time.perf_counter()
    # print(f'Time Taken: {e-s:.2f}s')

# example that runs all motors + nmots:
# NOTE: takes around 20s per motor    
    Design.MGTOWCruisePareto(motorlist = 'all', Ilimit = 105, xloflimit = 150,
                             SkipInvalid = True, AllPareto = False)

#%% performance funcs
    # Design.PrepMissionSim(CDtoPreR, CDtoPostR, CLtoPreR, CLtoPostR, 
    #                       CDturn, CLturn) 
    
    # Design.DetailedTakeoff(aoa_rotation = 10, t_expect = 60, plot = True)

    # Design.DBF_ThreeLaps(aoa_rotation = 10, climb_altitude = 100*ftm, climb_angle = 10, plot = False)
    # Design.PlotMission('Velocity')
    # Design.PlotMission('SOC')
    # Design.PlotMission('Current')

    # Design.DBF_MaxLaps(time_limit = 300, aoa_rotation = 10, climb_altitude = 100*ftm, climb_angle = 10)
    # main important quantities
    # Design.PlotMission('Velocity')
    # Design.PlotMission('SOC')
    # Design.PlotMission('Current')
    
    # others useful for diagnostics!
    # Design.PlotMission('Acceleration')
    # Design.PlotMission('Position')
    # Design.PlotMission('Altitude')
    # Design.PlotMission('Thrust')
    # Design.PlotMission('Load Factor')
    # Design.PlotMission('RPM')
    # Design.PlotMission('Torque')
    # Design.PlotMission('Power')

    # format for custom missions:
        # 'Takeoff', aoa_rotation (deg)
        # 'Climb', change in altitude (m), climb/descent angle (deg), horizontal distance limit (m) (set to a huge number to ignore)
        # 'Cruise', distance (m)
        # 'Turn', degrees
    # the turn performance forces the aircraft to at least be at stall speed, then instantaneously turn until reaching the nmax limit
        
    # custom_mission = [('Takeoff', 10),
    #                   ('Climb', 500*ftm, 10, 1e9),
    #                   ('Turn', 180), 
    #                   ('Cruise', 3000*ftm),
    #                   ('Climb', 300*ftm, 10, 1e9),
    #                   ('Cruise', 300*ftm),
    #                   ('Climb', 150*ftm, -5, 1e9),
    #                   ('Cruise', 1500*ftm)]
    
    # ##### FUTURE WORK: integrate high lift device options into Climb/Turn SEGMENTS ######
    # ##### (and potentially deflections?)
    
    # Design.MissionProfile(custom_mission)
    # Design.PlotMission('Velocity')
    # Design.PlotMission('SOC')
    # Design.PlotMission('Altitude')

    # others useful for diagnostics!
    # Design.PlotMission('Current')
    # Design.PlotMission('Acceleration')
    # Design.PlotMission('Thrust')
    # Design.PlotMission('Load Factor')
    # Design.PlotMission('RPM')
    # Design.PlotMission('Torque')
    # Design.PlotMission('Power')
    
    
    # Design.EnergyManeuverability()    
    

