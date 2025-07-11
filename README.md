# AeroDesign

## Documentation for AeroDesign usage<h3>
by Sammy Nassau, RPI 2025


### Objects:

#### PointDesign:
Initialize characteristics with:
- .Battery(batt_name, discharge frac)
- .Motor(motor_name, number of motors)
- .Prop(prop_name)
- .Parameters(
  - rho (density),
  - MGTOW (Max Gross Takeoff Weight in N),
  - Sw (wing area in m2),
  - AR (aspect ratio),
  - CLmax (max lift coeff.),
  - CLto (CL during takeoff),
  - CL (CL at 0 deg AoA), CD (Drag coeff.),
  - CD0 (Zero lift/parasitic drag coeff.),
  - e (Oswald efficiency)
)

Main Functions:
- .Runtimes(iters, verbose=False)
- .PointDesignData(iters, Ilimit=100, multiprocess=False)
- .ThrustCruisePareto(
  - proplist=[Optional, if not put it will run all APC props within bounds],
  - lb = lowerbound_int,
  - ub = upperbount_int,
  - Ilimit = Current_limit in A,
  - verbose=False,
  - AnnotateAll = False
)
- .TakeoffCruisePareto(
  - proplist = [optional],
  - lb = 0.0,
  - ub = 1000.0,
  - Ilimit = 100,
  - xloflimit = 150 (takeoff limit in ft!),
  - mufric = 0.04 (takeoff friction (see Raymer for values, 0.08 on wet grass, typically ~0.04 for concrete)),
  - verbose = False,
  - AnnotateAll = False
)
- .MGTOWCruisePareto(
  - motorlist = [optional, otherwise will do all in database],
  - nmots = [optional, each one matches with a motor in motorlist, if none provided it defaults to 1 motor each],
  - proplist = [opitonal],
  - lb = 0.0,
  - ub = 1000.0,
  - Ilimit = 100,
  - xloflimit = 150,
  - mufric = 0.04,
  - AnnotateAll = False,
  - SkipInvalid = False (this skips all invalid options bc the plot can get very cluttered!)
)
    
To view your current setup and potential propulsion options:
- .ViewSetup()
- .MotorOptions()
- .BatteryOptions()
- .PropellerOptions()

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
- the .Runtimes() function tends to raise RankWarning, doesn't affect the result, but it's annoying
- This documentation sucks balls, I'll work on it moving forward
