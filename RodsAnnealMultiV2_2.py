from __future__ import print_function
import numpy as np
import math
import csv
import time
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axis3d
import matplotlib.pyplot as plt
import random
#from functools import partial

def DefaultCustomFunc(xVect,**kwargs):
    x = xVect[0]
    y = xVect[1]
    if kwargs["FuncName"] == "A":
        Cost = x**2+y**2
    if kwargs["FuncName"] == "B":
        Cost = x+y
    return Cost

# def Perform(funct, **kwargs):
#     funct(kwargs)

class SA():
    def __init__(self, FuncOption = 'B',CustomFuction=DefaultCustomFunc, NUMITER=1, NUMpoints=6000, DimX=0,DimY=1,**kwargs):
        self.Xs = []
        self.Ys = []
        self.Zs = []
        self.DimX = DimX
        self.DimY = DimY
        self.Xtime = []
        self.Ytemperature = []
        self.YtempLimit = []
        self.YProbGraph = []
        self.YbestOBJ = []
        self.YOBJ = []
        self.ItValue = []
        self.ItNumber = 0
        # System Parameters
        self.FuncOption = FuncOption
        self.CustomFuction = CustomFuction
        self.kwargs = kwargs
        self.FIXseed = False  # establishes random seeds for all random numbers
        self.OBJfunctionRANGE = 1.5# Explores Objetive function from -OBJfunctionRANGE to +OBJfunctionRANGE in x and y directions
        self.DELTAObjFunc = 0.1    # Used to calculate the next guess for x and y:  xn = xbest + random.uniform(-DeltaObjFunc*OBJfunctionRANGE, DeltaObjFunc*OBJfunctionRANGE)
        self.WRITEtoFILE = False  # Exports the result of each Anneal Iteration to .csv file
        self.WRITEtoFILEshort = True  #Exports only the final result of all the Anneal Iterations
        self.GRAPH = 0
        self.GRAPHsurface = 0
        self.GRAPHresolution = 0.01
        self.GRAPHtemperSCHED = 1
        self.GRAPHProb = 1
        self.GRAPHobjValue = 1
        self.GRAPHhistogram = 1
        self.GRAPHcontour = 0
        self.PRINTiter = 0
        self.PRINTSOLUTION = 1
        self.PRINTbatchSOLUTION = True
        self.NUMpoints = NUMpoints  # Number of random points per Annealing simulation
        self.POINTSperTEMPERAT = 1
        self.NUMITER = NUMITER  # Number of times to run the Annealing Function
        self.APPLYcutoff = True
        self.CUTOFFvalue = 8
        self.GAUSSIANsampling = True
        self.STATEspaceNARROWING = True  # Turns space narrowing on and off
        self.STATEspaceNARvalue = 200.0  # Narrows the search space if Best Solution has not chenged in STATEspaceNARvalue trials
        self.DELTASpace = 0.35
        self.TSCHEDULE = 'Kirkpatrick'
        self.T0 = None
        self.RECURSIVEstart = False
        self.STARTdelta = 0.0

    def PickSched(self, SchedType = 'ExpDecay', limit = 4000):
        limit += 1
        if SchedType == 'ExpDecay':
            if self.T0 == None: self.T0 = 50.0
            lam = 0.004    #0.004
            SchedFunc = lambda t: (self.T0 * math.exp(-lam * t) if t < limit else 0)
        elif SchedType == 'Kirkpatrick':
            self.APPLYcutoff = True #Possible to improve furnter with Narrowing=25 DeltaSpace=0.35
            self.CUTOFFvalue = 8  #CutOff of 8 already finds the global minimun
            self.POINTSperTEMPERAT= 4000 #Original Points per Temperature set by Kirpatric was 50000, 4000 will sufice
            if self.T0 == None: self.T0 = 10  #Original To set by Kirpatric was 10
            SchedFunc = lambda t: ((0.9**t)*self.T0) if t < limit else 0
        elif SchedType == 'LogarithmicConvCond':
            if self.T0 == None: self.T0 = 1
            SchedFunc = lambda t: (self.T0/(math.log(1+t)) if t < limit else 0)
        elif SchedType == 'ConvCond-Log2':
            if self.T0 == None: self.T0 = 1
            SchedFunc = lambda t: (self.T0/(1+math.log(1+t)) if t < limit else 0)
        elif SchedType == 'Geometric':
            if self.T0 == None: self.T0 = 100000.0
            SchedFunc = lambda t: (self.T0 * 0.996**t) if t < limit else 0
        elif SchedType == 'LogMultip':
            if self.T0 == None: self.T0 = 5
            Alpha = 1.0001 # Alpha>1
            SchedFunc = lambda t: (self.T0/(1+Alpha*math.log(1+t)) if t < limit else 0)
        elif SchedType == 'LinearMultip':
            if self.T0 == None: self.T0 = 2000.0
            Alpha = 2  # Alpha>0
            SchedFunc = lambda t: (self.T0/(1+Alpha*t) if t < limit else 0)
        elif SchedType == 'QuadraticMultip':
            if self.T0 == None: self.T0 = 4000.0
            Alpha = 1.2  # Alpha>0
            SchedFunc = lambda t: (self.T0/(1+Alpha*t**2) if t < limit else 0)
        elif SchedType == 'LinearInvTime':
             if self.T0 == None: self.T0 = 2000.0
             SchedFunc = lambda t: (self.T0/t if t < limit else 0)
        elif SchedType == 'DivFactor':
             if self.T0 == None: self.T0 = 16000.0
             Mu = 1.004
             SchedFunc = lambda t: (self.T0/(Mu**t) if t < limit else 0)
        self.TSCHEDULE = SchedType
        return SchedFunc

    def Anneal(self, StartC=[0.0,0.0]):
        if self.WRITEtoFILE:
            self.csvADD(['ITER', 't', 'Points', 'x', 'y', 'Objective', 'To', 'Schedule:', self.TSCHEDULE, 'ObjRange:',
                    self.OBJfunctionRANGE,
                    'Narrowing:', self.STATEspaceNARROWING, 'NarrValue:', self.STATEspaceNARvalue,
                    'DeltaSp:', self.DELTASpace, 'CutOff:', self.APPLYcutoff, 'COvalue:', self.CUTOFFvalue, 'Points/T:',
                    self.POINTSperTEMPERAT])
        if self.FIXseed: random.seed(0.55)
        else: random.seed()
        if StartC == []:
            StartC = np.random.uniform(-self.OBJfunctionRANGE, self.OBJfunctionRANGE, 2)
            #StartC = np.zeros(len(StartC))
        Bxbest = StartC
        Bfbest = self.f(Bxbest)
        batchTime = time.time()
        ItValue=[]
        xns = np.zeros(len(StartC))
        for self.ItNumber in range(self.NUMITER):
            print("It:", self.ItNumber + 1)
            print("Starting Coordinate:",StartC)
            Ixbest, Ifbest = self.simulated_annealing(self.PickSched(self.TSCHEDULE, self.NUMpoints), StartC)
            if self.GRAPHhistogram or self.WRITEtoFILEshort or self.GRAPHcontour: ItValue.append(Ifbest)
            if Ifbest < Bfbest: Bxbest, Bfbest = Ixbest, Ifbest
            if self.RECURSIVEstart:
                if self.FIXseed: random.seed(self.ItNumber)
                for nd in range(len(StartC)):
                    xns[nd] = (Bxbest[nd] + random.uniform(-self.STARTdelta, self.STARTdelta))
                    if xns[nd] < (-1 * self.OBJfunctionRANGE): xns[nd] = (-1 * self.OBJfunctionRANGE)
                    if xns[nd] > self.OBJfunctionRANGE: xns[nd] = self.OBJfunctionRANGE
                StartC = np.array(xns)

        BFTime = time.time() - batchTime
        if self.PRINTbatchSOLUTION:
            print("")
            print("Number of Iterations:", self.NUMITER)
            print("Number of Points / Iteration: ", self.NUMpoints)
            print("T Schedule:", self.TSCHEDULE)
            print("Batch Time elapsed: ", BFTime)
            print('Best Batch solution: ' + str(Bxbest))
            print('Best Batch objective: ' + str(Bfbest))

        if self.GRAPHtemperSCHED:
            fig, ax1 = plt.subplots()
            ax1.plot(self.Xtime, self.Ytemperature, 'b.', markersize=2)
            ax1.set_xlabel('Time')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('Temperature', color='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')
            if False:  #if True plots To/ln(t) limit
                ax1.plot(self.Xtime,self.YtempLimit, 'g.', markersize=1)
                ax1.set_ylim([0,self.T0])
            if self.GRAPHProb:
                ax2 = ax1.twinx()
                ax2.plot(self.Xtime, self.YProbGraph, 'ro', markersize=2)
                ax2.set_ylabel('Prob.', color='r')
                ax2.set_ylim([-0.5, 2])
                for tl in ax2.get_yticklabels():
                    tl.set_color('r')

        if self.GRAPHobjValue:
            fig, ax1 = plt.subplots()
            ax1.plot(self.Xtime, self.YOBJ, 'bx', markersize=2)
            ax1.plot(self.Xtime, self.YbestOBJ, 'ro', markersize=3)
            ax1.set_xlabel('Time')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('Function Value')

        if self.GRAPH:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.scatter(self.Xs, self.Ys, self.Zs, c='b', marker='o')
            ax1.legend(['Points Tryed'])
            ax1.set_xlim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
            ax1.set_ylim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

        if self.GRAPHhistogram or self.WRITEtoFILE or self.WRITEtoFILEshort or self.GRAPHcontour:
            ItValue = np.array(ItValue)
            mu = np.mean(ItValue)
            sigma = np.std(ItValue)
            MaxValue = np.max(ItValue)
            MinValue = np.min(ItValue)
            #OverallMax = np.max(self.Zs)
            #OverallMin = np.min(self.Zs)
            #print("Max by Iteration: ",MaxValue,"   Max Achieved: ", OverallMax)
            #print("Min by Iteration: ", MinValue,"   Min Achieved:", OverallMin)
            if self.WRITEtoFILE or self.WRITEtoFILEshort : self.GRAPHhistogram=True

        if self.GRAPHsurface or self.GRAPHcontour:
            # Design variables at mesh points
            i1 = np.arange(-self.OBJfunctionRANGE, self.OBJfunctionRANGE, self.GRAPHresolution)
            i2 = np.arange(-self.OBJfunctionRANGE, self.OBJfunctionRANGE, self.GRAPHresolution)
            x1m, x2m = np.meshgrid(i1, i2)
            fm = np.zeros(x1m.shape)
            for i in range(x1m.shape[0]):
                for j in range(x1m.shape[1]):
                    xVect=[x1m[i][j], x2m[i][j]]
                    fm[i][j] = self.f(xVect)

        if self.GRAPHcontour:
            from matplotlib.widgets import Slider, Button
            # Create a contour plot
            axis_color = 'lightgoldenrodyellow'
            figC = plt.figure()
            axC = figC.add_subplot(111)
            figC.subplots_adjust(bottom=0.21)
            NumLines = 190
            def DrawContour():
                CS = axC.contour(x1m, x2m, fm, int(NumLines_Slider.val), cmap=cm.jet)
                title_string = 'Contour Plot: \nm=' + str(round(mu, 3)) + '  s=' + str(
                    round(sigma, 3)) + '  Min=' + str(round(MinValue, 6)) + \
                               '  Max=' + str(round(MaxValue, 5))
                axC.set_title(title_string, fontsize=13)
                axC.set_xlim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
                axC.set_ylim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
                axC.set_xlabel('X')
                axC.set_ylabel('Y')
                axC.plot(Bxbest[0], Bxbest[1], 'b^', markersize=13)
            NumLines_Slider_ax = figC.add_axes([0.19, 0.05, 0.65, 0.02], axisbg=axis_color)
            NumLines_Slider = Slider(NumLines_Slider_ax, '# Levels', 1, 200, valinit=NumLines, valfmt="%.0f")
            NumPoints_Slider_ax = figC.add_axes([0.19, 0.02, 0.65, 0.02], axisbg=axis_color)
            NumPoints_Slider = Slider(NumPoints_Slider_ax, '# Points', 0, len(self.Xs), valinit=0, valfmt="%.0f")
            DrawContour()
            plt.figtext(.50, 0.09,
                        'Niter=' + str(self.NUMITER) + '  NPts=' + str(self.NUMpoints) + '  Pts/T=' + str(
                            self.POINTSperTEMPERAT) +
                        '  To=' + str(int(self.T0)) + '  Schd: ' + self.TSCHEDULE + '  FRange: ' + str(
                            self.OBJfunctionRANGE) + '  DtaObj: ' + str(self.DELTAObjFunc) + '  RS: ' + str(
                            self.RECURSIVEstart) +
                        '\nStDta: '+str(self.STARTdelta)+ '    Narr: ' + str(self.STATEspaceNARROWING) + '    NarrValue: ' + str(
                            int(self.STATEspaceNARvalue)) +
                        '    DtaSp: ' + str(self.DELTASpace) + '    CutOff: ' + str(
                            self.APPLYcutoff) + '    COValue: ' + str(
                            self.CUTOFFvalue)+ '    Sol: '+str([round(float(Bxbest[self.DimX]),3), round(float(Bxbest[self.DimY]),3)]), fontsize=10, ha='center')
            #axC.plot(self.Xns, self.Yns, 'bx', markersize=4)
            axC.plot(self.Xs, self.Ys, 'ro', markersize=3)
            # Add a button
            Plus_button_ax = figC.add_axes([0.04, 0.05, 0.02, 0.02])
            Plus_button = Button(Plus_button_ax, '+', color=axis_color, hovercolor='0.975')
            def Plus_button_on_clicked(mouse_event):
                NumPoints_Slider.set_val(int(NumPoints_Slider.val)+1)
                #print(NumPoints_Slider.val)
                axC.plot(self.Xs[int(NumPoints_Slider.val)], self.Ys[int(NumPoints_Slider.val)], 'ro', markersize=3)
                #axC.plot(self.Xns[0:int(NumPoints_Slider.val)], self.Yns[0:int(NumPoints_Slider.val)], 'bx', markersize=4)
            Plus_button.on_clicked(Plus_button_on_clicked)

            Minus_button_ax = figC.add_axes([0.04, 0.02, 0.02, 0.02])
            Minus_button = Button(Minus_button_ax, '-', color=axis_color, hovercolor='0.975')
            def Minus_button_on_clicked(mouse_event):
                NumPoints_Slider.set_val(int(NumPoints_Slider.val)-1)
                #print(NumPoints_Slider.val)
                axC.plot(self.Xs[int(NumPoints_Slider.val)], self.Ys[int(NumPoints_Slider.val)], 'ro', markersize=3)
            Minus_button.on_clicked(Minus_button_on_clicked)
            def sliders_on_changed(val):
                axC.clear()
                DrawContour()
                axC.plot(self.Xs[0:int(NumPoints_Slider.val)], self.Ys[0:int(NumPoints_Slider.val)], 'ro', markersize=3)
                #axC.plot(self.Xns[0:int(NumPoints_Slider.val)], self.Yns[0:int(NumPoints_Slider.val)], 'bx', markersize=4)
                plt.draw()
            NumLines_Slider.on_changed(sliders_on_changed)
            NumPoints_Slider.on_changed(sliders_on_changed)

        if self.GRAPHsurface:
            # Design variables at mesh points
            i1 = np.arange(-self.OBJfunctionRANGE, self.OBJfunctionRANGE, self.GRAPHresolution)
            i2 = np.arange(-self.OBJfunctionRANGE, self.OBJfunctionRANGE, self.GRAPHresolution)
            x1m, x2m = np.meshgrid(i1, i2)
            fm = np.zeros(x1m.shape)
            for i in range(x1m.shape[0]):
                for j in range(x1m.shape[1]):
                    xVect=[x1m[i][j], x2m[i][j]]
                    fm[i][j] = self.f(xVect)
            fig = plt.figure()
            ax1 = fig.gca(projection='3d')
            ax1.plot_surface(x1m, x2m, fm, cmap=cm.jet)
            ax1.set_xlabel('X')
            ax1.set_xlim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
            ax1.set_ylim([-self.OBJfunctionRANGE, self.OBJfunctionRANGE])
            ax1.set_ylabel('Y')

        if self.GRAPHhistogram:
            import matplotlib.mlab as mlab
            # the histogram of the data
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            numBins = 50
            n, bins, patches = ax1.hist(ItValue, numBins, color='blue', alpha=0.8)
            plt.grid(True)
            rects = ax1.patches
            # Now make some labels
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax1.text(rect.get_x() + rect.get_width() / 2, 1.01 * height, '%d' % int(height), ha='center',
                             va='bottom', fontsize=8)
            # add a 'best fit' line
            ItValue = np.array(ItValue)

            MaxValue = np.max(ItValue)
            MinValue = np.min(ItValue)
            bincenters = 0.5 * (bins[1:] + bins[:-1])
            SmallestBin = bins[0:2]
            #print(bins[0:2])
            ax2 = ax1.twinx()
            if self.NUMITER > 1:
                mu = np.mean(ItValue)
                sigma = np.std(ItValue)
                NormCurve = mlab.normpdf(bincenters, mu, sigma)
                l = ax2.plot(bincenters, NormCurve, 'r-', linewidth=2)
            title_string = 'Histogram of Best Vals: \nm=' + str(round(mu, 3)) + '  s=' + str(
                round(sigma, 3)) + '  Min=' + str(round(MinValue, 6)) + \
                           '  Max=' + str(round(MaxValue, 5))
            plt.figtext(.5, 0.02,
                        'Niter=' + str(self.NUMITER) + '  NPts=' + str(self.NUMpoints) + '  Pts/T=' + str(self.POINTSperTEMPERAT) +
                        '  To=' + str(int(self.T0)) + '  Schd: ' + self.TSCHEDULE + '  FRange: ' + str(
                            self.OBJfunctionRANGE) + '  DtaObj: ' + str(self.DELTAObjFunc) + '  RS: ' + str(self.RECURSIVEstart) +
                        '\nStDta: '+str(self.STARTdelta) + '    Narr: ' + str(self.STATEspaceNARROWING) + '    NarrValue: ' + str(int(self.STATEspaceNARvalue)) +
                        '    DtaSp: ' + str(self.DELTASpace) + '    CutOff: ' + str(self.APPLYcutoff) + '    COValue: ' + str(
                            self.CUTOFFvalue) + '    BinMax: ' + str(round(bins[1], 6)), fontsize=10, ha='center')
            plt.title(title_string, fontsize=13)
            #ax2 = ax1.twinx()
            #l = ax2.plot(bincenters, NormCurve, 'r-', linewidth=2)
            ax1.set_ylabel('Frequency', color='b')
            ax2.set_ylabel('Fitted Normal Dist.', color='r')
            Nmim=(int(ax1.patches[0].get_height()))
            print(Nmim, SmallestBin)

        if self.WRITEtoFILEshort:
            self.csvADD(
                ['Func','Gauss' ,'Niter', 'Npts', 'Pts/T', 'To', 'Sched', 'Frange', 'DeltaObjF', 'Narr', 'NarrVal', 'DeltaSp',
                 'CutOff', 'COValue', 'Min', 'Nmin', 'Recur.', 'RcDelta', 'Time', 'FixSeed', 'SBin', 'Mu', 'Sigma',
                 'Max', 'Sol','COTemp'])
            self.csvADD([self.FuncOption, self.GAUSSIANsampling, self.NUMITER, self.NUMpoints, self.POINTSperTEMPERAT, self.T0, self.TSCHEDULE,
                         self.OBJfunctionRANGE, self.DELTAObjFunc, self.STATEspaceNARROWING, self.STATEspaceNARvalue,
                         self.DELTASpace, self.APPLYcutoff, self.CUTOFFvalue, Bfbest, Nmim, self.RECURSIVEstart,
                         self.STARTdelta, BFTime, self.FIXseed, SmallestBin, mu, sigma, MaxValue, [Bxbest[0], Bxbest[1]],self.COt])

        if self.WRITEtoFILE:
            self.csvADD(
                ['Process Time:', BFTime, 'Nmim:', Nmim, 'SBin:', SmallestBin, 'Mu:', mu, 'Sigma:', sigma, 'Max:', MaxValue, 'Min:', MinValue])

        if self.GRAPH or self.GRAPHtemperSCHED or self.GRAPHsurface or self.GRAPHhistogram:
            plt.show()
        return [Bxbest, Bfbest]


    def f(self, xVect):     # define objective functions
        x = xVect[self.DimX]
        y = xVect[self.DimY]
        Funct = {'A': 0.7 + x**2 + y**2 - 0.1*math.cos(6.0*3.1415*x) - 0.1*math.cos(6.0*3.1415*y),
                 'B': (math.e**math.sin(50*x) + math.sin(60*math.e**y) + math.sin(70*math.sin(x)) + math.sin(math.sin(80*y)) - math.sin(10*(x + y)) + 0.25*(x**2 + y**2)),
                 'C': x**2+ y**2 + 5,
                 'D': 10000*(math.e**math.sin(50*x) + math.sin(60*math.e**y) + math.sin(70*math.sin(x)) + math.sin(math.sin(80*y)) - math.sin(10*(x + y)) + 0.25*(x**2 + y**2)),
                 'E': (2*x**6-12.2*x**5+21.2*x**4+6.2*x-6.4*x**3-4.7*x**2+y**6-11*y**5+43.3*y**4-10*y-74.8*y**3+56.9*y**2-4.1*x*y-0.1*y**2*x**2+0.4*x*y**2+0.4*x**2*y),
                 'F': (1-x)**2 + 25*(y-x**2)**2 + (math.cos(3*math.pi*x+4*math.pi*y)+1),
                 #'Z': (self.CustomFuction(xVect,**self.kwargs)),
                 'G': (20*math.sin(x*np.pi/2-2*np.pi) + 20*math.sin(y*np.pi/2-2*np.pi)+(x-2*np.pi)**2+(y-2*np.pi)**2)}
        return Funct[self.FuncOption]



    def csvADD (self, line):
        import sys
        if sys.version_info > (3,4):
            with open('SimAnneal.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(line)
        else:
            with open(r'SimAnneal.csv', 'ab') as file:
                writer = csv.writer(file)
                writer.writerow(line)

    def simulated_annealing(self, schedule, StartCoord):
        random.seed()
        xbest = np.array(StartCoord)
        xmin = np.zeros(len(StartCoord),dtype=float)
        xmax = np.zeros(len(StartCoord),dtype=float)
        xn = np.array(StartCoord)
        Gxbest = np.array(xbest)
        fbest = self.f(xbest)
        Gfbest = fbest  # In certain situations we guess a global min. but the temperature is too high and we don't accept it. Gfbest captures this situation.
        t=1; CutOff = 0; StateSpaceCutOff = 0
        #global Xs, Ys, Zs, Xtime, Ytemperature, YProbGraph, YbestOBJ, YOBJ, DELTASpace
        start = time.time()
        while True:
            T = schedule(t)
            for SameTemperature in range(0, self.POINTSperTEMPERAT):
                if T == 0 or (self.APPLYcutoff==True and CutOff == (self.CUTOFFvalue+1)):
                    elapsed = time.time() - start
                    if Gfbest < fbest: xbest, fbest = Gxbest, Gfbest
                    if self.PRINTSOLUTION:
                        # print("T Schedule:", TSCHEDULE)
                        print("Time elapsed: ", elapsed)
                        print('Best solution: ' + str(xbest))
                        print('Best objective: ' + str(fbest))
                        print()
                    if self.WRITEtoFILE:
                        Result = [self.ItNumber + 1, elapsed, t - 1, xbest, fbest, self.T0]
                        self.csvADD(Result)
                    self.COt = t
                    if self.APPLYcutoff: print("Temperature/time cycles (t):", t)
                    return (xbest,fbest)
                self.Xtime.append(t)
                if self.GRAPHtemperSCHED:
                    self.Ytemperature.append(T)
                    self.YtempLimit.append(self.T0/math.log(t+0.1))

                if self.FIXseed: random.seed(t)
                if self.STATEspaceNARROWING and StateSpaceCutOff > self.STATEspaceNARvalue:
                    for nds in range(int(len(StartCoord))):
                        xmin[nds] = -1*(self.DELTASpace * abs(xbest[nds]))
                        xmax[nds] = (self.DELTASpace * abs(xbest[nds]))
                        xn[nds] = (xbest[nds] + random.uniform(xmin[nds], xmax[nds]))
                        if xn[nds] < (-1 * self.OBJfunctionRANGE): xn[nds] = (-1 * self.OBJfunctionRANGE)
                        if xn[nds] > self.OBJfunctionRANGE: xn[nds] = self.OBJfunctionRANGE
                        #xn[nds] = random.uniform(xmin[nds], xmax[nds])
                    if self.PRINTiter: print("Narrowing")
                    if StateSpaceCutOff/self.STATEspaceNARvalue > 2.0:  #If stuck on this local minimum (fbest), explore Gfbest
                        #StateSpaceCutOff = 0  ####
                        if self.PRINTiter: print("Testing against lowest value found...")
                        if Gfbest < fbest:
                            xbest, fbest = np.array(Gxbest), Gfbest
                            if self.PRINTiter: print ("Exploring around lowest value found...")
                else:
                    for nd in range(int(len(StartCoord))):
                        if self.GAUSSIANsampling:
                            xn[nd] = (random.gauss(xbest[nd],math.sqrt(T)))
                            # ###################################################
                            # fn = self.f(xn)
                            # try:
                            #     # Prob = math.exp(-(fn-fbest) / (1.00 * T))    #Metropolis Criterion
                            #     Prob = 1 / (1 + math.exp((fn - fbest) / T))  # Barker Criterion
                            # except OverflowError:
                            #     Prob = 0
                            # if fn < Gfbest: Gxbest[nd], Gfbest = xn[nd], fn  # if fn < Gfbest always accept smallest cost
                            # if fn < fbest or (random.uniform(0.0, 1.0) < Prob):
                            #     xbest[nd], fbest = xn[nd], fn
                            #     CutOff = 0
                            #     StateSpaceCutOff = 0
                            # else:
                            #     StateSpaceCutOff += 1
                            # #############################################
                        else:
                            xn[nd] = (xbest[nd] + random.uniform(-self.DELTAObjFunc * self.OBJfunctionRANGE, self.DELTAObjFunc * self.OBJfunctionRANGE))
                        #xn[nd] = (random.gauss(xbest[nd],math.sqrt(T)))
                        if xn[nd] < -self.OBJfunctionRANGE: xn[nd] = -self.OBJfunctionRANGE
                        if xn[nd] > self.OBJfunctionRANGE: xn[nd] = self.OBJfunctionRANGE
                    if self.PRINTiter: print("Not narrowing")
                fn = self.f(xn)
                try:
                    #Prob = math.exp(-(fn-fbest) / T)    #Metropolis Criterion  k= 1.38064852 *
                    Prob = 1/(1+math.exp((fn - fbest) /(T)))       #Barker Criterion
                except OverflowError:
                    Prob = 0
                if self.GRAPHProb:
                    self.YProbGraph.append(Prob)
                if self.PRINTiter:
                    # Monitor the temperature & cost
                    print("t:", "%.0f" % round(t, 0), "Temp:", "%.10fC" % round(T, 10),
                          #"xbest:", np.around(xbest, 5),
                          "xn:", np.around(xn, 5),
                          #"nd:", np.around(nd, 5),
                          "zn:", "%.5f" % round(fn, 5),
                          "zbest:", "%.5f" % round(fbest, 5),
                          "P:", "%.5f" % round(Prob, 5),
                          #"DeltaSpace:", "%.4f" % round(DELTASpace, 4),
                          "StateCutOff:", "%.0f" % StateSpaceCutOff,
                          "CutOff:", "%.0f" % CutOff)
                if self.GRAPH or self.GRAPHcontour:
                    self.Xs.append(np.array(xn[self.DimX]))
                    self.Ys.append(np.array(xn[self.DimY]))
                    self.Zs.append(fbest)


                if fn < Gfbest: Gxbest, Gfbest = np.array(xn), fn #if fn < Gfbest always accept smallest cost
                if fn < fbest or (random.uniform(0.0, 1.0) < Prob):
                    xbest, fbest = np.array(xn), fn
                    CutOff=0
                    StateSpaceCutOff = 0  ####
                    # if self.GRAPH:  #Tirar comentario se quiser plotar so os pontos aceitos
                    #     self.Xs.append(xbest[self.DimX])
                    #     self.Ys.append(xbest[self.DimY])
                    #     self.Zs.append(fbest)
                else:
                    StateSpaceCutOff += 1
                if self.GRAPHobjValue:
                    self.YbestOBJ.append(fbest)
                    self.YOBJ.append(fn)
            CutOff += 1
            t += 1

#SimAnneal = SA()
#print(SimAnneal.Anneal())
