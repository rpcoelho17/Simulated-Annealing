from __future__ import print_function
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pylab as pl
#from sklearn.datasets import load_digits
import math
import csv
import time
#import matplotlib
from matplotlib import cm
import random
from scipy.io import loadmat
from math import sqrt
import pickle
import RodsAnnealMultiV2_2 as sam

class Neural_Network(object):
    def __init__(self, Lambda=0.000032):  #Controls the trade-off beteween variance and bias
        #high Lambda = little variation, high bias; Low Lambda = high variance, low bias
    #def __init__(self, Lambda=0.0000):
        # Define Hyperparameters
        self.inputLayerSize = 784
        self.outputLayerSize = 10
        self.hiddenLayerSize = 50

        # Weights (parameters)
#        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
#        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.W1 = initializeWeights(n_input, n_hidden)
        self.W2 = initializeWeights(n_hidden, 10)
        # Regularization Parameter:
        self.Lambda = Lambda

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)  # multiplica a input layer pelos pesos
        self.a2 = self.sigmoid(self.z2)  # aplica a activation function
        self.z3 = np.dot(self.a2, self.W2) # multiplica a segunda camada layer pelos pesos W2
        yHat = self.sigmoid(self.z3)  # aplica a segunda activation function e calcula a previsao
        return yHat

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * np.sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        #print(J)
        return float(J)

    # Helper functions for interacting with other methods/classes
    def getParams(self):
        # Get W1 and W2 Rolled into vector:  (.ravel flattens the array)
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single parameter vector: (reverse the flattened array oper above)
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        # print("W1: ",self.W1)
        # print("W2: ", self.W2)
    def setWeights(self):
        self.W1 = initializeWeights(n_input, n_hidden)
        self.W2 = initializeWeights(n_hidden, 10)


def costFunctionWrapper(params,**kwargs):
    n_input, n_hidden, n_class, X, y, lambdaval = kwargs["args"]
    NN.setParams(params)
    cost = NN.costFunction(X, y)
    return cost

def costFunctionWrapper2(Wts,**kwargs):
    NN.setWeights()
    Wts = NN.getParams()
    args = kwargs["args"]
    #global args
    cost, grad = nnObjFunction(Wts, *args)
    return cost

# define objective function
def f(Wts):
    global args
    cost, grad = nnObjFunction(Wts, *args)
    return cost

def csvADD (line):
    with open(r'SimAnneal.csv', 'ab') as file:
        writer = csv.writer(file)
        writer.writerow(list(line))

def PickSched(SchedType = 'Boltzman', limit = 4000):
    global T0
    limit += 1
    if SchedType == 'ExpDecay':
        To = 1
        lam = 0.004
        SchedFunc = lambda t: (To * math.exp(-lam * t) if t < limit else 0)
    elif SchedType == 'Boltzman':
        To = 1.0
        SchedFunc = lambda t: (To/(1+math.log(1+t)) if t < limit else 0)
    elif SchedType == 'ExpMultiplicative':
        To = 2
        SchedFunc = lambda t: (To * 0.993**t) if t < limit else 0
    elif SchedType == 'LogMultip':
        To = 35.0
        Alpha = 1.1 # Alpha>1
        SchedFunc = lambda t: (To/(1+Alpha*math.log(1+t)) if t < limit else 0)
    elif SchedType == 'LinearMultip':
        To = 35.0
        Alpha = 2  # Alpha>0
        SchedFunc = lambda t: (To/(1+Alpha*t) if t < limit else 0)
    elif SchedType == 'QuadraticMultip':
        To = 35.0
        Alpha = 2  # Alpha>0
        SchedFunc = lambda t: (To/(1+Alpha*t**2) if t < limit else 0)
    elif SchedType == 'LinearInvTime':
         To = 1.0
         SchedFunc = lambda t: (To/t if t < limit else 0)
    global TSCHEDULE
    TSCHEDULE = SchedType
    T0 = To   #T0 is the initial temperature which is recorded in the .csv file
    return SchedFunc

def simulated_annealing(StartCoord, schedule=PickSched(),  DeltaObjFunc = 0.25):
    StartSim = time.time()
    random.seed()
    xbest = StartCoord
    xmin = np.zeros(len(StartCoord))
    xmax = np.zeros(len(StartCoord))
    xn = StartCoord
    Gxbest = xbest
    fbest = f(xbest)
    Gfbest = fbest  # In certain situations we guess a global min. but the temperature is too high and we don't accept it. Gfbest captures this situation.
    t = 1.0;
    CutOff = 0;
    StateSpaceCutOff = 0
    global Xs, Ys, Zs, Xtime, Ytemperature, YProbGraph, YbestOBJ, YOBJ, DELTASpace, start
    while True:
        T = schedule(t)
        for SameTemperature in range(0, POINTSperTEMPERAT):
            if T == 0 or (APPLYcutoff==True and CutOff == (CUTOFFvalue+1)):
                elapsed = time.time() - StartSim
                if Gfbest < fbest: xbest, fbest = Gxbest, Gfbest
                if PRINTSOLUTION:
                    #print("T Schedule:", TSCHEDULE)
                    print("Time elapsed: ", elapsed)
                    print ('Best solution: ' + str(xbest))
                    print ('Best objective: ' + str(fbest))
                    print ()
                if WRITEtoFILE:
                    Result = [Iteration + 1, elapsed, t - 1, xbest, fbest, T0]
                    csvADD(Result)
                return (xbest, fbest)
            Xtime.append(t)
            if GRAPHtemperSCHED:
                Ytemperature.append(T)
            if STATEspaceNARROWING and StateSpaceCutOff > STATEspaceNARvalue:
                for nd in range(len(StartCoord)):
                    xmax[nd] = (DELTASpace * abs(xbest[nd]))
                    xmin[nd] = -xmax[nd]  # Apply STATEspace Narrowing
#                xmax = DELTASpace * abs(xbest)
#                xmin = -xmax
#                xn = xbest + np.random.uniform(min(xmin), max(xmax), len(StartCoord))
                    xn[nd] = xbest[nd]+ random.uniform(xmin[nd], xmax[nd])
                    if xn[nd] < -OBJfunctionRANGE: xn[nd] = -OBJfunctionRANGE
                    if xn[nd] > OBJfunctionRANGE: xn[nd] = OBJfunctionRANGE
                print("Narrowing...")
                if StateSpaceCutOff / STATEspaceNARvalue > 2.0:  # If stuck on this local minimum (fbest), explore Gfbest
                    if Gfbest < fbest:
                        xbest, fbest = Gxbest, Gfbest
                        print("Exploring around lowest value found...")
            else:
                #xn = np.random.uniform(-OBJfunctionRANGE, OBJfunctionRANGE, len(StartCoord))  #Outra forma de trabalhar.  NAO APAGAR
                #yn = random.uniform(-OBJfunctionRANGE, OBJfunctionRANGE)  #Outra forma de trabalhar.  NAO APAGAR
#
#                for nd in range(len(StartCoord)):
#                    #nd = int(random.uniform(0, len(StartCoord) - 1))
#                #xn = xbest - Grad
#                    xn[nd] = xbest[nd] + random.uniform(-DeltaObjFunc*OBJfunctionRANGE, DeltaObjFunc*OBJfunctionRANGE)
#                    if xn[nd] < -OBJfunctionRANGE: xn[nd] = -OBJfunctionRANGE
#                    if xn[nd] > OBJfunctionRANGE: xn[nd] = OBJfunctionRANGE
            # nd = int(t % len(StartCoord))
            # xn[nd] = xbest[nd] + random.uniform(-DeltaObjFunc * OBJfunctionRANGE, DeltaObjFunc * OBJfunctionRANGE)
                NN.setWeights()
                xn = NN.getParams()

            fn = f(xn)
            try:
                #Prob = math.exp(-(fn-fbest)/ T)
                Prob = 1/(1+math.exp((fn - fbest) / T))
            except OverflowError:
                Prob = 0
            if GRAPHProb:
                YProbGraph.append(Prob)
            if PRINTiter:
                # Monitor the temperature & cost
                print("t:", "%.0f" % round(t, 0), "Temp:", "%.10fC" % round(T, 10),
                      "x:", np.around(xn, 5),
                      #"nd:", np.around(nd, 5),
                      "z:", "%.5f" % round(fn, 5),
                      "zbest:", "%.5f" % round(fbest, 5),
                      "P:", "%.5f" % round(Prob, 5),
                      #"DeltaSpace:", "%.4f" % round(DELTASpace, 4),
                      "StateCutOff:", "%.0f" % StateSpaceCutOff,
                      "CutOff:", "%.0f" % CutOff)
            CutOff += 1
            if fn < Gfbest: Gxbest, Gfbest = xn, fn #if fn < Gfbest always accept smallest cost
            if fn < fbest or (random.uniform(0.0, 1.0) < Prob):
                xbest, fbest = xn, fn
                CutOff=0
                StateSpaceCutOff = 0
                if GRAPH:
                    Xs.append(xbest)
                    Zs.append(fbest)
            else:
                StateSpaceCutOff += 1
            if GRAPHobjValue:
                YbestOBJ.append(fbest)
                YOBJ.append(fn)
        t += 1


Xs=[]; Ys=[]; Zs=[]; Xtime=[]; Ytemperature=[]; YProbGraph=[]; YbestOBJ=[]; YOBJ=[]; ItValue = []
#System Parameters
OBJfunctionRANGE = 6.0  #Explores Objetive function from -OBJfunctionRANGE to +OBJfunctionRANGE in x and y directions
DELTAObjFunc = 0.10 #Used to calculate the next guess for x and y:  xn = xbest + random.uniform(-DeltaObjFunc*OBJfunctionRANGE, DeltaObjFunc*OBJfunctionRANGE)
WRITEtoFILE = False  #Exports the result of each Anneal to .csv file
GRAPH = 0
GRAPHtemperSCHED = 1
GRAPHProb = 1
GRAPHsurface = 0
GRAPHobjValue = 1
GRAPHhistogram = 0
PRINTiter = 1
PRINTSOLUTION = 1
PRINTbatchSOLUTION = True
# Usar NUMpoints = 3000 T0 = 50 PointsperTEMPERAT = 1 NUMITER=5 para Neural networks
NUMpoints = 6000  #Number of random points per Annealing simulation
POINTSperTEMPERAT = 1
NUMITER = 2 #Number of times to run the Annealing Function
APPLYcutoff = 0
CUTOFFvalue = 405
STATEspaceNARROWING = 1 #Turns space narrowing on and off
STATEspaceNARvalue = 10 #Narrows the search space if Best Solution has not chenged in STATEspaceNARvalue trials
DELTASpace = 0.2
TSCHEDULE = 'ExpDecay'
T0 = 50
RECURSIVEstart = True
STARTdelta = 0
USEANNEAL = True
REDUCEDsample = True


Iteration = 0
def Anneal(XVect):
    ItValue = [] 
    global STARTdelta
    if WRITEtoFILE:
        csvADD(['ITER','t','Points','x','y','Objective','To','Schedule:',TSCHEDULE,'ObjRange:',OBJfunctionRANGE,
                'Narrowing:',STATEspaceNARROWING,'NarrValue:',STATEspaceNARvalue,
                'DeltaSp:', DELTASpace,'CutOff:',APPLYcutoff, 'COvalue:',CUTOFFvalue,'Points/T:', POINTSperTEMPERAT])
    Bxbest = XVect
    Bfbest = f(Bxbest)
    batchTime = time.time()
    #StartC = XVect #np.ones(len(XVect))
    #StartC = [ 6.60862299,  1.95226715, 0, -1.92782475, -0.16292057,-0.45597759, -0.58898731,  3.77457146, -5.05651567]
    #StartC = [ 6.60862299,  1.95226715, -3.67397825, -1.92782475, -0.16292057,-0.45597759, -0.58898731,  3.77457146, -5.05651567]
    for Iteration in range(NUMITER):
        print("It:", Iteration+1)
        print ("Start Guess:", XVect)
        Ixbest, Ifbest = simulated_annealing(XVect, PickSched(TSCHEDULE, NUMpoints), DELTAObjFunc)
        if GRAPHhistogram: ItValue.append(Ifbest)
        if Ifbest < Bfbest: Bxbest, Bfbest = Ixbest,  Ifbest
        # if Ifbest == Bfbest:
        #     Counter =+ 1
        #     if Counter ==10: STARTdelta =+ Bxbest*0.04
        if RECURSIVEstart:
            XVect = Bxbest + np.random.uniform(-STARTdelta, STARTdelta,len(Bxbest))

    if PRINTbatchSOLUTION:
        print("")
        print("Number of Iterations:", NUMITER)
        print("Best Minimum Found:")
        print("T Schedule:", TSCHEDULE)
        print("Batch Time elapsed: ", time.time()-batchTime)
        print('Best Batch solution: ' + str(Bxbest))
        print('Best Batch objective: ' + str(Bfbest))
    
    if WRITEtoFILE: csvADD(['Process Time:',time.time()-batchTime])
    
    if GRAPH:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(Xs,Ys,Zs, c='b', marker='o')
        ax1.legend(['Points Tryed'])
    
    if GRAPHtemperSCHED:
        fig, ax1 = plt.subplots()
        ax1.plot(Xtime, Ytemperature, 'b.', markersize=2)
        ax1.set_xlabel('Time')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Temperature', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        if GRAPHProb:
            ax2 = ax1.twinx()
            ax2.plot(Xtime, YProbGraph, 'ro',markersize=2)
            ax2.set_ylabel('Prob.', color='r')
            ax2.set_ylim([-0.5, 2])
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
    if GRAPHobjValue:
        fig3, ax3 = plt.subplots()
        ax3.plot(Xtime, YOBJ, 'bx',markersize=2)
        ax3.plot(Xtime, YbestOBJ, 'ro', markersize=3)
        ax3.set_xlabel('Time')
        # Make the y-axis label and tick labels match the line color.
        ax3.set_ylabel('Function Value')

    if GRAPHsurface:
        # Design variables at mesh points
        i1 = np.arange(-OBJfunctionRANGE, OBJfunctionRANGE, 0.1)
        i2 = np.arange(-OBJfunctionRANGE, OBJfunctionRANGE, 0.1)
        x1m, x2m = np.meshgrid(i1, i2)
        fm = np.zeros(x1m.shape)
        #xg = np.zeros(x1m.shape)
        for i in range(x1m.shape[0]):
            for j in range(x1m.shape[1]):
                #xg[i][j] =[x1m[i][j],x2m[i][j],0,0,0,0,0,0,0]
                fm[i][j], gr = f([x1m[i][j],x2m[i][j],1,1,1,1,1,1,0,0])
        fig2 = plt.figure()
        ax = fig2.gca(projection='3d')
        #ax.plot_surface(x1m, x2m, fm, rstride=4, cstride=4, color='y')
        ax.plot_surface(x1m, x2m, fm, cmap=cm.jet)
    if GRAPHhistogram:
        import matplotlib.mlab as mlab
        # the histogram of the data
        fig = plt.figure()
        ax = fig.add_subplot(111)
        numBins = 50
        n, bins, patches = ax.hist(ItValue, numBins, color='blue', alpha=0.8)
        plt.grid(True)
        rects = ax.patches
        # Now make some labels
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, 1.01 * height, '%d' % int(height), ha='center',
                        va='bottom', fontsize=8)

        # add a 'best fit' line
        ItValue = np.array(ItValue)
        mu = np.mean(ItValue)
        sigma = np.std(ItValue)
        MaxValue = np.max(ItValue)
        MinValue = np.min(ItValue)
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        print(bins[0:2])
        NormCurve = mlab.normpdf(bincenters, mu, sigma)
        title_string = 'Histogram of Best Vals: \nm=' + str(round(mu, 3)) + '  s=' + str(
            round(sigma, 3)) + '  Min=' + str(round(MinValue, 6)) + \
                       '  Max=' + str(round(MaxValue, 5))
        # plt.axes([.1, .1, .8, .7])
        plt.figtext(.5, 0.02,
                    'Niter=' + str(NUMITER) + '  NPts=' + str(NUMpoints) + '  Pts/T=' + str(POINTSperTEMPERAT) +
                    '  To=' + str(int(T0)) + '  Schd: ' + TSCHEDULE + '  FRange: ' + str(
                        OBJfunctionRANGE) + '  DtaObj: ' + str(DELTAObjFunc) + '  RS: ' + str(RECURSIVEstart) +
                    '\nNarr: ' + str(STATEspaceNARROWING) + '    NarrValue: ' + str(int(STATEspaceNARvalue)) +
                    '    DtaSp: ' + str(DELTASpace) + '    CutOff: ' + str(APPLYcutoff) + '    COValue: ' + str(
                        CUTOFFvalue) + '    BinMax: ' + str(round(bins[1], 6)), fontsize=10, ha='center')
        # plt.suptitle(title_string, y=1.05, fontsize=10)
        plt.title(title_string, fontsize=13)
        # plt.axis([-4, 0,0,50])
        # plt.grid(True)
        ax2 = ax.twinx()
        l = ax2.plot(bincenters, NormCurve, 'r-', linewidth=2)
        ax.set_ylabel('Frequency', color='b')
        ax2.set_ylabel('Fitted Normal Dist.', color='r')
        print(int(ax.patches[0].get_height()))

    if GRAPH or GRAPHtemperSCHED or GRAPHsurface or GRAPHhistogram:
        plt.show()
    return Bxbest



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer
    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sigmoid_result = 1.0 / (1.0 + np.exp(-1.0 * z));
    return sigmoid_result



def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    mat = loadmat('mnist_all.mat')
    # Dividing the data into training, test and Validation data
    data_train = np.empty((0, 784))
    trn_lab = np.empty((0, 1))
    data_test = np.empty((0, 784))
    tes_lab = np.empty((0, 1))
    data_val = np.empty((0, 784))
    val_lab = np.empty((0, 1))
    for i in range(10):
        m1 = mat.get('test' + str(i))
        m2 = mat.get('train' + str(i))
        num1 = m1.shape[0]
        num2 = m2.shape[0]
        num3 = int(0.83342 * num2)
        num4 = num2 - num3
        b = range(m2.shape[0])
        permut_b = np.random.permutation(b)
        Z1 = m2[permut_b[0:num3], :]
        Z2 = m2[permut_b[num3:], :]
        data_train = np.vstack([data_train, Z1])
        data_val = np.vstack([data_val, Z2])
        data_test = np.vstack([data_test, m1])
        for p in range(num3):
            trn_lab = np.append(trn_lab, i)
        for q in range(num4):
            val_lab = np.append(val_lab, i)
        for r in range(num1):
            tes_lab = np.append(tes_lab, i)

    # normalizing the data to values between to 0-1.
    data_test = data_test / 255
    data_train = data_train / 255
    data_val = data_val / 255

    train_data = data_train
    train_label = trn_lab
    validation_data = data_val
    validation_label = val_lab
    test_data = data_test
    test_label = tes_lab

    print("Train Data Size: ",train_data.shape)
    print("Train Label Size: ",train_label.shape)
    print("Validation Data Size: ",validation_data.shape)
    print("Validation Lable Size: ",validation_label.shape)
    print("Test Data Size: ",test_data.shape)
    print("Test Lable Size: ",test_label.shape)
    print()

    if REDUCEDsample:  #Resize training data set to TRAIN_NEWsize points
        TRAIN_NEWsize = 10000
        nsamples = len(train_data)
        x = train_data.reshape((nsamples, -1))
        Y = train_label
        #Create Random indices
        valid_index = random.sample(range(int(len(x))), TRAIN_NEWsize)
        train_data = [x[i] for i in valid_index]
        train_data = np.array(train_data)
        # validation targets
        train_label = [Y[i] for i in valid_index]
        train_label = np.array(train_label)

        VALIDATION_NEWsize = 50
        nsamples = len(validation_data)
        x = validation_data.reshape((nsamples, -1))
        Y = validation_label
        # #Create Random indices
        valid_index = random.sample(range(int(len(x))), VALIDATION_NEWsize)
        validation_data = [x[i] for i in valid_index]
        validation_data = np.array(validation_data)
        # validation targets
        validation_label = [Y[i] for i in valid_index]
        validation_label = np.array(validation_label)

        # Train images and labels
        # train_index = [i for i in range(len(x)) if i not in valid_index]
        # train_data =[x[i] for i in train_index]
        # train_label=[Y[i] for i in sample_index]

        # I do not reduce the test set size so that I have as many samples as I can to test the model

        print("Train Data New Size: ", train_data.shape)
        print("Train Label New Size: ", train_label.shape)
        print("Validation New Data Size: ", validation_data.shape)
        print("Validation New Data Size: ", validation_label.shape)
        print("Test Data New Size: ", test_data.shape)
        print("Test Data New Size: ", test_label.shape)

    return train_data, train_label, validation_data, validation_label, test_data, test_label






def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #
    training_label = np.array(training_label)
    rows = training_label.shape[0]
    rowsIndex = np.arange(rows, dtype="int")
    #
    tempLabel = np.zeros((rows, 10))
    tempLabel[rowsIndex, training_label.astype(int)] = 1
    training_label = tempLabel

    # nnFeedForwardward propogation
    # adding bias to the input data
    training_data = np.column_stack((training_data, np.ones(training_data.shape[0])))
    number_of_samples = training_data.shape[0]

    # passing the input data to the Hidden layer [calculating a2 = sigmoid(X.W1)]
    zj = sigmoid(np.dot(training_data, w1.T))
    #print(zj.shape)
    # adding bias to the hidden layer
    zj = np.column_stack((zj, np.ones(zj.shape[0])))
    # passing the hidden layer data to the output layer  [calculating Yhat = sigmoid(a2.W2)]
    #print(zj.shape)
    #print(w2.T.shape)
    ol = sigmoid(np.dot(zj, w2.T))

    # Back propogation
    deltaOutput = ol - training_label
    error = np.sum(-1 * (training_label * np.log(ol) + (1 - training_label) * np.log(1 - ol)))
    error = error / number_of_samples
    gradient_of_w2 = np.dot(deltaOutput.T, zj)
    gradient_of_w2 = gradient_of_w2 / number_of_samples

    gradient_of_w1 = np.dot(((1 - zj) * zj * (np.dot(deltaOutput, w2))).T, training_data)
    gradient_of_w1 = gradient_of_w1 / number_of_samples
    gradient_of_w1 = np.delete(gradient_of_w1, n_hidden, 0)
    obj_grad = np.concatenate((gradient_of_w1.flatten(), gradient_of_w2.flatten()), 0)

    error = error + (lambdaval / (2 * number_of_samples)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = error
    return (obj_val, obj_grad)


def nnFeedForward(data, w1, w2):
    a = np.dot(data, w1.T)
    z = sigmoid(a)
    z = np.append(z, np.zeros([len(z), 1]), 1)
    b = np.dot(z, w2.T)
    o = sigmoid(b)
    index = np.argmax(o, axis=1)
    label = np.zeros((o.shape[0], 10))
    for i in range(label.shape[0]):
        label[i][index[i]] = 1
    return (z, o)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    % w1(i, j) represents the weight of connection from unit i in input
    % layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    % w2(i, j) represents the weight of connection from unit i in input
    % layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    % vector of a particular image
    % Output:
    % label: a column vector of predicted labels"""
    data = np.append(data, np.zeros([len(data), 1]), 1)
    n = data.shape[0]
    z, o = nnFeedForward(data, w1, w2);
    label = np.empty((0, 1))
    for i in range(n):
        index = np.argmax(o[i]);
        label = np.append(label, index);
    return label


"""**************Neural Network Script Starts here********************************"""

start_Time = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
print("Input Layer size:",n_input)
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
print("Initial Weights before Anneal:")
print("Initial Weights Shape: ",initialWeights.shape)
print("Initial Weights Value: ",initialWeights)

lambdaval = 0.2;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
if USEANNEAL:
    NN = Neural_Network()
    initialWeights = Anneal(initialWeights)
    # SimAnneal = sam.SA("Z",costFunctionWrapper2,2,6000,args=args)
    # initialWeights, SolValue = SimAnneal.Anneal(initialWeights)
    print("Anneal Optimized Weights: ", initialWeights)
    #print("Anneal Solution: ", SolValue)

# TRAIN NEURAL NETWORK using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}  # Preferred value.  method CG Training set Accuracy:94.784%
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

print("Value after Optimization Gradient:", nn_params.fun)
print("Solution:", nn_params.x)
print("Max:", max(nn_params.x))
print("Min:", min(nn_params.x))
print

# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
# Test the computed parameters

# find the accuracy on Training Dataset
predicted_label = nnPredict(w1, w2, train_data)
print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# # find the accuracy on Validation Dataset
predicted_label = nnPredict(w1, w2, validation_data)
print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# find the accuracy on the Test Dataset
predicted_label = nnPredict(w1, w2, test_data)
print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print()
total_Time = time.time() - start_Time
print("Size test_label: ", len(test_label))
print("Size predicted_label: ", len(predicted_label))
print()
print("Runing Time:", total_Time)
print()
while True:
    ImageIndex = input('Enter a Test image index from 0 to 9999: ')
    if ImageIndex == 'x': break
    ImageIndex = int(ImageIndex)
    print("Test Label: ", int(test_label[ImageIndex]))
    #predicted_label = nnPredict(w1, w2, test_data[ImageIndex:ImageIndex+1])
    print("Pred Label: ", int(predicted_label[ImageIndex]))
    pl.matshow(test_data[ImageIndex].reshape(28, 28), cmap=plt.cm.gray)
    pl.show()

print()
pickle.dump((n_hidden, w1, w2, lambdaval), open('params.pickle', 'wb'))
pickle_data = pickle.load(open('params.pickle', 'rb'))
#print(pickle_data)


