# IceSheet class
# Created 2021-03-21

import os
import json
import numpy as np
import scipy.sparse as ss


import geomUtils


class IceSheet():
    """[summary]
    """
    def __init__(self, type='lake', thickness=10.0, crystalSize=100.0):
        """[summary]

        Args:
            type (str, optional): Type of water that the ice is formed in. Defaults to 'lake'.
            thickness (float, optional): Initial thickness of icesheet, in mm. Defaults to 10 mm.
            crystalSize (float, optional): Crystal size in mm. Defaults to 100 mm.
        """
        self._iceType = type
        self._h0 = thickness
        self._crystSize = crystalSize
        self._crystType = 'hex'

        currentDir = os.getcwd()
        dataFilePath = os.path.join(currentDir, '../data' )
        dataFilePath = os.path.join(dataFilePath, 'freshWaterProps.json')
        dataFile = open(dataFilePath, 'r')
        water = json.load(dataFile)
        dataFile.close()
        self._ice = water['solid']

    def getBorderDensity(self):
        return geomUtils.periphery(geom=self.crystType, dims=[self._crystSize])/geomUtils.area(geom=self.crystType, dims=[self._crystSize])

    def setupModel(self, IC=4., BC=0., TEnv=[-4.0, -20.0, -270.]):
        # process
        T0 = IC  # °C
        TSurfAve = BC  # °C
        TAmb = TEnv[0]  # °C
        TAtm = TEnv[1]  # °C
        TSpc = TEnv[2]  # °C



        aAmb = 0.
        aAtm = 0.75
        aSpc = 1 - aAmb - aAtm

        # properties
        lambdaW = 0.56  # W/(m K)
        rhoW = 999.8  # kg/m3
        cpW = 4218.  # J/(kg K)
        # lambdaW = 14.  # W/(m K)
        # rhoW = 8000.  # kg/m3
        # cpW = 500.  # J/(kg K)
        a = lambdaW/rhoW/cpW
        print(a)

        # spatial grid
        LScale = 0.2  # m
        XScale = 1.0  # ---
        Nx = 50  # number of space steps
        dx = XScale/(Nx - 1)
        x_grid = np.array([i*dx for i in range(Nx)])
        print(dx)

        # temporal grid
        tScale = LScale**2/a  # s
        tChar = tScale  # 3600  # s
        print(tScale, tScale/tChar)
        YScale = tScale/tChar
        print(YScale)
        Ny = 1000  # number of time steps
        dy = YScale/(Ny - 1)
        # dy = dt/1.
        y_grid = np.array([i*dy for i in range(Ny)])
        tSoln = YScale*tChar
        print(dy, tSoln)

        # parameters
        # sigma = a*dy/(2.*dx**2)
        sigma = a*(tChar*dy)/(2.*(LScale*dx)**2)
        print(sigma)

        epsilon = 0.95
        sigmaSB = 5.67E-8  # W/(m2 K4)
        qRad = epsilon*sigmaSB*(np.power(TSurfAve+273.15, 4) - aAmb*np.power(TAmb+273.15, 4) - aAtm*np.power(TAtm+273.15, 4) - aSpc*np.power(TSpc+273.15, 4))
        alphaRad = qRad/(TSurfAve - TAmb)
        Bi = alphaRad*LScale/lambdaW
        print(qRad, alphaRad, Bi)

        def u(T, T0, TChar):
            return (T-TChar)/(T0-TChar)

        U0 = u(T0, T0, TAmb)
        UAmb = u(TAmb, T0, TAmb)

        # set up matrices
        A = ss.diags([-sigma, 1 + 2*sigma, -sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()
        B = ss.diags([sigma, 1 - 2*sigma, sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()

        # BC at x = 0
        A[0, 0] = 1 + 2*(1 + Bi*dx)*sigma
        A[0, 1] = -2*sigma
        B[0, 0] = 1 - 2*(1 + Bi*dx)*sigma
        B[0, 1] = 2*sigma

        # BC at x = 1
        A[Nx-1, Nx-2] = -2*sigma
        B[Nx-1, Nx-2] = 2*sigma

        # boundary condition vector
        b = np.zeros(len(B))
        b[0] = 4*Bi*dx*sigma*UAmb

        # IC
        # U =  np.array([U0 for i in range(0, Nx)])
        U = np.full(Nx, U0)

        return A, B, b, U
