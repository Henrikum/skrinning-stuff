# IceSheet class
# Created 2021-03-21

import os
import json
import numpy as np
import scipy.sparse as ss


import geomUtils


# constants
EPSILON = 0.95
SIGMA_SB = 5.67E-8  # W/(m2 K4)
DT_K = 273.15  # K

class IceSheet():
    """[summary]
    """

    def __init__(self, type='lake', thickness=10.0, crystalSize=100.0):
        """[summary]

        Args:
            type (str, optional): Type of water that the ice is formed in. Defaults to 'lake'.
            thickness (float, optional): Initial thickness of icesheet (mm). Defaults to 10 mm.
            crystalSize (float, optional): Crystal size (mm). Defaults to 100 mm.
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


    def _u(self, T, THi, TLo):
        """Returns the non-dimensional temperature u.

        Args:
            T (float): the temperature (°C or K)
            THi (float): the high characteristic temperature of the system (°C or K)
            TLo (float): the low characteristic temperature of the system (°C or K)

        Returns:
            float: the non-dimensional temperature (---)
        """
        return (T - TLo)/(THi - TLo)


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
        lmbda = self._ice['lambda']  # W/(m K)
        rho = self._ice['rho']  # kg/m3
        cp = self._ice['cp']  # J/(kg K)
        a = lmbda/rho/cp
        print('thermal diffusivity a = {} m2/s'.format(a))

        # spatial grid
        LScale = self._h0*0.001  # m
        XScale = 1.0  # ---
        Nx = 50  # number of space steps
        dx = XScale/(Nx - 1)
        x_grid = np.array([i*dx for i in range(Nx)])
        print('spatial stepsize dx = {}'.format(dx))

        # temporal grid
        tScale = LScale**2/a  # s
        tChar = tScale  # 3600  # s
        print('timescale tScale = {} s, tScale/tChar = {}'.format(tScale, tScale/tChar))
        YScale = tScale/tChar
        print('YScale = {}'.format(YScale))
        Ny = 1000  # number of time steps
        dy = YScale/(Ny - 1)
        # dy = dt/1.
        y_grid = np.array([i*dy for i in range(Ny)])
        tSoln = YScale*tChar
        print('temporal stepsize dy = {}, solution time tSoln = {} s'.format(dy, tSoln))

        # parameters
        # sigma = a*dy/(2.*dx**2)
        sigma = a*(tChar*dy)/(2.*(LScale*dx)**2)
        print('sigma = {}'.format(sigma))

        qRad = EPSILON*SIGMA_SB*(np.power(TSurfAve+DT_K, 4) - aAmb*np.power(TAmb+DT_K, 4) - aAtm*np.power(TAtm+DT_K, 4) - aSpc*np.power(TSpc+DT_K, 4))
        alphaRad = qRad/(TSurfAve - TAmb)
        Bi = alphaRad*LScale/lmbda
        print('qRad = {} W/m2, alphaRad = {} W/(m2 K), Bi = {}'.format(qRad, alphaRad, Bi))

        U0 = self._u(T0, T0, TAmb)
        UAmb = self._u(TAmb, T0, TAmb)

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
        U = np.full(Nx, U0)

        return A, B, b, U
