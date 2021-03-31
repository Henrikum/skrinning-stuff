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

    def __init__(self, iceType='blackIce', thickness=10.0, crystalSize=100.0):
        """[summary]

        Args:
            iceType (str, optional): Type of ice being analyzed. Defaults to 'blackIce'.
            thickness (float, optional): Initial thickness of icesheet (mm). Defaults to 10 mm.
            crystalSize (float, optional): Crystal size (mm). Defaults to 100 mm.
        """
        self._iceType = iceType
        self._h0 = thickness*0.001  # m
        self._crystSize = crystalSize*0.001  # m
        self._crystType = 'hex'

        # pure water properties
        currentDir = os.getcwd()
        dataFilePath = os.path.join(currentDir, '../data' )
        dataFile = open(os.path.join(dataFilePath, 'pureWaterProps.json'), 'r')
        water = json.load(dataFile)
        dataFile.close()
        self._pureIce = water['solid']

        # lake ice properties
        dataFile = open(os.path.join(dataFilePath, 'lakeIceProps.json'), 'r')
        ice = json.load(dataFile)
        dataFile.close()
        self._ice = ice[iceType]

        print('Ice type: {}'.format(self._iceType))


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


    def setupModel(
        self, 
        IC=[-2., 0.], 
        windSpeed=5., 
        aEnv=[0., 0.75, 0.25], 
        TEnv=[-2.0, -20.0, -270.], 
        S0=100.,
        Nx=51, 
        Ny=101,
        isVerbose=False
    ):

        # properties
        lmbda = self._pureIce['lambda']  # W/(m K)
        rho = self._pureIce['rho']  # kg/m3
        cp = self._pureIce['cp']  # J/(kg K)
        a = lmbda/rho/cp  # m2/s
        if isVerbose:
            print('thermal diffusivity a = {:.2e} m2/s'.format(a))

        # spatial grid
        zScale = self._h0  # m
        zStep = zScale/(Nx - 1)
        dx = zStep/zScale
        if isVerbose:
            print('spatial stepsize dz = {:.2f} mm'.format(zStep*1000))

        # temporal grid
        tScale = zScale**2/a  # s
        tStep = tScale/(Ny - 1)  # s
        dy = tStep/tScale
        if isVerbose:
            print('temporal stepsize dt = {:.2f} s'.format(tStep))
            print('timescale tScale = {:.1f} s'.format(tScale))

        # parameter: sigma = a*dt/(2.*dz**2) = a*(tScale*dy)/(2.*(LScale*dx)**2) = dy/(2.*dx)
        sigma = dy/2./dx**2
        if isVerbose:
            print('sigma = {}'.format(sigma))

        # set up matrices (will not change with time)
        self._A = ss.diags([-sigma, 1 + 2*sigma, -sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()
        self._B = ss.diags([sigma, 1 - 2*sigma, sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()

        # store useful constants
        self._Nx = Nx
        self._dx = dx
        self._dy = dy
        self._sigma = sigma
        self._THi = 0.  # °C
        self._TLo = TEnv[0]  # TAmb; °C

        # set self._b
        self.setSourceVector(S0)

        # apply BCs
        A, B, b = self.applyBCs(IC, aEnv, TEnv)

        # IC, linear
        UInit = self.setIC(IC)

        return A, B, b, UInit


    def setSourceVector(self, S0):
        I0 = (1 - self._ice['albedo'])*S0  # W/m2
        # irradiance (differentiated, non-dimensionalized)
        i0 = I0/(self._pureIce['lambda']/self._h0)/(self._THi - self._TLo)
        absh = self._ice['absorptionCoefficient']*self._h0
        self._b = np.array([self.nonDimIrradSource(j, self._dx, self._dy, i0, absh) for j in range(self._Nx)])


    def applyBCs(self, IC, aEnv, TEnv):
        TAmb = TEnv[0]  # °C
        TAtm = TEnv[1]  # °C
        TSpc = TEnv[2]  # °C
        TSurfAve = np.array(IC).mean()  # °C
        aEnv = aEnv/np.array(aEnv).sum()  # rescale if not sum to 1
        aAmb = aEnv[0]
        aAtm = aEnv[1]
        aSpc = 1 - aAmb - aAtm
        
        # TODO: use the real surface temperature, but must be done when solving (stepping in time)!
        qRad = EPSILON*SIGMA_SB*(
            aAmb*np.power(TAmb+DT_K, 4) 
            + aAtm*np.power(TAtm+DT_K, 4) 
            + aSpc*np.power(TSpc+DT_K, 4) 
            - np.power(TSurfAve+DT_K, 4)
        )
        alphaRad = qRad/(TSurfAve - TAmb)
        # TODO: alphaConv, use windSpeed to get alphaConv
        alphaConv = 100.
        Bi = (alphaRad + alphaConv)*self._h0/self._pureIce['lambda']
        # print('qRad = {:.0f} W/m2, alphaRad = {:.0f} W/(m2 K), Bi = {:.2e}'.format(qRad, alphaRad, Bi))

        dx = self._dx
        sigma = self._sigma

        A = self._A.copy()
        B = self._B.copy()
        b = self._b.copy()

        # BC at x = 0
        A[0, 0] = 1 + 2*(1 + Bi*dx)*sigma
        A[0, 1] = -2*sigma
        B[0, 0] = 1 - 2*(1 + Bi*dx)*sigma
        B[0, 1] = 2*sigma
        
        # BC at x = 1
        # does not change A, B

        # boundary conditions in source term vector
        b[0] = b[0] + 4*sigma*dx*Bi*self._u(TAmb, self._THi, self._TLo)
        b[-1] = b[-1] + 2*sigma*self._u(IC[-1], self._THi, self._TLo)

        return A, B, b


    def setIC(self, IC):
        TInit = np.linspace(IC[0], IC[1], self._Nx)
        UInit = self._u(TInit, self._THi, self._TLo)
        return UInit


    def nonDimIrradSource(self, j, dx, dy, i0, alphah):
        return i0*alphah*dy*np.exp(-alphah*j*dx)
