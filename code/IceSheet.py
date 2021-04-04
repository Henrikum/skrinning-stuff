# IceSheet class
# Created 2021-03-21

import os
import json
import numpy as np
import scipy.sparse as ss

import datetime
import solarpy

import geomUtils


# constants
EPSILON = 0.95
SIGMA_SB = 5.67E-8  # W/(m2 K4)
DT_K = 273.15  # K

class IceSheet():
    """[summary]
    """

    def __init__(self, dateTime, lat=55.71, alt=10, iceType='blackIce', thickness=10.0, crystalSize=100.0):
        """[summary]

        Args:
            iceType (str, optional): Type of ice being analyzed. Defaults to 'blackIce'.
            thickness (float, optional): Initial thickness of icesheet (mm). Defaults to 10 mm.
            crystalSize (float, optional): Crystal size (mm). Defaults to 100 mm.
        """
        # when
        self._dateTimeStart = dateTime

        # where
        self._vnorm = np.array([0, 0, -1])  # plane pointing zenith
        self._lat = lat  # °
        self._alt = alt  # m (above sea-level)

        # what
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

        print('When: {}'.format(self._dateTimeStart))
        print('Where: lat = {:.2f}°, alt = {:.0f} m'.format(self._lat, self._alt))
        print('What: {}'.format(self._iceType))


    def getBorderDensity(self):
        return geomUtils.periphery(geom=self.crystType, dims=[self._crystSize])/geomUtils.area(geom=self.crystType, dims=[self._crystSize])


    def _u(self, T):
        return (T - self._TLo)/(self._THi - self._TLo)


    def _T(self, u):
        return u*(self._THi - self._TLo) + self._TLo


    def setupModel(
        self, 
        IC=[-2., 0.], 
        windSpeed=5., 
        aEnv=[0., 0.75, 0.25], 
        TEnv=[-2.0, -20.0, -270.], 
        zNodes=51, 
        tStep=120,
        isVerbose=False
    ):

        # properties
        lmbda = self._pureIce['lambda']  # W/(m K)
        rho = self._pureIce['rho']  # kg/m3
        cp = self._pureIce['cp']  # J/(kg K)
        a = lmbda/rho/cp  # m2/s
        if isVerbose:
            print('thermal diffusivity a = {:.2e} m2/s'.format(a))

        # sun stuff
        S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, self._dateTimeStart, self._lat)

        # spatial grid
        zScale = self._h0  # m
        zStep = zScale/(zNodes - 1)
        dx = zStep/zScale
        if isVerbose:
            print('spatial stepsize dz = {:.2f} mm'.format(zStep*1000))

        # temporal grid
        tScale = zScale**2/a  # s
        # tStep = tScale/(Ny - 1)  # s
        # Ny = tScale/tStep + 1
        dy = tStep/tScale
        if isVerbose:
            print('temporal stepsize dt = {:.2f} s'.format(tStep))
            print('timescale tScale = {:.1f} s'.format(tScale))

        # parameter: sigma = a*dt/(2.*dz**2) = a*(tScale*dy)/(2.*(LScale*dx)**2) = dy/(2.*dx)
        sigma = dy/2./dx**2
        if isVerbose:
            print('sigma = {}'.format(sigma))

        # set up matrices (will not change with time)
        Nx = zNodes
        self._A = ss.diags([-sigma, 1 + 2*sigma, -sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()
        self._B = ss.diags([sigma, 1 - 2*sigma, sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()

        # store useful constants
        self._Nx = Nx
        self._dx = dx
        self._tStep = tStep
        self._dy = dy
        self._sigma = sigma
        self._THi = 0.  # °C
        self._TLo = TEnv[0]  # TAmb; °C

        # IC, linear
        self.setIC(IC)

        # set self._b
        self.setSourceVector(S0)

        # apply BCs
        A, B, b = self.applyBCs(aEnv, TEnv)

        U = self._IC

        return A, B, b, U


    def setIC(self, IC):
        if len(IC) == 2:
            TInit = np.linspace(IC[0], IC[-1], self._Nx)
        else:
            TInit = IC
        UInit = self._u(TInit)
        self._IC = UInit

    def getIC(self):
        return self._IC


    def setSourceVector(self, S0):
        I0 = (1 - self._ice['albedo'])*S0  # W/m2
        # irradiance (differentiated, non-dimensionalized)
        i0 = I0/(self._pureIce['lambda']/self._h0)/(self._THi - self._TLo)
        absh = self._ice['absorptionCoefficient']*self._h0
        self._b = np.array([self.nonDimIrradSource(j, self._dx, self._dy, i0, absh) for j in range(self._Nx)])


    def applyBCs(self, aEnv, TEnv):
        TAmb = TEnv[0]  # °C
        TAtm = TEnv[1]  # °C
        TSpc = TEnv[2]  # °C
        TTop = self._T(self._IC[0])  # °C
        aEnv = aEnv/np.array(aEnv).sum()  # rescale if not sum to 1
        aAmb = aEnv[0]
        aAtm = aEnv[1]
        aSpc = 1 - aAmb - aAtm
        
        # TODO: use the real surface temperature, but must be done when solving (stepping in time)!
        qRad = EPSILON*SIGMA_SB*(
            aAmb*np.power(TAmb+DT_K, 4) 
            + aAtm*np.power(TAtm+DT_K, 4) 
            + aSpc*np.power(TSpc+DT_K, 4) 
            - np.power(TTop+DT_K, 4)
        )
        alphaRad = qRad/(TTop - TAmb + 0.0001)
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
        b[0] = b[0] + 4*sigma*dx*Bi*self._u(TAmb)
        b[-1] = b[-1] + 2*sigma*self._IC[-1]

        return A, B, b


    def nonDimIrradSource(self, j, dx, dy, i0, alphah):
        return i0*alphah*dy*np.exp(-alphah*j*dx)


    def simulate(self, U, aEnv, TEnv, timeStepCount):
        dateTime = self._dateTimeStart
        S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, dateTime, self._lat)
        clockHour = 0
        time = 0

        dateTimes = []
        S0s = []
        tSoln = []
        USoln = []
        for step in range(timeStepCount):
            dateTimes.append(dateTime)
            S0s.append(S0)
            tSoln.append(clockHour)
            
            # compute porosity epsilon(t)
        #     meltRate = 
            
            USoln.append(U)

            time = (step + 1)*self._tStep
            clockHour = (6 + time/3600) % 24
            dateTime = self._dateTimeStart + datetime.timedelta(seconds=+time)
            S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, dateTime, self._lat)
            self.setSourceVector(S0)
            A, B, b = self.applyBCs(aEnv, TEnv)
            U = np.linalg.solve(A, B.dot(U) + b)  
            # Note that the U on the right-hand side is the IC for this round's integration
            
            # output also: bIrradiance, porosity(z), ...
            
        USoln = np.array(USoln)
        return dateTimes, S0s, USoln