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

    def __init__(self, dateTime, lat=55.71, alt=10, sheet='blackIce', thickness=10.0, crystalSize=100.0):
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
        if sheet in ['freshWater', 'saltWater']:
            self.isIce = False
            self.isWater = True
        else:
            self.isIce = True
            self.isWater = False
        self._sheet = sheet
        self._h = thickness*0.001  # m
        self._crystSize = crystalSize*0.001  # m
        self._crystType = 'hex'

        # physical properties
        currentDir = os.getcwd()
        dataFilePath = os.path.join(currentDir, '../data' )
        dataFile = open(os.path.join(dataFilePath, 'physicalProps.json'), 'r')
        water = json.load(dataFile)
        dataFile.close()
        if self.isIce:
            self._material = water['solid'][sheet]
        else:
            self._material = water['liquid'][sheet]
        self._materialDH = water['enthalpy']
        self._transmittance = 1

        # lake ice properties
        # dataFile = open(os.path.join(dataFilePath, 'lakeIceProps.json'), 'r')
        # ice = json.load(dataFile)
        # dataFile.close()
        # self._ice = ice[sheet]

        print('When: {}'.format(self._dateTimeStart))
        print('Where: lat = {:.2f}°, alt = {:.0f} m'.format(self._lat, self._alt))
        print('What: {}'.format(self._sheet))


    def getBorderDensity(self):
        return geomUtils.periphery(geom=self.crystType, dims=[self._crystSize])/geomUtils.area(geom=self.crystType, dims=[self._crystSize])


    def makeTZeroD(self, T):
        return (T - self._TLo)/(self._THi - self._TLo)


    def makeUOneD(self, u):
        return u*(self._THi - self._TLo) + self._TLo


    def setupModel(
        self, 
        IC=[0., 4.], 
        windSpeed=5., 
        aEnv=[1.], 
        TEnv=[0.0], 
        zNodes=51, 
        tStep=120,
        isVerbose=False
    ):
        """Use default values for water sheet (always the same),
        otherwise (i.e. if ice sheet) always pass parameter values.

        Args:
            IC (list, optional): [description]. Defaults to [-2., 0.].
            windSpeed ([type], optional): [description]. Defaults to 5..
            aEnv (list, optional): [description]. Defaults to [0., 0.75, 0.25].
            TEnv (list, optional): [description]. Defaults to [-2.0, -20.0, -270.].
            zNodes (int, optional): [description]. Defaults to 51.
            tStep (int, optional): [description]. Defaults to 120.
            isVerbose (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        # properties
        lmbda = self._material['lambda']  # W/(m K)
        rho = self._material['rho']  # kg/m3
        cp = self._material['cp']  # J/(kg K)
        a = lmbda/rho/cp  # m2/s
        if isVerbose:
            print('thermal diffusivity a = {:.2e} m2/s'.format(a))

        # spatial grid
        zScale = self._h  # m
        zStep = zScale/(zNodes - 1)
        dx = zStep/zScale
        if isVerbose:
            print('spatial stepsize dz = {:.2f} mm'.format(zStep*1000))

        # temporal grid
        tScale = zScale**2/a  # s
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
        self._THi = IC[-1]  # °C
        if self.isIce:
            self._TLo = TEnv[0]  # TAmb; °C
        else:
            self._TLo = IC[0]  # TBot; °C

        # IC, linear
        self.setIC(IC)
        U = self._IC

        # set self._b
        # sun stuff
        S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, self._dateTimeStart, self._lat)
        S0 = S0*self._transmittance
        if isVerbose:
            print('irradiance S0 = {:.0f} W/m2'.format(S0))
        self.setSourceVector(S0)

        # apply BCs
        A, B, b = self.applyBCs(aEnv, TEnv)

        return A, B, b, U


    def setIC(self, IC):
        if len(IC) == 2:
            TInit = np.linspace(IC[0], IC[-1], self._Nx)
        else:
            TInit = IC
        UInit = self.makeTZeroD(TInit)
        self._IC = UInit

    def getIC(self):
        return self._IC


    def setSourceVector(self, S0):
        I0 = (1 - self._material['albedo'])*S0  # W/m2
        # irradiance (differentiated, non-dimensionalized)
        i0 = I0/(self._material['lambda']/self._h)/(self._THi - self._TLo)
        absh = self._material['absorptionCoefficient']*self._h
        self._b = np.array([self.nonDimIrradSource(j, self._dx, self._dy, i0, absh) for j in range(self._Nx)])


    def applyBCs(self, aEnv, TEnv):
        if self.isIce:
            TAmb = TEnv[0]  # °C
            TAtm = TEnv[1]  # °C
            TSpc = TEnv[2]  # °C
            TTop = self.makeUOneD(self._IC[0])  # °C
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
            alphaConv = 100.  # W/m2
            alphaBdry = alphaRad + alphaConv
            # print('qRad = {:.0f} W/m2, alphaRad = {:.0f} W/(m2 K), Bi = {:.2e}'.format(qRad, alphaRad, Bi))
        else:
            TAmb = self.makeUOneD(self._IC[0])  # °C
            TBulk = self.makeUOneD(self._IC[-1])  # °C
            # TODO: alphaConv, use windSpeed to get alphaConv
            # alphaConv = 0.01  # W/m2
            # alphaCond = -self._material['lambda']*(self.makeUOneD(self._IC[1]) - TAmb)/(-self._dx*self._h)/(TBulk - TAmb)
            # alphaBdry = alphaConv + alphaCond  # max(alphaConv, alphaCond)
            alphaBdry = 0
        Bi = alphaBdry*self._h/self._material['lambda']

        dx = self._dx
        sigma = self._sigma

        A = self._A.copy()
        B = self._B.copy()
        b = self._b.copy()

        if self.isIce:
            # BC at x = 0
            A[0, 0] = 1 + 2*(1 + Bi*dx)*sigma
            A[0, 1] = -2*sigma
            B[0, 0] = 1 - 2*(1 + Bi*dx)*sigma
            B[0, 1] = 2*sigma
            
            # BC at x = 1
            # does not change A, B

            # boundary conditions in source term vector
            b[0] = b[0] + 4*sigma*dx*Bi*self.makeTZeroD(TAmb)
            b[-1] = b[-1] + 2*sigma*self._IC[-1]
        else:
            # BC at x = 0
            # does not change A, B
            
            # BC at x = 1
            # does not change A, B

            # boundary conditions in source term vector
            b[0] = b[0] + 2*sigma*self._IC[0]  # TAmb
            b[-1] = b[-1] + 2*sigma*self._IC[-1]  # TBulk

        return A, B, b


    def nonDimIrradSource(self, j, dx, dy, i0, alphah):
        return i0*alphah*dy*np.exp(-alphah*j*dx)


    def simulate(
        self, U, aEnv, TEnv, timeStepCount, 
        TIce=np.array([]), rhoIce=0, lambdaIce=0, dxIce=0, hIce=0
    ):
        dateTime = self._dateTimeStart
        S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, dateTime, self._lat)
        S0 = S0*self._transmittance
        
        time = 0

        dateTimes = []
        S0s = []
        USoln = []
        epsSoln = []
        botMeltRates = []
        for step in range(timeStepCount):
            dateTimes.append(dateTime)
            S0s.append(S0)
            
            if self.isIce:
                # compute porosity epsilon(t, z)
                DTMelt = np.maximum(0, self.makeUOneD(U) - 0)
                porosity = self._material['cp']*DTMelt/(self._materialDH['fusion']*1000)
                epsSoln.append(porosity)
                # limit U (due to melting)
                U = np.minimum(U, np.full(len(U), 1))
            else:
                # compute bottom surface melting
                dT = self.makeUOneD(U[1]) - 0
                qDotToBot = -self._material['lambda']*dT/(2*self._dx*self._h)
                dT = 0 - TIce[dateTimes.index(dateTime), -2]
                qDotFromBot = -lambdaIce*dT/(2*dxIce*hIce)
                botMeltRate = -(qDotToBot - qDotFromBot)/(rhoIce*self._materialDH['fusion']*1000)
                botMeltRates.append(botMeltRate)

            USoln.append(U)

            time = (step + 1)*self._tStep
            dateTime = self._dateTimeStart + datetime.timedelta(seconds=+time)
            S0 = solarpy.irradiance_on_plane(self._vnorm, self._alt, dateTime, self._lat)
            S0 = S0*self._transmittance
            self.setSourceVector(S0)
            A, B, b = self.applyBCs(aEnv, TEnv)
            U = np.linalg.solve(A, B.dot(U) + b)  
            # Note that the U on the right-hand side is the IC for this round's integration
            
            # output also: bIrradiance, ...
            
        S0s = np.array(S0s)
        USoln = np.array(USoln)
        if self.isIce:
            melting = np.array(epsSoln)  # fraction at (t, z)
        else:
            melting = np.array(botMeltRates)  # m/s at (t, z=hIce)

        return dateTimes, S0s, USoln, melting

    
    def setTransmittance(self, sheet):
        self._transmittance = np.exp(-sheet._material['absorptionCoefficient']*sheet._h)

