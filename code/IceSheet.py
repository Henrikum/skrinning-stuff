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
        IC=[-4., 0.], 
        windSpeed=5., 
        aEnv=[0., 0.75, 0.25], 
        TEnv=[-2.0, -20.0, -270.], 
        S0=100.,
        Nx=51, 
        Ny=101
    ):
        # process
        TAmb = TEnv[0]  # °C
        TAtm = TEnv[1]  # °C
        TSpc = TEnv[2]  # °C
        THi = 0.  # °C
        TLo = TAmb
        TSurfAve = np.array(IC).mean()  # °C

        aEnv = aEnv/np.array(aEnv).sum()  # rescale if not sum to 1
        aAmb = aEnv[0]
        aAtm = aEnv[1]
        aSpc = 1 - aAmb - aAtm

        I0 = (1 - self._ice['albedo'])*S0  # W/m2
        print('I0 = {:.3f} W/m2'.format(I0))

        # properties
        lmbda = self._pureIce['lambda']  # W/(m K)
        rho = self._pureIce['rho']  # kg/m3
        cp = self._pureIce['cp']  # J/(kg K)
        a = lmbda/rho/cp  # m2/s
        alpha = self._ice['absorptionCoefficient']
        print('thermal diffusivity a = {:.2e} m2/s'.format(a))
        print('absorption coefficient alpha = {:.3f} 1/m'.format(alpha))

        # spatial grid
        zScale = self._h0  # m
        zStep = zScale/(Nx - 1)
        dx = zStep/zScale
        # x_grid = np.array([i*dx for i in range(Nx)])
        print('spatial stepsize dz = {:.2f} mm'.format(zStep*1000))

        # temporal grid
        tScale = zScale**2/a  # s
        tStep = tScale/(Ny - 1)  # s
        dy = tStep/tScale
        # y_grid = np.array([i*dy for i in range(Ny)])
        print('temporal stepsize dt = {:.2f} s'.format(tStep))
        print('timescale tScale = {:.1f} s'.format(tScale))

        # parameter: sigma = a*dt/(2.*dz**2) = a*(tScale*dy)/(2.*(LScale*dx)**2) = dy/(2.*dx)
        sigma = dy/2./dx**2
        print('sigma = {}'.format(sigma))

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
        Bi = (alphaRad + alphaConv)*zScale/lmbda
        print('qRad = {:.0f} W/m2, alphaRad = {:.0f} W/(m2 K), Bi = {:.2e}'.format(qRad, alphaRad, Bi))

        # sun irradiance (differentiated, non-dimensionalized)
        i0 = I0/(lmbda/self._h0)/(THi - TLo)
        alphah = alpha*self._h0
        print('i0 = {:.3f}'.format(i0))
        print('non-dim irradiance source term at top, bottom = {:.3f}, {:.3f}'.format(
            self.nonDimIrradSource(0, dx, 1, i0, alphah), self.nonDimIrradSource(Nx, dx, 1, i0, alphah))
        )

        # U0 = self._u(0., THi, TAmb)  # BC at x = 1: T = 0 °C
        # UAmb = self._u(TAmb, THi, TAmb)

        # set up matrices
        A = ss.diags([-sigma, 1 + 2*sigma, -sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()
        B = ss.diags([sigma, 1 - 2*sigma, sigma], [-1, 0, 1], shape=(Nx, Nx)).toarray()

        # BC at x = 0
        A[0, 0] = 1 + 2*(1 + Bi*dx)*sigma
        A[0, 1] = -2*sigma
        B[0, 0] = 1 - 2*(1 + Bi*dx)*sigma
        B[0, 1] = 2*sigma

        # BC at x = 1
        # A[Nx-1, Nx-2] = -2*sigma
        # B[Nx-1, Nx-2] = 2*sigma
        # A[Nx-1, Nx-2] = -sigma  # i.e. not different!
        # B[Nx-1, Nx-2] = sigma  # i.e. not different!

        # boundary conditions and source term vector
        # b = np.zeros(len(B))
        b = np.array([self.nonDimIrradSource(j, dx, dy, i0, alphah) for j in range(Nx)])
        b[0] = b[0] + 4*sigma*dx*Bi*self._u(TAmb, THi, TLo)
        b[-1] = b[-1] + 2*sigma*self._u(IC[1], THi, TLo)

        # IC, linear
        TInit = np.linspace(IC[0], IC[1], Nx)
        UInit = self._u(TInit, THi, TLo)

        return A, B, b, UInit

    def nonDimIrradSource(self, j, dx, dy, i0, alphah):
        return i0*alphah*dy*np.exp(-alphah*j*dx)
