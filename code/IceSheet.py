# IceSheet class
# Created 2021-03-21

import os
import json

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

