import numpy as np
from scabha import init_logger
from . import BIN

log = init_logger(BIN.im_plane)

class ContSub():
    """
    a class for performing continuum subtraction on data
    """
    def __init__(self, x, cube, function, mask=None):
        """
        each object can be initiliazed by passing a data cube, a fitting function, and a mask
        cube : a fits cube containing the data
        function : a fitting function should be built on FitFunc class
        mask : a fitting mask where the pixels that should be used for fitting has a `True` value
        """
        self.cube = cube
        self.function = function
        if mask is None:
            self.mask = None
        else:
            self.mask = np.array(mask, dtype = bool)
        self.x = x
        
    def fitContinuum(self):
        """
        fits the data with the desired function and returns the continuum and the line
        """
        dimy, dimx = self.cube.shape[-2:]
        cont = np.zeros(self.cube.shape)
        line = np.zeros(self.cube.shape)
        self.function.prepare(self.x)
            
        if self.mask is None:
            for i in range(dimx):
                for j in range(dimy):
                    cont[:,j,i], line[:,j,i] = self.function.fit(self.x, self.cube[:,j,i], mask = None, weight = None)
        else:
            for i in range(dimx):
                for j in range(dimy):
                    cont[:,j,i], line[:,j,i] = self.function.fit(self.x, self.cube[:,j,i], mask = self.mask[:,j,i], weight = None)
                
        return cont, line
                