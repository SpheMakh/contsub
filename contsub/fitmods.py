import numpy as np
from scipy.interpolate import splev, splrep
from scipy.signal import convolve
from scipy import ndimage
import sys
from scabha import init_logger
from abc import ABC, abstractmethod
from . import BIN
from omegaconf import OmegaConf
from typing import List

from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass
from scipy.interpolate import make_smoothing_spline

np.seterr(all='raise') 



log = init_logger(BIN.main)


class alreadyOpen(Exception):
    pass

class alreadyClosed(Exception):
    pass

class noBeamTable(Exception):
    pass

class tableExists(Exception):
    pass

class tableDimMismatch(Exception):
    pass

class CubeDimIsSmall(Exception):
    pass

class FitFunc(ABC):
    """
    abstract class for writing fitting functions
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def prepare(self, x, data, mask, weight):
        pass
    
    @abstractmethod
    def fit(self, x, data, mask, weight):
        pass
    
class FitBSpline(FitFunc):
    """
    BSpline fitting function based on `splev`, `splrep` in `scipy.interpolate` 
    """
    def __init__(self, order, velWidth, randomState=None, seq=None):
        """
        needs to know the order of the spline and the number of knots
        """
        self._order = order
        self._velwid = velWidth
        if randomState and seq:
            rs = np.random.SeedSequence(entropy = randomState, spawn_key = (seq,))
        else:
            rs = np.random.SeedSequence()
        self.rng = np.random.default_rng(rs)
        
    def prepare(self, x):
        msort = np.argpartition(x, -2)
        m1l, m2l = msort[-2:]
        m1h, m2h = msort[:2]
        if np.abs(m1l - m2l) == 1 and np.abs(m1h - m2h) == 1:
            dvl = np.abs(x[m1l]-x[m2l])/np.mean([x[m1l],x[m2l]])*3e5
            dvh = np.abs(x[m1h]-x[m2h])/np.mean([x[m1h],x[m2h]])*3e5
            dv = (dvl+dvh)/2
            self._imax = int(len(x)/(self._velwid//dv))+1
            log.debug('nchan = {}, dv = {}, {}km/s in chans: {}, max order spline = {}'.format(len(x), dv, self._velwid, self._velwid//dv, self._imax))
        else:
            raise RuntimeError('The frequency values are not changing monotonically, aborting')
        
        knotind = np.linspace(0, len(x), self._imax, dtype = int)[1:-1]
        chwid = (len(x)//self._imax)//8
        self._knots = lambda: self.rng.integers(-chwid, chwid, size = knotind.shape)+knotind
    
    def fit(self, x, data, mask=None, weight=None):
        """
        returns the spline fit and the residuals from the fit
        
        x : x values for the fit
        y : values to be fit by spline
        mask : a mask
        weights : weights for fitting the Spline (not implemented), using mask as weight
        """
        self.prepare(x)
        inds = self._knots()
        # use mask as weights if weight not set
        if not isinstance(weight, np.ndarray):
            weight = mask
        splCfs = splrep(x, data, task = -1, w = weight, t = x[inds], k = self._order)
        spl = splev(x, splCfs)
        return spl, data-spl

class FitMedFilter(FitFunc):
    """
    Median filtering class for continuum subtraction 
    """
    def __init__(self, velWidth):
        """
        needs to know the order of the spline and the number of knots
        """
        self._velwid = velWidth
        
    def prepare(self, x, data = None, mask = None, weight = None):
        msort = np.argpartition(x, -2)
        m1l, m2l = msort[-2:]
        m1h, m2h = msort[:2]
        if np.abs(m1l - m2l) == 1 and np.abs(m1h - m2h) == 1:
            dvl = np.abs(x[m1l]-x[m2l])/np.mean([x[m1l],x[m2l]])*3e5
            dvh = np.abs(x[m1h]-x[m2h])/np.mean([x[m1h],x[m2h]])*3e5
            dv = (dvl+dvh)/2
            self._imax = int(self._velwid//dv)
            if self._imax %2 == 0:
                self._imax += 1
            log.debug('len(x) = {}, dv = {}, {}km/s in chans: {}'.format(len(x), dv, self._velwid, self._velwid//dv))
        else:
            log.info('probably x values are not changing monotonically, aborting')
            sys.exit(1)
            
    
    def fit(self, x, data, mask, weight):
        """
        returns the median filtered data as line emission
        
        x : x values for the fit
        y : values to be fit
        mask : a mask (not implemented really)
        weight : weights
        """
        cp_data = np.copy(data)
        if not (mask is None):
            data[np.logical_not(mask)] = np.nan
        nandata = np.hstack((np.full(self._imax//2, np.nan), data, np.full(self._imax//2, np.nan)))
        nanMed = np.nanmedian(np.lib.stride_tricks.sliding_window_view(nandata,self._imax), axis = 1)
        # resMed = nanMed[~np.isnan(nanMed)]
        resMed = nanMed
        return resMed, cp_data-resMed


class Mask():
    """
    mask class creates a mask using a specific masking method
    """
    def __init__(self, method):
        """
        method should be defined when creating a Mask object
        Method should be built on the ClipMethod class
        """
        self.method = method
        
    def getMask(self, data):
        """
        calculates the mask given the data
        """
        return self.method.createMask(data)
        
class ClipMethod(ABC):
    """
    Abstract class for different methods of making masks
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def createMask(self, data):
        pass
    
class PixSigmaClip(ClipMethod):
    """
    simple sigma clipping class
    """
    def __init__(self, n, sm_kernel = None, dilation = 0, method = 'rms'):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma
        
        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        """
        self.n = n
        self.dilate = dilation
        if sm_kernel is None:
            self.sm = None
        else:
            sm_kernel = np.array(sm_kernel)
            if len(sm_kernel.shape) == 1:
                self.sm = sm_kernel[:, None, None]
            else:
                self.sm = sm_kernel
        if method == 'rms':
            self.function = self.__rms()
        elif method == 'mad':
            self.function = self.__mad()
        
    def createMask(self, data):
        """
        calculate a mask from the given data 
        """
        sm_data = self.__smooth(data)
        sigma = self.function(sm_data)
        mask = np.abs(sm_data) < self.n*sigma
        
        struct_dil = ndimage.generate_binary_structure(len(data.shape), 1)
        struct_erd = ndimage.generate_binary_structure(len(data.shape), 2)
        
        for i in range(self.dilate):
            mask = ndimage.binary_dilation(mask, structure=struct_dil, border_value=1).astype(mask.dtype)
            
        for i in range(self.dilate+2):
            mask = ndimage.binary_erosion(mask, structure=struct_erd, border_value=1).astype(mask.dtype)
            
        return mask
    
    def __smooth(self, data):
        if self.sm is None:
            return data
        else:
            sm_data = convolve(data, self.sm, mode = 'same')
            return sm_data
    
    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis = (0)))
    
    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x)-x), axis = (0))
        
class ChanSigmaClip(ClipMethod):
    """
    simple sigma clipping class
    """
    def __init__(self, n, method = 'rms'):
        """
        has to define the multiple of sigma for clipping and the method for calculating the sigma
        
        n : multiple of sigma for clipping
        method : 'rms' or 'mad' for calculating the rms
        """
        self.n = n
        if method == 'rms':
            self.function = self.__rms()
        elif method == 'mad':
            self.function = self.__mad()
        
    def createMask(self, data):
        """
        calculate a mask from the given data 
        """
        sigma = self.function(data)[:,None,None]
        return np.abs(data) < self.n*sigma
    
    def __rms(self):
        return lambda x: np.sqrt(np.nanmean(np.square(x), axis = (1,2)))
    
    def __mad(self):
        return lambda x: np.nanmedian(np.abs(np.nanmean(x)-x), axis = (1,2))


def baseline_arPLS(data, ratio=1e-6, lam=100, niter=10, weights=None, full_output=False):
    size = len(data)

    diag = np.ones(size - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], size, size - 2)

    H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252

    if not isinstance(weights, np.ndarray):
        weights = np.ones_like(data)
        
    weights_mat = sparse.spdiags(weights, 0, size, size)

    crit = 1
    count = 0
    sigma, mean = 1,1
    while crit > ratio:
        bsln = linalg.spsolve(weights_mat + H, weights_mat * data)
        resid = data - bsln
        #dn = resid[resid < 0]

        mean = np.mean(data)
        sigma = np.std(data)
        
        w_new = 1 / (1 + np.exp(2 * (resid - (2*sigma - mean))/sigma))
        
        crit = norm(w_new - weights) / norm(weights)
            

        weights = w_new
        weights_mat.setdiag(weights)  # Do not create a new matrix, just update diagonal values

        count += 1

        if count > niter:
            log.debug('Maximum number of iterations exceeded')
            break

    if full_output:
        info = {'num_iter': count, 'stop_criterion': crit}
        return bsln, resid, info
    else:
        return bsln

@dataclass
class FitArPLS:
    ratio:float = 1e-6
    lam:float = 1e3
    niter:int = 4
    
    def fit(self, freqs, data, mask=None, weight=None):
        if not isinstance(weight, np.ndarray):
            weight = np.ones_like(freqs)
            
        if not isinstance(mask, np.ndarray):
            mask = np.zeros_like(freqs, dtype=bool)
            
        weight[mask] = 0.0 
        baseline = baseline_arPLS(data, ratio=self.ratio,
                                lam=self.lam, niter=self.niter, weights=weight)
        
        return baseline, data - baseline


FITFUNCS = OmegaConf.create({
    "spline": "FitBSpline",
    "medfilter": "FitMedFilter",
})


def iterative_clip(data:np.ndarray, clips:List[float]=[8,5,4]) -> np.ndarray:
    """AI is creating summary for iterative_clip

    Args:
        data ([type]): [description]
        clips (List[float], optional): [description]. Defaults to [8,5,4].
    """
    mask = np.zeros_like(data, dtype=bool)
    for clip in clips:
        mean = np.nanmean(data)
        median = np.nanmedian(data)
        sigma = np.nanmedian( np.absolute(mean - data) )
        thresh = median + sigma*clip
        mask = mask | (np.absolute(data) > thresh)
        data[mask] = np.nan

    return mask


def padit(niter, vals):
    nvals = len(vals)
    if nvals < niter:
        return np.pad(vals, (0,niter - nvals), mode="edge")
    else:
        return vals[:niter]
