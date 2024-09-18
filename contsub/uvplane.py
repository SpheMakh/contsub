import os
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
from typing import Dict, Union, List
from scabha.basetypes import File, MS
from omegaconf import OmegaConf
import numpy as np
from contsub import fitmods, BIN
from contsub.utils import radec2lm, direction_rad
from scabha import init_logger
from tqdm.dask import TqdmCallback
from numpy import ma
from scipy.ndimage import binary_dilation

thisdir = os.path.dirname(__file__)

log = init_logger(BIN.uv_plane)

class UVContSub(object):
    def __init__(self, data:np.ndarray, 
                freqs: np.ndarray,
                uvw:np.ndarray,
                flags:np.ndarray,
                weights:np.ndarray,
                times:np.ndarray,
                ant1:np.ndarray,
                ant2:np.ndarray,
                corrs:List[int],
                fitopts:Dict, 
                directions_info:Dict,
                return_continuum=False) -> int:
        """AI is creating summary for __init__

        Args:
            data (np.ndarray): [description]
            fitopts (Dict): [description]
            corrs (List[int]): [description]
            flags (np.ndarray): [description]
            weights (np.ndarray, optional): [description]. Defaults to None.
            directions_info (Dict, optional): [description]. Defaults to None.
            uvw (np.ndarray, optional): [description]. Defaults to None.
        """
        
        self.data = data
        self.niter = fitopts["niter"]
        self.nrow, self.nchan, self.ncorr = data.shape
        
        # set fit params from fitopts
        self.funcname = fitopts["funcname"]
        self.sigma_clips = fitopts["sigma_clip"]
        for param in "filter_width spline_segment spline_order arpls_ratio arpls_lam arpls_niter".split():
            vals = fitopts.get(param, None)
            if vals:
                setattr(self, param, fitmods.padit(self.niter, vals))
            else:
                setattr(self, param, None)
        self.usermask = fitopts["mask"]
        self.freqs = freqs
        self.corrs = corrs
        self.flags = flags
        self.weights = weights
        self.chanweights = len(weights.shape) == 3
        self.directions_info = directions_info
        self.uvw = uvw
        self.residual = None
        self.continuum = None
        self.ant1 = ant1
        self.ant2 = ant2
        self.times = times
        self.return_continuum = return_continuum
    
    def get_bta_mask(self, corr, dilate=3):
        # time averages for each baseline (aka, baseline time averages)
        data = self.data[...,corr].real
        flags = self.flags[...,corr]
        mdata = ma.array(data, mask=flags)
        mask = fitmods.iterative_clip(ma.mean(mdata, axis=0).data, self.sigma_clips)
        if dilate:
            mask = binary_dilation(mask, iterations=dilate)
            
        rnd = "".join(map(str, np.random.randint(1,9,4)))
        with open(f"btamask_{rnd}.npy", "wb") as stdw:
            np.save(stdw, mask)
            
        return mask
        
    def get_spline_func(self, order, segment, **kw):
        return fitmods.FitBSpline(order, segment, **kw)
        
    def get_medfilter_func(self, filter_width, **kw):
        return fitmods.FitMedFilter(filter_width, **kw)
    
    def get_arpls_func(self, ratio, lam, niter, **kw):
        return fitmods.FitArPLS(ratio, lam, niter, **kw)
    
    def fit_iter(self, profile:np.ndarray, iter:int,
                 weights:np.ndarray=None, mask:np.ndarray=None, **kw):
        if self.funcname == "spline":
            fitfunc = self.get_spline_func(self.spline_order[iter],
                                           self.spline_segment[iter], **kw)
        elif self.funcname == "medfilter":
            fitfunc = self.get_spline_func(self.filter_width[iter], **kw)
        elif self.funcname == "arpls":
            fitfunc = self.get_arpls_func(self.arpls_ratio[iter],
                                        self.arpls_lam[iter], self.arpls_niter[iter])
        else:
            raise RuntimeError(f"Requested continnum fit function '{self.funcname}' is not supported.")
        
        if isinstance(mask, np.ndarray):
            weights[mask] = 0.0
        line_real = fitfunc.fit(self.freqs, profile.real, weight=weights, mask=mask)[-1]
        line_imag = fitfunc.fit(self.freqs, profile.imag, weight=weights, mask=mask)[-1]
        
        residual = line_real + 1j*line_imag
        return residual

    
    def fitall(self):
        residual = np.zeros_like(self.data)
        pid = os.getpid()
        log.info(f"Ingesting new chuck: PID={pid} ")
        for corr in self.corrs:
            avg_mask = self.get_bta_mask(corr=corr)
            for row in range(self.nrow):
                slc = row, slice(None), corr
                profile = self.data[slc]
                rowflags = self.flags[slc]
                if self.chanweights:
                    roweights = self.weights[row]
                else:
                    roweights = np.ones_like(self.freqs)
                mask = rowflags | avg_mask
                for iter in range(self.niter):
                    profile = self.fit_iter(profile, iter, weights=roweights, mask=mask)
                
                residual[slc] = profile
        log.info(f"Finished processing chunk: PID={pid} ")
        return  residual
        

def uv_subtract(data:np.ndarray, freqs: np.ndarray, 
                uvw:np.ndarray, flags:np.ndarray, 
                weights:np.ndarray, times:np.ndarray,
                ant1:np.ndarray, ant2:np.ndarray, corrs: List[int],
                fitopts:Dict, directions_info:Dict = None):
    
    #TODO(Sphe)
    # Add phase rotation stuff to this function)
    
    uvsub = UVContSub(data, freqs,
                uvw,
                flags,
                weights,
                times,
                ant1,
                ant2,
                corrs,
                fitopts,
                directions_info,
                )
    
    return uvsub.fitall()


def ms_subtract(ms: MS, incol:str, outcol:str, field:Union[str,int] = 0, spwid:int = 0, 
                corrs:List[int] = [0,-1],
                weight_column:str = "WEIGHT_SPECTRUM", fitopts:Dict = None,
                subtract_dirs:File = None, chunks=None, subtract_col_first=None):
    """_summary_

    Args:
        ms (MS): _description_
        field (int|str, optional): _description_. Defaults to 0.
    """
    
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    fields = list(field_ds.NAME.data.compute())
    
    spwds = xds_from_table(f"{ms}::SPECTRAL_WINDOW")[0]
    # get frequencies and convert to MHz
    directions_dict = None
    if subtract_dirs:
        ra0, dec0 = field_ds.PHASE_DIR.data.compute()[0][0]
        directions_info = OmegaConf.load(subtract_dirs)
        dirs = directions_info.directions
        log.info("Will perform continnum subtraction at these phase directions: ")
        directions = []
        for i,(ra, dec) in enumerate(dirs):
            ra, dec = direction_rad(ra, dec)
            ra -= ra0
            dec -= dec0
            directions.append(radec2lm(ra0, dec0, ra, dec))
            log.info(f"     Direction {i+1}: {np.rad2deg(ra)}, {np.rad2deg(dec)} Deg")
            
        directions_dict = {
            "centre" : directions_info.centre,
            "directions": directions,
        }
    if isinstance(field, str):
        field = fields.index(field)
    elif isinstance(field, int):
        field = field
        
    ms_dsl = xds_from_ms(ms, group_cols=["FIELD_ID", "DATA_DESC_ID"],
                        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
                        chunks={"row": chunks.row})
    
    base_dims = ("row", "chan", "corr")
    
    if weight_column == "WEIGHT_SPECTRUM":
        if not hasattr(ms_dsl[0], "WEIGHT_SPECTRUM"):
            raise ValueError("Input MS doesn't have a WEIGHT_SPECTRUM column")
        weight_dims = base_dims
    elif weight_column == "WEIGHT":
        if not hasattr(ms_dsl[0], "WEIGHT"):
            raise ValueError("Input MS doesn't have a WEIGHT column")
        weight_dims = "row", "corr"
    
    
    residuals = []
    log.info(f"Ready to process with row chunks: {ms_dsl[0].chunks['row']}")  
    
    for ds in ms_dsl: 

        if subtract_col_first:
            data = getattr(ds, incol).data - getattr(ds, subtract_col_first).data
        else:
            data = getattr(ds, incol).data
            
        residual = da.blockwise(uv_subtract, base_dims, 
                                data, base_dims,
                                spwds.CHAN_FREQ.data[spwid], ("chan",),
                                ds.UVW.data, ("row", "uvw"),
                                ds.FLAG.data, base_dims,
                                getattr(ds, weight_column).data, weight_dims,
                                ds.TIME.data, ("row",),
                                ds.ANTENNA1.data, ("row",),
                                ds.ANTENNA2.data, ("row",),
                                corrs, None,
                                fitopts, None,
                                directions_dict, None,
                                dtype=data.dtype, concatenate=True,
                                )
        
        residuals.append(residual)

    writes = []
    for i, ds in enumerate(ms_dsl):
        ms_dsl[i] = ds.assign(**{
                outcol: ( base_dims, 
                    residuals[i]),
        })
    
        writes.append(xds_to_table(ms_dsl, ms, [outcol]))

    with TqdmCallback(desc="compute"):
        da.compute(writes)
    log.info("\n Continuum subtraction finished without errors.\n ")
    
    return 0
    