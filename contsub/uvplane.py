import os
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
from typing import Callable, List, Dict, Union
from scabha.basetypes import File, MS
from omegaconf import OmegaConf
import numpy as np
from numpy import ma
from contsub import fitmods

thisdir = os.path.dirname(__file__)

STOKES_MAP = OmegaConf.load(f"{thisdir}/data/casa_stokes_corr.yaml").STOKES_MAP

def uv_subtract(data, fitopts, flags, weights=None):
    """_summary_

    Args:
        func (Callble): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    nrow, nchan, ncorr = data.shape
    if ncorr == 2:
        corr1, corr2 = 0,1
    else:
        corr1, corr2 = 0,3
    residuals = np.zeros_like(data)
    
    func = getattr(fitmods, fitmods.FITFUNCS[fitopts["func"]])
    
    fitfunc = func(*[fitopts["order"], fitopts["segments"]])
    for corr in [corr1, corr2]:
        for row in range(nrow):
            flagrow = flags[row,:,corr]
            # leave data as is if row is flagged
            if sum(flagrow) == flagrow.size:
                residuals[row,:,corr] = data[row,:,corr]
                continue
            profile = data[row,:,corr]
            # 2 dimensions => no spectral weights
            if len(weights.shape) == 2:
                wt = np.ones_like(profile.real)
            else:
                wt = weights[row,:,corr]
            wt[flagrow] = 0.0
            
            cont_re, line_re = fitfunc.fit(fitopts["xvar"], profile.real, weight = wt)
            cont_im, line_im = fitfunc.fit(fitopts["xvar"], profile.imag, weight = wt)
            
            residuals[row,:,corr] = line_re + 1j*line_im
            
    return residuals


def ms_subtract(ms: MS, incol:str, outcol:str, field:Union[str,int] = 0, spwid:int = 0,
                weight_column:str = "WEIGHT_SPECTRUM", fitopts:Dict = None):
    """_summary_

    Args:
        ms (MS): _description_
        field (int|str, optional): _description_. Defaults to 0.
    """
    
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    fields = list(field_ds.NAME.data.compute())
    #pol_ds = xds_from_table(ms+"::POLARIZATION")[0]
    #corrs = pol_ds.CORR_TYPE.data.compute()[spwid]
    #corr_names = [STOKES_MAP[int(x)] for x in corrs]
    #ncorrs = len(corr_names)
    
    spwds = xds_from_table(f"{ms}::SPECTRAL_WINDOW", 
                        chunks={"row": (1,)})[0]
    # get frequencies and convert to MHz
    fitopts["xvar"] = np.around(spwds.CHAN_FREQ.data[spwid].compute() / 1e6, decimals=7)
    
    
    if isinstance(field, str):
        field = fields.index(field)
    elif isinstance(field, int):
        field = field
    
    # ms dataset list
    ms_dsl = xds_from_ms(ms, group_cols=["DATA_DESC_ID", "SCAN_NUMBER"],
                        index_cols=["TIME", "ANTENNA1", "ANTENNA2"])
    
    if weight_column == "WEIGHT_SPECTRUM" and not hasattr(ms_dsl[0], "WEIGHT_SPECTRUM"):
            raise ValueError("Input MS doesn't have WEIGHT_SPECTRUM column")
    
    residuals = []    
    for ds in ms_dsl:
        data = getattr(ds, incol).data
        residual = da.blockwise(uv_subtract, ("row", "chan", "corr"), 
                                data, ("row", "chan", "corr"),
                                fitopts, None,
                                ds.FLAG.data, ("row", "chan", "corr"),
                                getattr(ds, weight_column).data, None,
                                dtype=data.dtype,
                                )
        
        residuals.append(residual)
    
    writes = []
    for i, ds in enumerate(ms_dsl):
        ms_dsl[i] = ds.assign(**{
                outcol: ( ("row", "chan", "corr"), 
                        residuals[i]),
            })
        
        writes.append(xds_to_table(ms_dsl, ms, [outcol]))
        
    da.compute(writes)
    
    return 0
    
