import os
import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
from typing import Callable, List, Dict, Union
from scabha.basetypes import File, MS
from omegaconf import OmegaConf
import numpy as np
from numpy import ma
from contsub import fitmods, BIN
from contsub.utils import radec2lm, direction_rad
from scabha import init_logger
from tqdm import tqdm

thisdir = os.path.dirname(__file__)

log = init_logger(BIN.uv_plane)

#STOKES_MAP = OmegaConf.load(f"{thisdir}/data/casa_stokes_corr.yaml").STOKES_MAP

def uv_subtract(data, fitopts, flags, weights=None, directions_info:Dict = None, uvw = None):
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
    freqs = fitopts["xvar"]
    
    func = getattr(fitmods, fitmods.FITFUNCS[fitopts["func"]])
    fitfunc = func(*[fitopts["order"], fitopts["segments"]])
    
    def doit(data_, residuals_, label="phase centre"):
        for corr in [corr1, corr2]:
            with tqdm(total=nrow, desc=f"Subtracting continuum from [{label}], correlation: {corr}", unit='rows') as pbar:
                for row in range(nrow):
                    flagrow = flags[row,:,corr]
                    if isinstance(fitopts["mask"], np.ndarray):
                        flagrow = flagrow | fitopts["mask"]
                    # leave data as is if row is flagged
                    if sum(flagrow) == flagrow.size:
                        residuals_[row,:,corr] = data_[row,:,corr]
                        continue
                    profile = data_[row,:,corr]
                    # 2 dimensions => no spectral weights
                    if len(weights.shape) == 2:
                        wt = np.ones_like(profile.real)
                    else:
                        wt = weights[row,:,corr]
                    wt[flagrow] = 0.0

                    cont_re, line_re = fitfunc.fit(freqs, profile.real, weight = wt)
                    cont_im, line_im = fitfunc.fit(freqs, profile.imag, weight = wt)

                    residuals_[row,:,corr] = line_re + 1j*line_im
                    pbar.update(1)
        return residuals_
    
    if directions_info:
        directions = directions_info["directions"]
        # convert to wavelengths
        uvw = uvw.T
        uvwl = uvw[...,None] * freqs/2.99792458e8
        phase = 0
        phase_prev = 0
        for i, (li, mi) in enumerate(directions):
            log.info(f"Phase shifting to direction {i+1}")
            nterm = np.sqrt(1 - li*li - mi*mi)
            phasei = uvwl[0] * li + uvwl[1] * mi + uvwl[2] * nterm
            phase -= phasei
            if i == 0:
                vis = data * np.exp(2j*np.pi * phase[...,None])
            else:
                phase += phase_prev
                vis = residuals * np.exp(2*np.pi * phase[...,None])
            phase_prev = phasei
            
            residuals = doit(vis, residuals, label=f"direction {i+1}")
        # shift back to phase centre
        log.info("Rotating data back to the phase centre")
        vis = residuals * np.exp(2*np.pi * (phase + phase_prev)[...,None])
        # Do MS centre last because data has to be shifted back either way 
        if directions_info["centre"]:
            residuals = doit(vis, residuals)
        else:
            return vis
    else:
        residuals = doit(data, residuals)
            
    return residuals


def ms_subtract(ms: MS, incol:str, outcol:str, field:Union[str,int] = 0, spwid:int = 0,
                weight_column:str = "WEIGHT_SPECTRUM", fitopts:Dict = None,
                subtract_dirs:File = None, chunks=10000):
    """_summary_

    Args:
        ms (MS): _description_
        field (int|str, optional): _description_. Defaults to 0.
    """
    
    field_ds = xds_from_table(f"{ms}::FIELD")[0]
    fields = list(field_ds.NAME.data.compute())
    
    spwds = xds_from_table(f"{ms}::SPECTRAL_WINDOW", 
                        chunks={"row": (1,)})[0]
    # get frequencies and convert to MHz
    fitopts["xvar"] = np.around(spwds.CHAN_FREQ.data[spwid].compute() / 1e6, decimals=7)
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
    
    # ms dataset list
    ms_dsl = xds_from_ms(ms, group_cols=["DATA_DESC_ID", "SCAN_NUMBER"],
                        index_cols=["TIME", "ANTENNA1", "ANTENNA2"],
                        chunks={"row": chunks})
    
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
                                directions_dict, None,
                                ds.UVW.data, ("row", "uvw"),
                                dtype=data.dtype, concatenate=True,
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
    log.info("\n Continuum subtraction finished without errors.\n ")
    
    return 0
    