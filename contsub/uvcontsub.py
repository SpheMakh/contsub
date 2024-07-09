import dask
import dask.array as da
from daskms import xds_from_ms, xds_to_table, xds_from_table
from typing import Callable, List, Dict, Union
from scabha.basetypes import File, MS
from contsub.imcontsub import FitBSpline, ContSub, Mask, PixSigmaClip
from omegaconf import OmegaConf
import numpy as np

thisdir = os.path.dirname(__file__)

STOKES_MAP = OmegaConf.load(f"{thisdir}/casa_stokes_corr.yaml")

def uv_subtract(func:Callble, data, *args, **kwargs):
    """_summary_

    Args:
        func (Callble): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    return residual

def _chunk(x, keepdims, axis):
    return x

def _combine(x, keepdims, axis):
    if isinstance(x, list):
        return np.vstack(x, axis=axis)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Invalid type %s" % type(x))


def _agregate(x, keepdims, axis):
    return _combine(x, keepdims, axis)


def ms_subtract(ms: MS, column:str, field:Union[str,int] = 0):
    """_summary_

    Args:
        ms (MS): _description_
        field (int|str, optional): _description_. Defaults to 0.
    """
    
    field_ds = xds_from_table(ms+"::FIELD")[0]
    pol_ds = xds_from_table(ms+"::POLARIZATION")[0]
    corr_names = [STOKES_MAP[corr] for corr in list(pol_ds.CORR_TYPE.data.compute()[0])]
    ncorrs = len(corr_names)
    
    if isinstance(field, str):
        field_id = list(field_ds.NAME.data).index(field)
    elif isinstance(field, int):
        field_id = field
    
    ms_ds = xds_from_ms(ms, columns=[column], group_cols=["DATA_DESC_ID", "SCAN_NUMBER"])
    
    residual_futures = []    
    for ds in ms_ds:
        blocks = da.blockwise(uv_subtract, ("row",),
                                    corr_ids, ("corr",),
                                    getattr(ds, column).data, ("row", "chan", "corr"),
                                    adjust_chunks={"corr": ncorrs},
                                    dtype=numpy.ndarray)

        redux = da.reduction(flag_sums,
                                chunk=_chunk,
                                combine=_combine,
                                aggregate=_aggregate,
                                concatenate=False,
                                dtype=numpy.float64)
        
        residual_futures.append(redux)

    residuals = dask.compute(residual_futures)[0]
    
    

    