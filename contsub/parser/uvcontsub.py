import numpy as np
import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File, MS
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from scabha import init_logger
from contsub.uvplane import ms_subtract

log = init_logger(BIN.uv_plane)

command = BIN.uv_plane
thisdir  = os.path.dirname(__file__)
source_files = glob.glob(f"{thisdir}/library/*.yaml")
sources = [File(item) for item in source_files]
parserfile = File(f"{thisdir}/{command}.yaml")
config = paramfile_loader(parserfile, sources)[command]

@click.command(command)
@click.version_option(str(contsub.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    ms = MS(opts.ms)
    
    if opts.freq_mask:
        if opts.freq_mask.endswith(".npy"):
            mask = np.load(opts.freq_mask)
        elif opts.freq_mask.endswith(".npz"):
            with np.load(opts.freq_mask) as stdr:
                try:
                    mask = stdr[opts.freq_mask_key]
                except KeyError:
                    raise click.BadOptionUsage("freq_mask_key",
                        f"Compressed mask file '{opts.freq_mask}' "
                        f"does not have key '{opts.freq_mask_key}'. See option --freq-mask-key ")
        else:
            raise click.BadOptionUsage("numpy mask file must be either a .npy or .npz file.")
                
    else:
        mask = None
    
    fitopts = {
        "funcname": "spline",
        "spline_order": opts.spline_order,
        "spline_segment": opts.spline_segment,
        "sigma_clip": opts.sigma_clip,
        "xvar": None,
        "filter_width": opts.filter_width,
        "mask": mask,
        "niter": opts.niter
    }
    
    chunks = OmegaConf.create({
        "row": opts.chunks,
        "time": opts.time_chunks,
        "baseline": opts.baseline_chunks 
    })
    
    
    ms_subtract(ms, incol=opts.incol, outcol=opts.outcol, field=opts.field,
                spwid=opts.spw, weight_column=opts.weight_column,
                fitopts=fitopts, subtract_dirs=opts.subtract_dirs, chunks=chunks,
                subtract_col_first=opts.subtract_col_first, corrs=opts.correlations)
    
    
    