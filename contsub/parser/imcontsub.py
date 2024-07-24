import contsub
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from scabha.basetypes import File
from omegaconf import OmegaConf
import glob
import os
from contsub import BIN
from contsub.cubes import RCube, FitsHeader
from contsub.fitmods import FitBSpline, Mask, PixSigmaClip
from contsub.implane import ContSub
from scabha import init_logger
import astropy.io.fits as fitsio
import numpy as np

log = init_logger(BIN.im_plane)

command = BIN.im_plane
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
    infits = File(opts.input_image)
    
    if opts.output_prefix:
        prefix = opts.output_prefix
    else:
        prefix = f"{infits.BASEPATH}-contsub"
    
    outcont = File(f"{prefix}-cont.fits")
    outline = File(f"{prefix}-line.fits")
    if opts.overwrite is False and (outcont.EXISTS or outline.EXISTS):
        raise RuntimeError("At least one output file exists, but --no-overwrite has been set. Unset it to proceed.")
    
    if not infits.EXISTS:
        raise FileNotFoundError(f"Input FITS image could not be found at: {infits.PATH}")
    # get rid of stokes axis if it exists
    # TODO(sphe) Automate this
    # Needs to fixed later
    if opts.stokes_axis:
        dslice = 0, slice(None), slice(None), slice(None)
    else:
        dslice = slice(None)
        
    if len(opts.order) != len(opts.segments):
        raise RuntimeError("The --order and --segments lists must be the size.")
    niter = len(opts.segments)
    # get Fits image cube primary HDU
    phdu = fitsio.open(opts.input_image)[0]
    header = FitsHeader(phdu.header)
    cube = phdu.data[dslice]
    freqs = header.retFreq()         

    #get the mask for the first round
    nomask = True
    if opts.mask_image:
        log.info("Loading mask image")
        mask = fitsio.getdata(opts.mask_image)[dslice]
        mask_isnan = np.isnan(mask)
        
        if mask_isnan.any():
            mask[mask_isnan] = 0
            mask[~mask_isnan] = 1
        else:
            del mask_isnan
            mask = ~np.array(mask, dtype=bool)
        nomask = False
        
    for i in range(niter):
        
        fitfunc = FitBSpline(*[opts.order[i], opts.segments[i]])
        if nomask:
            if i == 0:
                log.info(f'Creating initial mask from input image')
                contsub = ContSub(freqs, cube, fitfunc)
                cont, line = contsub.fitContinuum()
            
            #create mask from line emission of first iteration
            clip = PixSigmaClip(opts.sigma_clip[i])
            mask = Mask(clip).getMask(line)
            
        log.info(f'Running iteration {i+1}')
        constsub = ContSub(freqs, cube, fitfunc, mask)
        #do the fitting
        cont, line = constsub.fitContinuum()
    log.info("Continuum fitting successful. Ready to write output products.")
        

    phdu._close()
    del cube
        
    outcube = RCube(infits)
    outcube.openR()
    
    log.debug(f'Pixel sums line: {line.sum()}, cont: {cont.sum()}')
    outcube.write_like(cont, outfits=outcont, overwrite=opts.overwrite)
    outcube.write_like(line, outfits=outline, overwrite=opts.overwrite)
    outcube.close()
    