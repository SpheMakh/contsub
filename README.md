# contsub
Image-Plane continuum subtraction for FITS cubes

## Installation
```bash
pip install git+https://github.com/SpheMakh/contsub.git
```

## Documentation
```bash
Usage: imcontsub [OPTIONS] INPUT_IMAGE

Options:
  --version                       Show the version and exit.
  --order TEXT                    Order of spline. If given as a list of size
                                  N, then N iterations will be perfomed.
  --segments TEXT                 Width of spline segments in km/s. If given
                                  as a list, then it must have same sixe as
                                  --order.
  --output-prefix PATH            Name of ouput image
  --mask-image PATH               Mask image
  --sigma-clip TEXT               Sigma clip for each iteration. Only required
                                  if mask-image is not given.
  --fit-model [polyn|spline|dct]
  --overwrite / --no-overwrite    Overwrite output image if it already exists
  --stokes-axis / --no-stokes-axis
                                  Set this flag if the input image has a
                                  stokes dimension. (Default is True).
  --help                          Show this message and exit.

```
