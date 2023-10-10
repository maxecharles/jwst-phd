import dLuxWebbpsf as dlW
import webbpsf

from utils import plot_and_compare
from jax import config
from jax import numpy as np

config.update("jax_enable_x64", True)

filt = "F480M"
mask = "MASK_NRM"

niriss_tel = dlW.NIRISS(
    filter=filt,
    pupil_mask=mask,
    aperture="NIS_CEN"
)

# webbpsf
niriss = webbpsf.NIRISS()  # WebbPSF instrument
niriss.filter = filt
niriss.pupil_mask = mask  # applies the NRM mask
psfs = niriss.calc_psf()

plot_and_compare(niriss_tel.model(), psfs[-1].data, titles=['dLux', 'WebbPSF'], colorbars=True, pixel_crop=5)
