from jax import config
import jax.numpy as np
import webbpsf
import matplotlib.pyplot as plt
import dLux as dl
from optics import NIRISSOptics, find_wavelengths, find_diameter
from detector_layers import DistortionFromSiaf, ApplyBFE
from dLux.utils import deg_to_rad as d2r

config.update("jax_enable_x64", True)
plt.rcParams["image.origin"] = 'lower'


def plot_and_compare(PSF1, PSF2, titles=None, pixel_crop: int = None, save_fig: bool = False):
    if titles is None:
        titles = ['WebbPSF $\sqrt{PSF}$', r'$\partial$Lux $\sqrt{PSF}$']

    if pixel_crop is not None:
        PSF1 = PSF1[pixel_crop:-pixel_crop, pixel_crop:-pixel_crop]
        PSF2 = PSF2[pixel_crop:-pixel_crop, pixel_crop:-pixel_crop]

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.2)
    ticks = [0, PSF1.shape[0] - 1]
    # WebbPSF PSF
    c0 = ax[0].imshow(PSF1 ** .5, cmap='magma')
    ax[0].set(title=titles[0], xticks=ticks, yticks=ticks)
    # fig.colorbar(c0, label='Relative Intensity')

    # dLux PSF
    c1 = ax[1].imshow(PSF2 ** .5, cmap='magma')
    ax[1].set(title=titles[1], xticks=ticks, yticks=ticks)
    # fig.colorbar(c1, label='Relative Intensity')

    # Residuals
    residuals = PSF1 - PSF2
    bounds = np.array([-residuals.min(), residuals.max()])
    c2 = ax[2].imshow(residuals, cmap='seismic',
                      vmin=-bounds.max(), vmax=bounds.max())
    ax[2].set(title=f'Residuals', xticks=ticks, yticks=ticks)
    fig.colorbar(c2, label='Residual')

    if save_fig: plt.savefig('psfs.pdf', dpi=400, bbox_inches='tight')
    plt.show()


# creating NIRISS object
NIRISS = webbpsf.NIRISS()

# updating NIRISS configuration
NIRISS.filter = 'F480M'
NIRISS.pupil_mask = 'MASK_NRM'

psfs = NIRISS.calc_psf()  # calculating fits files
webbpsfdetpsf = psfs[3].data  # PSF Array WITH DETECTOR EFFECTS from WebbPSF

# creating dLux optics
optics = NIRISSOptics()

baseline = 0.
# NIS_CEN_aperture = NIRISS.siaf.apertures['NIS_CEN']
siaf_aperture = NIRISS._detector_geom_info.aperture

# Construct Detector
detector = dl.LayeredDetector([
    # dl.ApplyJitter(sigma=1.5),  # Gaussian Jitter
    dl.detector_layers.RotateDetector(-d2r(getattr(siaf_aperture, "V3IdlYAngle")), order=3),  # Rotates PSF
    DistortionFromSiaf(aperture=siaf_aperture, oversample=optics.psf_oversample),  # Wavefront sphere to wavefront plane
    dl.IntegerDownsample(kernel_size=int(optics.psf_oversample)),  # Downsample to detector pixel scale
    # ApplyBFE(1e-6),  # Apply BFE TODO fix 'Image' object has no attribute 'shape'
    # dl.AddConstant(baseline),  # Add baseline
])

filter_config = np.load(f'filter_configs/{NIRISS.filter}.npz')
source = dl.PointSource(wavelengths=filter_config['wavels'], weights=filter_config['weights'])
instrument = dl.Instrument(optics=optics, sources=source, detector=detector)

dluxdetpsf = instrument.model()

# plotting
plot_and_compare(webbpsfdetpsf, dluxdetpsf, pixel_crop=5)
