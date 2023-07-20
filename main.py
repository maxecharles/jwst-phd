from jax import config
import jax.numpy as np
import webbpsf
import matplotlib.pyplot as plt
import dLux as dl
from optics import NIRISSOptics, find_wavelengths, find_diameter
from layers import DistortionFromSiaf, ApplyBFE
from dLux.utils import deg_to_rad as d2r

config.update("jax_enable_x64", True)
plt.rcParams["image.origin"] = 'lower'

# creating NIRISS object
NIRISS = webbpsf.NIRISS()

# updating NIRISS configuration
NIRISS.filter = 'F480M'
NIRISS.pupil_mask = 'MASK_NRM'

psfs = NIRISS.calc_psf()  # calculating fits files
# webbpsfpsf = psfs[0].data  # PSF Array from WebbPSF
webbpsfdetpsf = psfs[2].data  # PSF Array WITH DETECTOR EFFECTS from WebbPSF
AMI_mask = np.array(NIRISS.optsys.planes[3].amplitude)  # transmission array of AMI Mask
diameter = find_diameter(NIRISS.optsys)  # finding JWST diameter

# hardcoded by WebbPSF for a genius reason
det_npix = 304
oversample = 4

# aperture aberrations
aperture = dl.Optic(transmission=NIRISS.optsys.planes[0].amplitude,
                    opd=NIRISS.optsys.planes[0].opd,
                    normalise=True
                    )

# Field dependent aberrations
FDA = dl.Optic(transmission=NIRISS.optsys.planes[2].amplitude,
               opd=NIRISS.optsys.planes[2].opd,
               )

# creating dLux optics
optics = NIRISSOptics(aperture=aperture,
                      FDA=FDA,
                      pupil_mask=AMI_mask,
                      psf_npixels=det_npix,
                      psf_oversample=oversample,
                      )

# Generating PSF with dLux
wavels, weights = find_wavelengths(psfs[0])  # finding wavelengths and spectral weights
# dluxpsf = optics.propagate(np.array(wavels), weights=np.array(weights))

baseline = 0.
# NIS_CEN_aperture = NIRISS.siaf.apertures['NIS_CEN']
siaf_aperture = NIRISS._detector_geom_info.aperture

# Construct Detector
detector = dl.LayeredDetector([
    dl.ApplyJitter(sigma=1.5),  # Gaussian Jitter
    dl.detector_layers.RotateDetector(-d2r(getattr(siaf_aperture, "V3IdlYAngle")), order=3),  # Rotates PSF by half a degree
    DistortionFromSiaf(aperture=siaf_aperture, oversample=oversample),  # Wavefront sphere to wavefront plane
    # dl.IntegerDownsample(kernel_size=oversample),  # Downsample to detector pixel scale
    # ApplyBFE(1e-6),  # Apply BFE TODO fix 'Image' object has no attribute 'shape'
    # dl.AddConstant(baseline),  # Add baseline
])

source = dl.PointSource(wavelengths=wavels, weights=weights)
instrument = dl.Instrument(optics=optics, sources=source, detector=detector)

dluxdetpsf = instrument.model()

# plotting
fig, ax = plt.subplots(1, 3, figsize=(11, 4))
fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.2)
ticks = [0, webbpsfdetpsf.shape[0]-1]
# WebbPSF PSF
c0 = ax[0].imshow(webbpsfdetpsf**.5, cmap='magma')
ax[0].set(title='WebbPSF $\sqrt{PSF}$', xticks=ticks, yticks=ticks)
# fig.colorbar(c0, label='Relative Intensity')

# dLux PSF
c1 = ax[1].imshow(dluxdetpsf**.5, cmap='magma')
ax[1].set(title=r'$\partial$Lux $\sqrt{PSF}$', xticks=ticks, yticks=ticks)
# fig.colorbar(c1, label='Relative Intensity')

# Residuals
residuals = webbpsfdetpsf - dluxdetpsf
bounds = np.array([-residuals.min(), residuals.max()])
c2 = ax[2].imshow(residuals, cmap='seismic',
                  vmin=-bounds.max(), vmax=bounds.max())
ax[2].set(title=f'All Close = {np.allclose(webbpsfdetpsf, dluxdetpsf)}', xticks=ticks, yticks=ticks)
fig.colorbar(c2, label='Residual')

plt.show()