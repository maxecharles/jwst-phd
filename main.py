from jax import config
import jax.numpy as np
import webbpsf
import matplotlib.pyplot as plt
import dLux as dl
from optics import NIRISSOptics, find_wavelengths, find_diameter

config.update("jax_enable_x64", True)
plt.rcParams["image.origin"] = 'lower'

# creating NIRISS object
NIRISS = webbpsf.NIRISS()

# updating NIRISS configuration
NIRISS.filter = 'F480M'
NIRISS.pupil_mask = 'MASK_NRM'

psfs = NIRISS.calc_psf()  # calculating fits files
webbpsfpsf = psfs[0].data  # PSF Array from WebbPSF
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
dluxpsf = optics.propagate(np.array(wavels), weights=np.array(weights))

# plotting
fig, ax = plt.subplots(1, 3, figsize=(11, 4))
fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.2)

# dLux PSF
c0 = ax[0].imshow(dluxpsf, cmap='magma')
ax[0].set(title=r'$\partial$Lux PSF')
# fig.colorbar(c0, label='Relative Intensity')

# WebbPSF PSF
c1 = ax[1].imshow(webbpsfpsf, cmap='magma')
ax[1].set(title='WebbPSF PSF')
# fig.colorbar(c1, label='Relative Intensity')

# Residuals
residuals = webbpsfpsf - dluxpsf
bounds = np.array([-residuals.min(), residuals.max()])
c2 = ax[2].imshow(residuals, cmap='seismic',
                  vmin=-bounds.max(), vmax=bounds.max())
ax[2].set(title=f'All Close = {np.allclose(webbpsfpsf, dluxpsf)}')
fig.colorbar(c2, label='Residual')

plt.show()


#
# # Make optics
# optical_layers = [
#     dl.CreateWavefront(npix, diameter, 'Angular'),
#     dl.TransmissiveOptic(aperture),
#     dl.NormaliseWavefront(),
#     InvertY(),
#     dl.ApplyBasisOPD(basis),
#     AMI_aperture,
#     dl.AngularMFT(oversample*(det_npix + pad), pixel_scale/oversample)]
#
# # Make detector
# detector_layers = [
#     ApplyJitter(1.5),
#     dl.detectors.Rotate(-d2r(getattr(aper, "V3IdlYAngle"))),
#     DistortionFromSiaf(aper, oversample),
#     dl.IntegerDownsample(oversample),
#     Cut(det_npix),
#     ApplyBFE(1e-6),
#     dl.AddConstant(0)]
#
# # Make instrument
# optics = dl.Optics(optical_layers)
# detector = dl.Detector(detector_layers)
# tel = dl.Instrument(optics, detector=detector, observation=observation)
