import webbpsf
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import numpy
from matplotlib import pyplot as plt
from jax import config
from poppy.zernike import hexike_basis

from optics import NIRISSOptics
from observations import NIRISSFilters
from detector_layers import ApplyBFE, DistortionFromSiaf

config.update("jax_enable_x64", True)
plt.rcParams["image.origin"] = 'lower'

nterms = 10
npix = 512

# Get webbpsf model
niriss = webbpsf.NIRISS()
niriss_osys = niriss.get_optical_system()
seg_cens = niriss_osys.planes[0]._seg_centers_m
pscale = niriss_osys.planes[0].pixelscale.value * 1024 / npix

# Scale mask
pupil = niriss_osys.planes[0].amplitude
mask = dl.utils.scale_array(pupil, npix, 1)

# Hard-coded scaling params
diam = 1.32 * 1.001

# Gen basis
bases = []

for centre in list(seg_cens.values()):
    rhos, thetas = numpy.array(dlu.pixel_coordinates((npix, npix), pscale, offsets=tuple(centre), polar=True))
    bases.append(hexike_basis(10, npix, diam * rhos, thetas, outside=0.))

bases = np.array(bases)
sample_bases = mask*bases.sum(0)

fig, ax = plt.subplots(2, 5, figsize=(15, 8))
for i in range(2):
    for j in range(5):
        bound = np.array([sample_bases[i*5+j].max(), -sample_bases[i*5+j].min()]).max()
        ax[i, j].imshow(sample_bases[i*5+j],
                        cmap='seismic',
                        vmin=-bound,
                        vmax=bound,
                        )

        ax[i, j].axis('off')




# # creating NIRISS object
# webbpsfobj = webbpsf.NIRISS()
# webbpsfobj.calc_psf()  # calculating fits files
# NIS_CEN_aperture = webbpsfobj.siaf.apertures['NIS_CEN']
#
# osys = NIRISSOptics()
# src = dl.PointSource(**dict(np.load('filter_configs/F480M.npz')))
# obs = NIRISSFilters()
# det = dl.LayeredDetector([
#     dl.detector_layers.RotateDetector(-d2r(getattr(NIS_CEN_aperture, "V3IdlYAngle")), order=3),
#     # Rotates PSF by half a degree
#     DistortionFromSiaf(aperture=NIS_CEN_aperture),  # Wavefront sphere to wavefront plane
#     dl.IntegerDownsample(kernel_size=4),  # Downsample to detector pixel scale
#     # ApplyBFE(1e-6),  # Apply BFE
# ])
# instrument = dl.Instrument(optics=osys, sources=src, detector=det, observation=obs)
#
