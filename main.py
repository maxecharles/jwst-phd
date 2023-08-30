from bases import generate_jwst_hexike_basis
from optics import JWSTAberratedPrimary
import jax.random as jr
import matplotlib.pyplot as plt
from dLuxWebbpsf.instruments import NIRISS
from webbpsf import constants as const
import dLuxWebbpsf as dlW
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import webbpsf
from dLux.utils import deg_to_rad as d2r
from detector_layers import DistortionFromSiaf, ApplyBFE
from optics import _construct_optics

# Primary mirror - note this class automatically flips about the y-axis
webbpsfobj = webbpsf.NIRISS()
webbpsfobj.calc_psf()  # calculating fits files
webbpsfobj.pupil_mask = "MASK_NRM"
NIS_CEN_aperture = webbpsfobj.siaf.apertures["NIS_CEN"]
webbpsf_osys = webbpsfobj.get_optical_system()
planes = webbpsf_osys.planes

radial_orders = np.array([0, 1, 2], dtype=int)
secondary_radial_orders = np.array([2, 3], dtype=int)
hexike_shape = (7, int(np.sum(np.array([dl.utils.triangular_number(i+1) - dl.utils.triangular_number(i) for i in radial_orders]))))
secondary_shape = (int(np.sum(np.array([dl.utils.triangular_number(i+1) - dl.utils.triangular_number(i) for i in secondary_radial_orders]))),)

true_flux = 1e6
true_position = dlu.arcsec_to_rad(0.3*jr.normal(jr.PRNGKey(0), (2,)))
true_coeffs = 2e-7 * jr.normal(jr.PRNGKey(0), hexike_shape)
true_secondary_coeffs = 1e-7 * jr.normal(jr.PRNGKey(0), secondary_shape)

oversample = 4
pscale = (planes[-1].pixelscale).to("arcsec/pix").value
pupil_plane = planes[-2]

osys = dl.LayeredOptics(
    wf_npixels=1024,
    diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
    layers=[
        (dlW.optical_layers.JWSTAberratedPrimary(
            planes[0].amplitude,
            planes[0].opd,
            radial_orders=radial_orders,
            coefficients=true_coeffs,
            secondary_radial_orders=secondary_radial_orders,
            secondary_coefficients=true_secondary_coeffs,
            AMI=True,
        ), "Primary"),
        (dl.Flip((0, 1)), "InvertXY"),
        (dl.Optic(planes[2].amplitude, planes[2].opd), "PriorOPD"),
        (dl.Optic(pupil_plane.amplitude), "Mask"),
        (dlW.MFT(npixels=oversample * 64, oversample=oversample, pixel_scale=pscale), "Propagator"),
    ]
)

src = dl.PointSource(position=true_position, flux=true_flux, **dict(np.load("filter_configs/F480M.npz")))
detector = dl.LayeredDetector(
    [
        dlW.detector_layers.Rotate(-d2r(getattr(NIS_CEN_aperture, "V3IdlYAngle"))),
        DistortionFromSiaf(
            aperture=NIS_CEN_aperture
        ),  # TODO implement dLuxWebbpsf version
        dl.IntegerDownsample(kernel_size=4),  # Downsample to detector pixel scale
    ]
)

instrument = dl.Instrument(sources=[src], detector=detector, optics=osys)

plt.imshow(instrument.model())
plt.colorbar()
