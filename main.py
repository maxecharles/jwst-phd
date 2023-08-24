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

coefficients = 1e-7 * jr.normal(jr.PRNGKey(0), (7, 6))

osys = _construct_optics(
    planes=planes,
    instrument=webbpsfobj,
    wf_npix=1024,
    oversample=4,
    radial_orders=[0, 1, 2],
    coefficients=np.array(coefficients),
    AMI=True,
)

src = dl.PointSource(**dict(np.load("filter_configs/F480M.npz")))
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
