from __future__ import annotations
import dLux
import dLux.utils as dlu
import jax.numpy as np
from jax import Array
import webbpsf
from bases import generate_jwst_hexike_basis
import dLuxWebbpsf

OpticalLayer = lambda: dLux.optics.OpticalLayer


class JWSTPrimary(dLux.Optic):
    """
    A class used to represent the JWST primary mirror. This is essentially a
    wrapper around the dLux.Optic class that simply enforces the normalisation
    at this plane, inverts the y axis automatically, and is slightly more
    efficient than the native implementation.
    """

    def __init__(
            self: OpticalLayer,
            transmission: Array = None,
            opd: Array = None,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        """
        super().__init__(transmission=transmission, opd=opd, normalise=True)

    def __call__(self, wavefront):
        # Apply transmission and normalise
        amplitude = wavefront.amplitude * self.transmission
        amplitude /= np.linalg.norm(amplitude)

        # Apply phase
        phase = wavefront.phase + wavefront.wavenumber * self.opd

        # Update and return
        return wavefront.set(["amplitude", "phase"], [amplitude, phase])


class JWSTAberratedPrimary(JWSTPrimary, dLux.optical_layers.BasisLayer):
    def __init__(self, basis, coefficients):
        """
        Parameters
        ----------
        basis : Array, metres
            The basis vectors to be used for the OPD.
        coefficients : Array, metres
            The coefficients to be applied to the basis vectors.
        """
        if basis.shape[:2] != coefficients.shape:
            raise ValueError(
                "Basis and coefficients must have the same shape, excluding the "
                "pixel dimensions."
            )

        super().__init__()
        self.basis = basis
        self.coefficients = coefficients

    @property
    def basis_opd(self):
        return self.calculate(self.basis, self.coefficients)

    def __call__(self, wavefront):
        # Apply transmission and normalise
        amplitude = wavefront.amplitude * self.transmission
        amplitude /= np.linalg.norm(amplitude)

        total_opd = self.opd + self.basis_opd

        # Apply phase
        phase = wavefront.phase + wavefront.wavenumber * total_opd

        # Update and return
        return wavefront.set(["amplitude", "phase"], [amplitude, phase])


class NIRISSOptics(dLux.optics.AngularOptics):
    wf_npixels: int
    diameter: float

    aperture: OpticalLayer()
    aberrations: OpticalLayer()
    FDA: OpticalLayer()
    mask: OpticalLayer()

    zernike_coeffs: Array
    Basis: OpticalLayer()

    psf_pixel_scale: float
    psf_npixels: int
    psf_oversample: float

    def __init__(
            self,
            wf_npixels: int = 1024,
            aperture=None,
            mask=None,
            psf_pixel_scale=0.0656,
            psf_oversample=4,
            psf_npixels=304,
            FDA=None,
            aberrations=None,
            zernike_coeffs=None,
            basis=None,
    ):
        """ """
        NIRISS = webbpsf.NIRISS()
        NIRISS.pupil_mask = "MASK_NRM"
        NIRISS.calc_psf()

        if aperture is None:
            self.aperture = dLux.Optic(
                transmission=NIRISS.optsys.planes[0].amplitude,
                opd=NIRISS.optsys.planes[0].opd,
                normalise=True,
            )
        if FDA is None:
            self.FDA = dLux.Optic(
                transmission=NIRISS.optsys.planes[2].amplitude,
                opd=NIRISS.optsys.planes[2].opd,
            )
        if mask is None:
            self.mask = np.array(NIRISS.optsys.planes[3].amplitude)

        self.diameter = find_diameter(NIRISS.optsys)  # finding JWST diameter

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=self.diameter,
            aperture=self.aperture,
            mask=self.mask,
            psf_pixel_scale=psf_pixel_scale,
            psf_oversample=psf_oversample,
            psf_npixels=psf_npixels,
        )

        self.aberrations = aberrations
        self.wf_npixels = wf_npixels

        if zernike_coeffs is not None:
            self.zernike_coeffs = zernike_coeffs
        else:
            self.zernike_coeffs = np.zeros(36)

        if basis is None:
            self.Basis = generate_jwst_hexike_basis(npix=wf_npixels, AMI=True)
        else:
            self.Basis = basis

    def propagate_mono(
            self: NIRISSOptics,
            wavelength: Array,
            offset: Array = np.zeros(2),
            return_wf: bool = False,
    ) -> Array:
        """ """
        # Create Wavefront and Tilt
        wf = dLux.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply Aperture and aberrations
        wf *= self.aperture
        wf += self.aberrations

        # Flip and apply FDA
        wf = wf.flip(0)
        wf += self.FDA

        # Apply Pupil Mask
        if wf.npixels != self.wf_npixels:
            wf = wf.crop_to(self.wf_npixels)
        wf *= self.mask

        # Propagate to detector
        pixel_scale = self.psf_pixel_scale / self.psf_oversample
        pixel_scale_radians = dlu.arcsec_to_rad(pixel_scale)
        wf = wf.MFT(self.psf_npixels, pixel_scale_radians)
        return wf.psf


class NIRISSAMIOptics(dLux.optics.LayeredOptics):

    def __init__(
            self,
            radial_orders: Array = None,
            hexike_coeffs: Array = None,
            wf_npixels: int = 1024,
            oversample: int = 4,
            wss_date: str = None,
    ):
        niriss = webbpsf.NIRISS()  # WebbPSF instrument

        niriss.pupil_mask = "MASK_NRM"  # applies the NRM mask -- remove this line for full pupil simulation
        if wss_date is not None:
            niriss.load_wss_opd_by_date(date=wss_date, verbose=False)  # loads the WSS OPD map for the given date
        niriss.calc_psf()
        niriss_osys = niriss.get_optical_system()

        planes = niriss_osys.planes  # [Linear Model WSS, Coord inversion, FDA, NRM, Detector]
        pscale = planes[-1].pixelscale.to("arcsec/pix").value  # arcsec/pix, grabbing pixel scale from detector plane

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
            layers=[
                (dLuxWebbpsf.JWSTAberratedPrimary(
                    planes[0].amplitude,  # transmission
                    planes[0].opd,  # WSS opd data
                    radial_orders=radial_orders,
                    coefficients=hexike_coeffs,
                    AMI=True,  # FALSE FOR FULL PUPIL
                ), "Pupil"),
                (dLux.Flip(0), "InvertY"),
                (dLux.Optic(planes[-2].amplitude), "Mask"),
                (dLux.MFT(
                    npixels=oversample * 64,
                    pixel_scale=dlu.arcsec_to_rad(pscale) / oversample),
                "Propagator"),
            ]
        )


def find_wavelengths(PSF):
    head = PSF.header
    nwavels = head["NWAVES"]
    wavels, weights = [], []
    for i in range(nwavels):
        wavels.append(head["WAVE" + str(i)])
        weights.append(head["WGHT" + str(i)])
    return np.array(wavels), np.array(weights)


def find_diameter(optical_system):
    pupil_plane = optical_system.planes[0]
    pscale = pupil_plane.pixelscale.to("m/pix").value
    diameter = pscale * pupil_plane.npix
    return diameter


def _construct_optics(
        # self,
        planes,
        instrument,
        wf_npix,
        oversample=4,
        clean=False,
        **kwargs,
):
    """Constructs an optics object for the instrument."""

    # Primary mirror - note this class automatically flips about the y-axis
    layers = [
        (
            dLuxWebbpsf.optical_layers.JWSTAberratedPrimary(
                planes[0].amplitude,
                planes[0].opd,
                npix=wf_npix,
                **kwargs
            ),
            "pupil",
        ),
        (dLux.Flip((0, 1)), "InvertXY"),
    ]

    # If 'clean', then we don't want to apply pre-calc'd aberrations
    if not clean:
        # NOTE: We keep amplitude here because the circumscribed circle clips
        # the edge of the primary mirrors, see:
        # https://github.com/spacetelescope/webbpsf/issues/667, Long term we
        # are unlikely to want this.
        FDA = dLux.Optic(planes[2].amplitude, planes[2].opd)
        layers.append((FDA, "aberrations"))

    # Index this from the end of the array since it will always be the second
    # last plane in the osys. Note this assumes there is no OPD in that optic.
    # We need this logic here since MASK_NRM plane has an amplitude, where
    # as the CLEARP plane has a transmission... for some reason.
    pupil_plane = planes[-2]
    if instrument.pupil_mask == "CLEARP":
        layer = dLux.Optic(pupil_plane.transmission)
    elif instrument.pupil_mask == "MASK_NRM":
        layer = dLux.Optic(pupil_plane.amplitude)
    else:
        raise NotImplementedError("Only CLEARP and MASK_NRM are supported.")
    layers.append((layer, "pupil_mask"))

    pscale = (planes[-1].pixelscale).to("arcsec/pix").value
    layers.append(
        dLuxWebbpsf.MFT(npixels=oversample * 64, oversample=oversample, pixel_scale=pscale)
    )

    # Finally, construct the actual Optics object
    return dLux.LayeredOptics(
        wf_npixels=wf_npix,
        diameter=planes[0].pixelscale.to("m/pix").value * planes[0].npix,
        layers=layers,
    )
