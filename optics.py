from __future__ import annotations
import dLux
import dLux.utils as dlu
import jax.numpy as np
from jax import Array
import webbpsf

OpticalLayer = lambda: dLux.optics.OpticalLayer


class NIRISSOptics(dLux.optics.AngularOptics):
    wf_npixels: int
    diameter: float

    aperture: OpticalLayer()
    aberrations: OpticalLayer()
    FDA: OpticalLayer()
    mask: OpticalLayer()

    psf_pixel_scale: float
    psf_npixels: int
    psf_oversample: float

    def __init__(self,
                 wf_npixels: int = 1024,
                 aperture=None,
                 mask=None,
                 psf_pixel_scale=0.0656,
                 psf_oversample=4,
                 psf_npixels=304,

                 FDA=None,
                 aberrations=None,
                 ):
        """

        """
        NIRISS = webbpsf.NIRISS()
        NIRISS.pupil_mask = 'MASK_NRM'
        NIRISS.calc_psf()

        if aperture is None:
            self.aperture = dLux.Optic(transmission=NIRISS.optsys.planes[0].amplitude,
                                       opd=NIRISS.optsys.planes[0].opd,
                                       normalise=True,
                                       )
        if FDA is None:
            self.FDA = dLux.Optic(transmission=NIRISS.optsys.planes[2].amplitude,
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

    def propagate_mono(self: NIRISSOptics,
                       wavelength: Array,
                       offset: Array = np.zeros(2),
                       return_wf: bool = False) -> Array:
        """

        """
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


def find_wavelengths(PSF):
    head = PSF.header
    nwavels = head['NWAVES']
    wavels, weights = [], []
    for i in range(nwavels):
        wavels.append(head['WAVE' + str(i)])
        weights.append(head['WGHT' + str(i)])
    return np.array(wavels), np.array(weights)


def find_diameter(optical_system):
    pupil_plane = optical_system.planes[0]
    pscale = pupil_plane.pixelscale.to('m/pix').value
    diameter = pscale * pupil_plane.npix
    return diameter
