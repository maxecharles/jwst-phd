from __future__ import annotations
import dLux
import dLux.utils as dlu
import jax.numpy as np
from jax import Array
import webbpsf

OpticalLayer = lambda: dLux.OpticalLayer


class NIRISSOptics(dLux.optics.AngularOptics):
    wf_npixels: int
    diameter: float

    aperture: OpticalLayer()
    aberrations: OpticalLayer()
    FDA: OpticalLayer()
    mask: OpticalLayer()

    image_mask: OpticalLayer()
    FFT_pad: int

    psf_pixel_scale: None
    psf_npixels: None
    psf_oversample: None

    def __init__(self,
                 aperture,
                 FDA,
                 pupil_mask,
                 aberrations=None,
                 image_mask=None,
                 FFT_pad=None,
                 psf_npixels=256,
                 psf_oversample=2,
                 ):
        """

        """
        self.diameter = 6.603464
        self.wf_npixels = 1024

        self.aperture = aperture
        self.aberrations = aberrations
        self.FDA = FDA
        self.mask = pupil_mask

        # Coronographic stuff
        self.image_mask = image_mask
        if self.image_mask is None:
            self.FFT_pad = None
        else:
            self.FFT_pad = FFT_pad

        # PSF Properties
        self.psf_pixel_scale = 0.0656  # arcsec/pixel
        self.psf_npixels = psf_npixels
        self.psf_oversample = psf_oversample

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

        # Optional Coronagraphic Mask
        if self.image_mask is not None:
            wf = wf.pad_to(self.FFT_pad)
            wf = wf.FFT()
            wf *= self.image_mask
            wf = wf.IFFT()

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


class SpecificNIRISSOptics(NIRISSOptics):
    filter: str = 'F480M'
    pupil_mask: str = 'MASK_NRM'
    det_npix: int = 1024
    pixel_scale: float

    psfs: list
    planes: list
    mask_transmission: Array

    def __init__(self,
                 aberrations=None,
                 image_mask=None,
                 FFT_pad=None,
                 psf_npixels=256,
                 psf_oversample=4,
                 ):
        NIRISS = webbpsf.NIRISS()
        NIRISS.filter = self.filter
        NIRISS.pupil_mask = self.pupil_mask

        self.pixel_scale = NIRISS.pixelscale
        self.psfs = NIRISS.calc_psf()
        self.planes = NIRISS.optsys.planes
        self.mask_transmission = np.array(self.planes[3].amplitude)

        self.aperture = dLux.Optic(transmission=NIRISS.optsys.planes[0].amplitude,
                                   opd=NIRISS.optsys.planes[0].opd,
                                   normalise=True
                                   )
        self.FDA = dLux.Optic(transmission=NIRISS.optsys.planes[2].amplitude,
                              opd=NIRISS.optsys.planes[2].opd,
                              )

        super().__init__(self.aperture,
                         self.FDA,
                         self.self.mask_transmission,
                         aberrations,
                         image_mask,
                         FFT_pad,
                         psf_npixels,
                         psf_oversample,
                         )

    def get_psf(self):
        """
        """
        # finding wavelengths and spectral weights
        head = self.psfs[0].header
        nwavels = head['NWAVES']
        wavels, weights = [], []
        for i in range(nwavels):
            wavels.append(head['WAVE' + str(i)])
            weights.append(head['WGHT' + str(i)])

        return self.propagate(np.array(wavels), weights=np.array(weights))
