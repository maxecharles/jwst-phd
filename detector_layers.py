from jax import numpy as np
from jax import vmap, jit
from jax.scipy.signal import convolve, correlate
from jax.scipy.ndimage import map_coordinates
from functools import partial

import dLux as dl
import dLux
# from dLux.utils import get_pixel_positions
# from dLux.optics import OpticalLayer
from dLux.detector_layers import DetectorLayer

Array = np.ndarray


"""
Distortions
"""


class DistortionFromSiaf:
    def __new__(cls, aperture, oversample=4):
        degree = aperture.Sci2IdlDeg + 1
        coeffs_dict = aperture.get_polynomial_coefficients()
        coeffs = np.array([coeffs_dict['Sci2IdlX'], coeffs_dict['Sci2IdlY']])
        sci_refs = np.array([aperture.XSciRef, aperture.YSciRef])
        sci_cens = np.array([aperture.XSciRef, aperture.YSciRef])  # Note this may not be foolproof
        pixelscale = 4*0.0164  # np.array([aperture.XSciScale, aperture.YSciScale]).mean()  # this may not be foolproof
        return ApplySiafDistortion(degree, coeffs, sci_refs, sci_cens, pixelscale / oversample, oversample)


class ApplySiafDistortion(DetectorLayer):
    """
    Applies Science to Ideal distortion following webbpsf/pysaif
    """
    degree: int
    Sci2Idl: float
    SciRef: float
    sci_cen: float
    pixel_scale: float
    oversample: int
    xpows: Array
    ypows: Array

    def __init__(self,
                 degree,
                 Sci2Idl,
                 SciRef,
                 sci_cen,
                 pixel_scale,
                 oversample):
        super().__init__()
        self.degree = int(degree)
        self.Sci2Idl = np.array(Sci2Idl, dtype=float)
        self.SciRef = np.array(SciRef, dtype=float)
        self.sci_cen = np.array(sci_cen, dtype=float)
        self.pixel_scale = np.array(pixel_scale, dtype=float)
        self.oversample = int(oversample)
        self.xpows, self.ypows = self.get_pows()

    def get_pows(self):
        n = self.triangular_number(self.degree)
        vals = np.arange(n)

        # Ypows
        tris = self.triangular_number(np.arange(self.degree))
        ydiffs = np.repeat(tris, np.arange(1, self.degree + 1))
        ypows = vals - ydiffs

        # Xpows
        tris = self.triangular_number(np.arange(1, self.degree + 1))
        xdiffs = np.repeat(n - np.flip(tris), np.arange(self.degree, 0, -1))
        xpows = np.flip(vals - xdiffs)

        return xpows, ypows

    def __call__(self, image):
        """
        
        """
        new_image = self.apply_Sci2Idl_distortion(image.image)
        return image.set('image', new_image)
        # return image_out

    def apply_Sci2Idl_distortion(self, image):
        """
        Applies the distortion from the science (i.e. images) frame to the idealised telescope frame
        """

        # Convert sci cen to idl frame
        xidl_cen, yidl_cen = self.distort_coords(self.Sci2Idl[0],
                                                 self.Sci2Idl[1],
                                                 self.sci_cen[0] - self.SciRef[0],
                                                 self.sci_cen[1] - self.SciRef[1])

        # Get paraxial pixel coordinates and detector properties.
        nx, ny = image.shape
        nx_half, ny_half = ((nx - 1) / 2., (ny - 1) / 2.)
        xlin = np.linspace(-1 * nx_half, nx_half, nx)
        ylin = np.linspace(-1 * ny_half, ny_half, ny)
        xarr, yarr = np.meshgrid(xlin, ylin)

        # Scale and shift coordinate arrays to 'sci' frame
        xnew = xarr / self.oversample + self.sci_cen[0]
        ynew = yarr / self.oversample + self.sci_cen[1]

        # Convert requested coordinates to 'idl' coordinates
        xnew_idl, ynew_idl = self.distort_coords(self.Sci2Idl[0],
                                                 self.Sci2Idl[1],
                                                 xnew - self.SciRef[0],
                                                 ynew - self.SciRef[1])

        # Create interpolation coordinates
        centre = (xnew_idl.shape[0] - 1) / 2

        coords_distort = (np.array([ynew_idl - yidl_cen,
                                    xnew_idl - xidl_cen])
                          / self.pixel_scale) + centre

        # Apply distortion
        return map_coordinates(image, coords_distort, order=1)

    def triangular_number(self, n):
        # TODO: Add to utils/math.py
        return n * (n + 1) / 2

    def distort_coords(self, A, B, X, Y, ):
        """
        Applts the distortion to the coordinates
        """

        # Promote shapes for float inputs
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Exponentiate
        Xpow = X[None, :, :] ** self.xpows[:, None, None]
        Ypow = Y[None, :, :] ** self.ypows[:, None, None]

        # Calcaulate new coordinates
        Xnew = np.sum(A[:, None, None] * Xpow * Ypow, axis=0)
        Ynew = np.sum(B[:, None, None] * Xpow * Ypow, axis=0)

        return Xnew, Ynew


"""
BFE Model
"""


class ApplyBFE(DetectorLayer):
    """
    Applies a non-linear convolution with a gaussian to model the BFE
    """
    m: float
    ksize: int

    def __init__(self, m, ksize=5):
        """
        Constructor
        """
        super().__init__()
        self.m = np.asarray(m, dtype=float)
        self.ksize = int(ksize)
        assert self.ksize % 2 == 1

    def __call__(self, image):
        """
        
        """
        im_out = self.apply_BFE(image.image, ksize=self.ksize)
        return im_out

    # Building the kernels
    def pixel_variance(self, pixel_flux):
        return self.m * pixel_flux

    def gauss(self, coords, sigma, eps=1e-8):
        gaussian = np.exp(-np.square(coords / (2 * np.sqrt(sigma) + eps)).sum(0))
        return gaussian / gaussian.sum()

    def build_kernels(self, image, npix=5):
        sigmas = self.pixel_variance(image)
        # coords = get_pixel_positions(npix)
        coords = dl.utils.pixel_coords(npix)
        fn = vmap(vmap(self.gauss, in_axes=(None, 0)), in_axes=(None, 1))
        return fn(coords, sigmas)

    @partial(vmap, in_axes=(None, None, None, 0, None, None))
    @partial(vmap, in_axes=(None, None, None, None, 0, None))
    def conv_kernels(self, padded_image, kernels, i, j, k):
        return padded_image[i, j] * kernels[j - k, i - k]

    # Get indexes
    def build_indexs(self, ksize, i, j):
        vals = np.arange(ksize)
        xs, ys = np.tile(vals, ksize), np.repeat(vals, ksize)
        out = np.array([xs, ys]).T
        inv_out = np.flipud(out)
        out_shift = out + np.array([i, j])
        indxs = np.concatenate([out_shift, inv_out], 1)
        return indxs

    @partial(vmap, in_axes=(None, None, None, None, 0))
    @partial(vmap, in_axes=(None, None, None, 0, None))
    def build_and_sum(self, array, ksize, i, j):
        indexes = self.build_indexs(ksize, j, i)
        return vmap(lambda x, i: x[tuple(i)],
                    in_axes=(None, 0))(array, indexes).sum()

    def apply_BFE(self, array, ksize=5):
        assert ksize % 2 == 1
        k = ksize // 2
        npix = array.shape[0]
        kernels = self.build_kernels(array, npix=ksize)
        Is, Js = np.arange(k, array.shape[0] + k), np.arange(k, array.shape[1] + k)
        convd = self.conv_kernels(np.pad(array, k), kernels, Is, Js, k)
        convd_pad = np.pad(convd, ((k, k), (k, k), (0, 0), (0, 0)))
        Is, Js = np.arange(npix), np.arange(npix)
        return self.build_and_sum(convd_pad, ksize, Is, Js)


"""
Jax version of the scipy gaussian filt
"""


def gaussian_kernel_1d(sigma):
    """
    Computes a 1-d Gaussian convolution kernel.
    """
    radius = int(4. * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma ** 2 * x ** 2)
    return phi_x / phi_x.sum()


# Convolution
def conv_axis(arr, kernel, axis):
    """
    Convolves the kernel along the axis
    """
    return vmap(convolve, in_axes=(axis, None, None))(arr, kernel, 'same')


def gaussian_filter(arr, sigma):
    """
    Applies a 1d gaussian filt along each axis of the input array
    
    Note this currently does not work correctly near the edges
    """
    kernel = gaussian_kernel_1d(sigma)[::-1]
    k = len(kernel) // 2
    arr = np.pad(arr, k, mode='symmetric')

    for i in range(arr.ndim):
        arr = conv_axis(arr, kernel, i)
    return arr[tuple([slice(k, -k, 1) for i in range(arr.ndim)])]


# Correlation
def corr_axis(arr, kernel, axis):
    """
    Correlates the kernel along the axis
    """
    return vmap(correlate, in_axes=(axis, None, None))(arr, kernel, 'same')


def gaussian_filter_correlate(arr, sigma):
    """
    Applies a 1d gaussian filt along each axis of the input array
    """
    kernel = gaussian_kernel_1d(sigma)[::-1]
    k = len(kernel) // 2
    arr = np.pad(arr, k, mode='symmetric')

    for i in range(arr.ndim):
        arr = corr_axis(arr, kernel, i)
    return arr[tuple([slice(k, -k, 1) for i in range(arr.ndim)])]


class ApplyJitter(DetectorLayer):
    """
    I actualy think this wont work properly becuase the array sizes change
    based on sigma - this may need to be fixed.
    """
    sigma: Array

    def __init__(self, sigma):
        super().__init__('ApplyJitter')
        self.sigma = np.asarray(sigma, dtype=float)

    def gaussian_kernel_1d(self, sigma, ksize=11):
        """
        Computes a 1-d Gaussian convolution kernel.
        """
        # radius = int(4. * sigma + 0.5)
        # x = np.arange(-radius, radius+1)
        x = np.linspace(-ksize / 2, ksize / 2, ksize)
        phi_x = np.exp(-0.5 / sigma ** 2 * x ** 2)
        return phi_x / phi_x.sum()

    # Correlation
    def corr_axis(self, arr, kernel, axis):
        """
        Correlates the kernel along the axis
        """
        return vmap(correlate, in_axes=(axis, None, None))(arr, kernel, 'same')

    def gaussian_filter_correlate(self, arr, sigma):
        """
        Applies a 1d gaussian filt along each axis of the input array
        """
        kernel = self.gaussian_kernel_1d(sigma)[::-1]
        k = len(kernel) // 2
        arr = np.pad(arr, k, mode='symmetric')

        for i in range(arr.ndim):
            arr = self.corr_axis(arr, kernel, i)
        arr_out = arr[tuple([slice(k, -k, 1) for i in range(arr.ndim)])]
        return arr_out.T

    def __call__(self, im):
        im_out = self.gaussian_filter_correlate(im, self.sigma)
        return im_out
