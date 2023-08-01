import webbpsf
import webbpsf.constants as const
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import numpy
from matplotlib import pyplot as plt
from jax import config
from poppy.zernike import hexike_basis


def jwst_hexike_bases(nterms=10, npix=1024, AMI=False):
    assert isinstance(AMI, bool), "AMI must be a boolean"

    # Get webbpsf model
    niriss = webbpsf.NIRISS()

    if AMI:
        seg_rad = 0.64 * const.JWST_SEGMENT_RADIUS
        # shifts from WebbPSF: CV3 on-orbit estimate (RPT028027) + OTIS delta from predicted (037134)
        shifts = const.JWST_CIRCUMSCRIBED_DIAMETER * np.array([0.0243, -0.0141])
        amplitude_plane = 3
        keys = ['B2-9', 'B3-11', 'B4-13', 'C5-16', 'B6-17', 'C1-8', 'C6-18']  # the segments that are used in AMI
        niriss.pupil_mask = 'MASK_NRM'

    else:
        seg_rad = const.JWST_SEGMENT_RADIUS  # 1.32 * 1.001  # TODO check for overlapping pixels
        shifts = np.zeros(2)  # no shift for full pupil
        amplitude_plane = 0
        keys = const.SEGNAMES_WSS  # all mirror segments

    niriss_osys = niriss.get_optical_system()
    seg_cens = dict(const.JWST_PRIMARY_SEGMENT_CENTERS)
    pscale = niriss_osys.planes[0].pixelscale.value * 1024 / npix

    # Scale mask
    transmission = niriss_osys.planes[amplitude_plane].amplitude
    mask = dl.utils.scale_array(transmission, npix, 1)

    # Generating a basis for each segment
    bases = []
    for key in keys:  # cycling through segments
        centre = np.array(seg_cens[key]) - shifts
        rhos, thetas = numpy.array(dlu.pixel_coordinates((npix, npix), pscale, offsets=tuple(centre), polar=True))
        bases.append(hexike_basis(nterms, npix, rhos/seg_rad, thetas, outside=0.))  # appending basis

    if AMI:
        return np.flip(np.array(bases), axis=(2, 3)), mask, pscale  # need to apply flip for AMI as plane 1 is a flip

    else:
        return np.array(bases), mask, pscale


if __name__ == '__main__':
    config.update("jax_enable_x64", True)
    plt.rcParams["image.origin"] = 'lower'


    def plot_bases(bases, mask, npix, pscale, save=False, edges=False):
        sample_bases = bases.sum(0) # * mask
        fig, ax = plt.subplots(2, 5, figsize=(12.5, 5))

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0., hspace=0.)
        for i in range(2):
            for j in range(5):
                bound = np.array([sample_bases[i * 5 + j].max(), -sample_bases[i * 5 + j].min()]).max()
                ax[i, j].imshow(sample_bases[i * 5 + j],
                                cmap='seismic',
                                vmin=-bound,
                                vmax=bound,
                                extent=(pscale * -npix / 2, pscale * npix / 2, pscale * -npix / 2, pscale * npix / 2)
                                )

                ax[i, j].set(xticks=[], yticks=[])

                if edges:
                    corners = const.JWST_PRIMARY_SEGMENTS
                    for segment in corners:
                        corner = segment[1].T
                        ax[i, j].plot(corner[0], corner[1], marker='', c='k', alpha=0.5, linestyle='--')

        if save:
            plt.savefig('hexike_bases.pdf', dpi=1000)
        plt.show()

    npix = 1024

    # bases, mask, pscale = jwst_hexike_bases(npix=npix, AMI=False)
    # plot_bases(bases, mask, npix, pscale)

    AMI_bases, AMI_mask, pscale = jwst_hexike_bases(AMI=True)
    plot_bases(AMI_bases, AMI_mask, npix, pscale, edges=True)
