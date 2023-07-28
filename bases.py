import webbpsf
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import numpy
from matplotlib import pyplot as plt
from jax import config
from poppy.zernike import hexike_basis


def jwst_hexike_bases(nterms=10, npix=512, AMI=False):
    assert isinstance(AMI, bool), "AMI must be a boolean"

    # Get webbpsf model
    niriss = webbpsf.NIRISS()
    if AMI:
        niriss.pupil_mask = 'MASK_NRM'
    niriss_osys = niriss.get_optical_system()
    seg_cens = niriss_osys.planes[0]._seg_centers_m  # grabbing centres from primary mirror plane

    if AMI:
        diam = 2.05
        shifts = np.array([0.15, -0.09])
        amplitude_plane = 3
        keys = ['B2', 'B3', 'B4', 'C5', 'B6', 'C1', 'C6']  # the segments that are used in AMI

    else:
        diam = 1.32 * 1.001
        shifts = np.zeros(2)
        amplitude_plane = 0
        keys = seg_cens.keys()

    pscale = niriss_osys.planes[0].pixelscale.value * 1024 / npix

    # Scale mask
    pupil = niriss_osys.planes[amplitude_plane].amplitude
    mask = dl.utils.scale_array(pupil, npix, 1)

    # Gen basis
    bases = []

    for key in keys:
        centre = seg_cens[key] + shifts
        rhos, thetas = numpy.array(dlu.pixel_coordinates((npix, npix), pscale, offsets=tuple(centre), polar=True))
        bases.append(hexike_basis(nterms, npix, diam * rhos, thetas, outside=0.))

    if AMI:
        return np.flip(np.array(bases), axis=2), mask  # need to apply flip to AMI as plane 1 is a flip
    else:
        return np.array(bases), mask


if __name__ == '__main__':
    config.update("jax_enable_x64", True)
    plt.rcParams["image.origin"] = 'lower'


    def plot_bases(bases, mask, save=False):
        sample_bases = bases.sum(0)  # * mask
        fig, ax = plt.subplots(2, 5, figsize=(12.5, 5))

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0., hspace=0.)
        for i in range(2):
            for j in range(5):
                bound = np.array([sample_bases[i * 5 + j].max(), -sample_bases[i * 5 + j].min()]).max()
                ax[i, j].imshow(sample_bases[i * 5 + j],
                                cmap='seismic',
                                vmin=-bound,
                                vmax=bound,
                                )

                ax[i, j].set(xticks=[], yticks=[])
        if save:
            plt.savefig('hexike_bases.pdf', dpi=1000)
        plt.show()


    bases, mask = jwst_hexike_bases(AMI=False)
    plot_bases(bases, mask)

    AMI_bases, AMI_mask = jwst_hexike_bases(AMI=True)
    plot_bases(AMI_bases, AMI_mask)
