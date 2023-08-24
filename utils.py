import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from webbpsf import constants as const

plt.rcParams["image.origin"] = "lower"


def plot_and_compare(
    PSF1,
    PSF2,
    titles=None,
    pixel_crop=None,
    stretch="sqrt",
    colorbars=False,
    save_fig: bool = False,
    cbar_label: str = "Relative Intensity",
):
    if titles is None:
        titles = ["WebbPSF PSF", r"$\partial$Lux$ PSF"]

    if pixel_crop not in [None, 0]:
        PSF1 = PSF1[pixel_crop:-pixel_crop, pixel_crop:-pixel_crop]
        PSF2 = PSF2[pixel_crop:-pixel_crop, pixel_crop:-pixel_crop]

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 4))
    if colorbars:
        fig.subplots_adjust(left=0.0, right=0.954, top=0.8, bottom=0.2, wspace=0.01)
    else:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.2)
    ticks = [0, PSF1.shape[0] - 1]
    # WebbPSF PSF
    c0 = ax[0].imshow(PSF1, cmap="magma", norm=simple_norm(PSF1, stretch))
    ax[0].set(title=titles[0], xticks=ticks, yticks=ticks)

    # dLux PSF
    c1 = ax[1].imshow(PSF2, cmap="magma", norm=simple_norm(PSF2, stretch))
    ax[1].set(title=titles[1], xticks=ticks, yticks=ticks)

    if colorbars:
        fig.colorbar(c0, label=cbar_label)
        fig.colorbar(c1, label=cbar_label)

    # Residuals
    residuals = np.array(PSF1) - np.array(PSF2)
    bounds = np.array([-residuals.min(), residuals.max()])
    c2 = ax[2].imshow(residuals, cmap="seismic", vmin=-bounds.max(), vmax=bounds.max())
    ax[2].set(title=f"Residuals", xticks=ticks, yticks=ticks)
    fig.colorbar(c2, label="Residual")

    if save_fig:
        plt.savefig("psfs.pdf", dpi=400, bbox_inches="tight")
    plt.show()


def plot_basis(basis, pscale=None, mask=None, save=False, edges=False):
    npix = basis.shape[-1]
    if pscale is None:
        pscale = 0.0064486953125 * 1024 / npix
    if mask is None:
        mask = np.ones((npix, npix))
    sample_basis = basis.sum(0) * mask
    fig, ax = plt.subplots(2, 5, figsize=(12.5, 5))

    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.0, hspace=0.0
    )
    for i in range(2):
        for j in range(5):
            bound = np.array(
                [sample_basis[i * 5 + j].max(), -sample_basis[i * 5 + j].min()]
            ).max()
            ax[i, j].imshow(
                sample_basis[i * 5 + j],
                cmap="seismic",
                vmin=-bound,
                vmax=bound,
                extent=(
                    pscale * -npix / 2,
                    pscale * npix / 2,
                    pscale * -npix / 2,
                    pscale * npix / 2,
                ),
            )

            ax[i, j].set(xticks=[], yticks=[])

            if edges:
                corners = const.JWST_PRIMARY_SEGMENTS
                for segment in corners:
                    corner = segment[1].T
                    ax[i, j].plot(
                        corner[0],
                        corner[1],
                        marker="",
                        c="k",
                        alpha=0.5,
                        linestyle="--",
                    )

    if save:
        plt.savefig("hexike_basis.pdf", dpi=1000)
    plt.show()


def plot_opd(opd, pscale, mask=None):
    """
    Plots OPD

    Parameters
    ----------
    opd : Array
        OPD array in nanometers
    pscale : float
        Pixel scale in meters per pixel (?)
    mask : Array
        Mask array
    """
    if mask is None:
        mask = np.ones_like(opd)
    bound = np.array([(opd * mask).max(), -(opd * mask).min()]).max()
    npix = opd.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    c = ax.imshow(
        opd * mask,
        extent=(
            pscale * -npix / 2,
            pscale * npix / 2,
            pscale * -npix / 2,
            pscale * npix / 2,
        ),
        vmin=-bound,
        vmax=bound,
        cmap="coolwarm",
    )
    ax.set(title="JWST AMI Initial OPD", xlabel="x [m]", ylabel="y [m]")
    corners = const.JWST_PRIMARY_SEGMENTS
    for segment in corners:
        corner = segment[1].T
        ax.plot(
            corner[0],
            corner[1],
            marker="",
            c="k",
            alpha=0.5,
            linestyle="--",
        )
    fig.colorbar(c, label="OPD [nm]")
    plt.show()
