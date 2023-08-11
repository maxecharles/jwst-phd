import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
plt.rcParams["image.origin"] = "lower"


def plot_and_compare(
    PSF1,
    PSF2,
    titles=None,
    pixel_crop=None,
    stretch="sqrt",
    colorbars=False,
    save_fig: bool = False,
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
        fig.colorbar(c0, label="Relative Intensity")
        fig.colorbar(c1, label="Relative Intensity")

    # Residuals
    residuals = np.array(PSF1) - np.array(PSF2)
    bounds = np.array([-residuals.min(), residuals.max()])
    c2 = ax[2].imshow(residuals, cmap="seismic", vmin=-bounds.max(), vmax=bounds.max())
    ax[2].set(title=f"Residuals", xticks=ticks, yticks=ticks)
    fig.colorbar(c2, label="Residual")

    if save_fig:
        plt.savefig("psfs.pdf", dpi=400, bbox_inches="tight")
    plt.show()
