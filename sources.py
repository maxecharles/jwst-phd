from __future__ import annotations
import jax.numpy as np
from jax import Array
import dLux

Source = lambda: dLux.sources.BaseSource


class NIRISSSource(dLux.sources.BaseSource):
    """

    """
    filt: str
    wavelengths: Array
    weights: Array
    log_flux: float
    offset: tuple

    def __init__(self, filt: str, log_flux: float = 0., offset: Array = np.zeros(2)):
        """

        """
        if filt not in ['F480M', 'F430M', 'F380M']:
            raise ValueError(f"Filter {filt} not supported.")

        super().__init__()

        filter_config = np.load(f"filter_configs/{filt}.npz")
        self.filt = filt
        self.wavelengths = filter_config['wavels']
        self.weights = filter_config['weights']
        self.log_flux = log_flux
        self.offset = offset

    def normalise(self):
        pass

    def model(self, optics, detector=None):
        return 10 ** self.log_flux * optics.propagate(self.wavelengths, offset=self.offset, weights=self.weights)
