import dLux
from dLux.observations import BaseObservation
import jax.numpy as np
from jax import Array
from jax import tree_map

Instrument = lambda: dLux.instruments.BaseInstrument


class NIRISSFilters(BaseObservation):
    """
    NOTE: This class is under development.

    A class for modelling optical filters.
    """

    filters: list
    spectra: list

    def __init__(self, filters: list = None):
        """
        """
        super().__init__()

        if filters is None:
            self.filters = ['F480M', 'F430M', 'F380M']
        else:
            self.filters = filters

        for filt in self.filters:
            if filt not in ['F480M', 'F430M', 'F380M']:
                raise ValueError(f"Filter {filt} not supported.")

        self.spectra = [dLux.Spectrum(**dict(np.load(f'filter_configs/{filt}.npz'))) for filt in self.filters]

    def model(self, instrument: Instrument, *args, **kwargs) -> Array:
        """
        """

        psfs = [instrument.set('sources.PointSource.spectrum', spectrum).model() for spectrum in self.spectra]

        # set_and_model = lambda spectrum: instrument.set('sources.spectrum', spectrum).model(*args, **kwargs)
        # leaf_fn = lambda leaf: isinstance(leaf, dLux.spectra.Spectrum)
        # psfs = tree_map(f=set_and_model, tree=instrument.sources, rest=self.spectra, is_leaf=leaf_fn)
        return np.array(psfs)
