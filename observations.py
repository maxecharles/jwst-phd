import dLux
from dLux.observations import BaseObservation
from jax.tree_util import tree_map
from equinox import tree_at

Instrument = lambda: dLux.instruments.BaseInstrument


class NIRISSFilters(BaseObservation):
    """
    NOTE: This class is under development.

    A class for modelling optical filters.
    """

    filters: list

    def __init__(self, filters: list = None):
        """
        Initialize a filt object.
        """
        super().__init__()
        if filters is None:
            self.filters = ['F480M', 'F430M', 'F380M']
        else:
            self.filters = filters

        for filt in self.filters:
            if filt not in ['F480M', 'F430M', 'F380M']:
                raise ValueError(f"Filter {filt} not supported.")

    @staticmethod
    def apply_filters(instrument: Instrument, filters: list) -> Instrument:
        """
        """
        filter_fn = lambda source: source.add('filt', filters)

        filtered_source = tree_map(
            f=filter_fn,
            tree=instrument.source,
            is_leaf=lambda leaf: isinstance(leaf, dLux.optics.AngularOptics)
        )

        # Apply updates
        return tree_at(
            where=lambda instrument: instrument.source,
            pytree=instrument,
            replace=filtered_source
        )

    def model(self, instrument: Instrument):
        pass
