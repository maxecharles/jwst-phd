import dLux

# Alias classes for simplified type-checking
Optics = lambda: dLux.optics.BaseOptics
Detector = lambda: dLux.detectors.BaseDetector
Source = lambda: dLux.sources.BaseSource
Observation = lambda: dLux.observations.BaseObservation
Image = lambda: dLux.images.Image


class NIRISS(dLux.instruments.Instrument):
    """

    """
    optics: None
    source: None
    detector: None
    observation: None

    def __init__(self,
                 optics: Optics = None,
                 source: Source = None,
                 detector: Detector = None,
                 observation: Observation = None):
        """

        """
        self.optics = optics
        self.source = source
        self.detector = detector
        self.observation = observation
        super().__init__(optics=optics, sources=source, detector=detector, observation=observation)

    def normalise(self):
        """
        Normalises the source flux to 1.
        """
        return self.set('source', self.source.normalise())

    def model(self):
        """
        Method to model the Instrument source through the optics, giving the PSF of the instrument.
        """
        return self.optics.model(self.source)
