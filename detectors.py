import dLux
import dLuxWebbpsf
import webbpsf

class NIRISSAMIDetector(dLux.LayeredDetector):

    def __init__(
        self,
        downsample: int = 4,
        wss_date: str = None,
    ):
        niriss = webbpsf.NIRISS()  # WebbPSF instrument

        niriss.pupil_mask = "MASK_NRM" # applies the NRM mask -- remove this line for full pupil simulation
        if wss_date is not None:
            niriss.load_wss_opd_by_date(date=wss_date, verbose=False)  # loads the WSS OPD map for the given date
        niriss.calc_psf()
        niriss_osys = niriss.get_optical_system()

        super().__init__([
            dLuxWebbpsf.detector_layers.Rotate(-dLux.utils.deg_to_rad(getattr(niriss.siaf.apertures["NIS_CEN"], "V3IdlYAngle"))),
            dLuxWebbpsf.DistortionFromSiaf(instrument=niriss, optics=niriss_osys),
            dLux.IntegerDownsample(kernel_size=downsample),  # Downsample to detector pixel scale
            ])