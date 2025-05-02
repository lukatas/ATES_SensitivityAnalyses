import os
from ATES_SensitivityAnalyses.main import main_dir


class Directories:
    """Define main directories and file names"""

    cwd: str = main_dir  # folder from where I submit jobscript
    od: str = os.environ.get("VSC_SCRATCH_VO_USER")
    od_lt: str = os.environ.get("VSC_DATA_VO_USER")
    exe_mf_dir: str = os.path.join(cwd, "Software", "MF2005.1_12u", "make", "mf2005")
    exe_mt_dir: str = os.path.join(cwd, "Software", "mt3dusgs1.1.0u", "mt3dusgs")
    ws_dir: str = os.path.join(cwd, "Models")
    output_dir: str = os.path.join(od, "Output")
    output_dir_lt: str = os.path.join(
        od_lt, "Rijkevorsel", "Output"
    )  # long term storage


class ModelParameters:
    """K (upper and lower part of aquifer)"""

    Kh_aqf1_min: float = 1e-04
    Kh_aqf1_max: float = 6e-04

    Kh_aqf2_min: float = 5e-05
    Kh_aqf2_max: float = 2e-04

    Kv_from_Kh_min: float = 2  # divide Kh by this factor to get Kv
    Kv_from_Kh_max: float = 10

    """ porosity """
    por_Taqf_min: float = 0.25
    por_Taqf_max: float = 0.50

    effective_aqf_min: float = 0.5
    effective_aqf_max: float = 0.8

    """ gradient """  # in %
    gradient_min: float = 0
    gradient_max: float = 0.3

    """ dispersie """
    longitudinal_min: float = 0.5
    longitudinal_max: float = 5
