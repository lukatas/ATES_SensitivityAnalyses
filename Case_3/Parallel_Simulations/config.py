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
    output_dir_lt: str = os.path.join(od_lt, "Campus", "Output_Delta5_60")
    fig_dir: str = os.path.join(cwd, "figures_svg")


class ModelParameters:
    """K (aquifer vs aquitard)"""

    Kh_aqf_min: float = 5e-06
    Kh_aqf_max: float = 5e-05

    Kh_aqt_min: float = 1e-08
    Kh_aqt_max: float = 1e-06

    Kv_from_Kh_min: float = 2  # divide Kh by this factor to het Kv
    Kv_from_Kh_max: float = 10

    """ porosity (aquifer vs aquitard) """
    por_Taqf_min: float = 0.20  # clayey sand
    por_Taqf_max: float = 0.50

    por_Taqt_min: float = 0.40  # sandy clay
    por_Taqt_max: float = 0.70

    effective_aqf_min: float = 0.5
    effective_aqf_max: float = 0.8

    effective_aqt_min: float = 0.05
    effective_aqt_max: float = 0.20

    """ gradient """  # in %
    gradient_min: float = 0  # in %
    gradient_max: float = 0.3

    """ thickness aquifer """  # komt overeen met +/- 2 dikte Yd4 (+/-21 procent verschil)
    aqf_dz_min: float = 14.7
    aqf_dz_max: float = 22.3

    """ dispersion """
    longitudinal_min: float = 1
    longitudinal_max: float = 5

    """ deltaT_inj """
    deltaT_inj_min: float = 5
    deltaT_inj_max: float = 7

    """ flowrate """
    flowrate_min: float = 3
    flowrate_max: float = 6
