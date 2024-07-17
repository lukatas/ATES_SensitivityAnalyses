import os
from main import main_dir

class Directories:
    ''' Define main directories and file names '''
    cwd: str = main_dir             #folder from where I submit jobscript
    od: str = os.environ.get('VSC_SCRATCH_VO_USER')
    od_lt: str = os.environ.get('VSC_DATA_VO_USER')
    exe_mf_dir: str = os.path.join(cwd, 'Software', 'MF2005.1_12u', 'make', 'mf2005')
    exe_mt_dir: str = os.path.join(cwd, 'Software', 'mt3dusgs1.1.0u', 'mt3dusgs')
    ws_dir: str = os.path.join(cwd, 'Models')
    output_dir: str = os.path.join(od, 'Output')
    output_dir_lt: str = os.path.join(od_lt, 'Luik','Output') #long term storage


class ModelParameters:

    ''' K (upper and lower part aquifer) '''
    Kh_aqf1_min: float = 1e-05
    Kh_aqf1_max: float = 1e-03

    Kh_aqf2_min: float = 1e-03
    Kh_aqf2_max: float = 1e-01

    Kv_from_Kh_min: float = 2  # divide Kh by this factor to het Kv
    Kv_from_Kh_max: float = 10


    ''' porosity (aquifer vs aquitard) '''
    por_Taqf_min: float = 0.25
    por_Taqf_max: float = 0.50

    effective_aqf_min: float = 0.5
    effective_aqf_max: float = 0.8

    ''' gradient '''  # in %
    gradient_min: float = 0
    gradient_max: float = 0.2

    ''' dispersion '''
    longitudinal_min: float = 0.5
    longitudinal_max: float = 5

    ''' volume '''
    volume_min: float = 12500
    volume_max: float = 200000

    ''' Ttop '''
    Twinter_min: float = 2.5
    Twinter_max: float = 8

    Tzomer_min: float = 15
    Tzomer_max: float = 20.5

    ''' recharge '''
    Rch_min: float = 5.28e-09
    Rch_max: float = 8.46e-09
