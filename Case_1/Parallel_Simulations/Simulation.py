import uuid
import time
import os
import shutil
from os.path import join as jp
from loguru import logger
from ATES_SensitivityAnalyses.Case_1.Parallel_Simulations.config import Directories
from ATES_SensitivityAnalyses.Case_1.Parallel_Simulations.Rijkevorsel_FlowTransport import (
    FlowTransport,
)
from ATES_SensitivityAnalyses.Case_1.Parallel_Simulations.utils import (
    dirmaker,
    keep_essential,
)


def forward_modelling(**kwargs):
    """Data collection"""
    # Extract the required keyword arguments
    sample_point = kwargs.get("sample_point")

    # Main results directory name (random)
    res_dir = uuid.uuid4().hex

    # Create the result directory
    results_dir = jp(Directories.output_dir, res_dir)
    dirmaker(results_dir)

    logger.info(f"fwd {res_dir}")

    start_fwd = time.time()

    # Run Flow and Transport
    if kwargs["FlowTransport"]:
        FlowTransport(
            exe_mf=Directories.exe_mf_dir,
            exe_mt=Directories.exe_mt_dir,
            sim_ws=Directories.ws_dir,
            results_dir=results_dir,
            Kh_aqf1=sample_point[0],
            Kh_aqf2=sample_point[1],
            Kv_aqf1=sample_point[0] / sample_point[2],
            Kv_aqf2=sample_point[1] / sample_point[2],
            gradient=sample_point[3],
            por_Taqf=sample_point[4],
            por_Eaqf=sample_point[4] * sample_point[5],
            longitudinal=sample_point[6],
        )

    # Deletes everything except final results
    hl = (time.time() - start_fwd) // 60
    logger.info(f"done in {hl} min")

    if kwargs["flush"]:
        keep_essential(results_dir)

    # copy data from scratch to data directory for long-term storage
    source_path = results_dir
    destination_path = os.path.join(Directories.output_dir_lt, res_dir)

    shutil.move(source_path, destination_path)

    return results_dir
