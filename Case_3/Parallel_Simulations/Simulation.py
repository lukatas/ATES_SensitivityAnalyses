import uuid
from os.path import join as jp
from loguru import logger
import time
import os
import shutil

from config import Directories
from Campus_FlowTransport import FlowTransport
from utils import dirmaker, keep_essential


def forward_modelling(**kwargs):
    """Data collection"""
    # Extract the required keyword arguments
    n_sim = kwargs.get("n_sim")
    sample_point = kwargs.get("sample_point")

    # Main results directory.
    res_dir = uuid.uuid4().hex

    # Generates the result director
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
            Kh_aqf=sample_point[0],
            Kh_aqt=sample_point[1],
            Kv_aqf=sample_point[0] / sample_point[2],
            Kv_aqt=sample_point[1] / sample_point[2],
            gradient=sample_point[3],
            por_Taqf=sample_point[4],
            por_Taqt=sample_point[5],
            por_Eaqf=sample_point[4] * sample_point[6],
            por_Eaqt=sample_point[5] * sample_point[7],
            longitudinal = sample_point[8],
            aqf_dz = sample_point[9],
            deltaT_inj = sample_point[10],
            flowrate = sample_point[11]
            )

    # Deletes everything except final results
    hl = (time.time() - start_fwd) // 60
    logger.info(f"done in {hl} min")

    if kwargs["flush"]:
        keep_essential(results_dir)
        # else:
    #     shutil.rmtree(results_dir)
    #     logger.info(f"terminated {res_dir}")
    #     return 0

    # copy data from scratch to data directory for long-term storage
    source_path = results_dir
    destination_path = os.path.join(Directories.output_dir_lt, res_dir)

    shutil.move(source_path, destination_path)

    return results_dir