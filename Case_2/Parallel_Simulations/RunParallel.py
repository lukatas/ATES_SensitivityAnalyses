#! /usr/bin/env python3


import time
import multiprocessing as mp
import queue
from scipy.stats import qmc
from loguru import logger
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.Simulation import (
    forward_modelling,
)
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.config import ModelParameters


def worker_process(process_id, sample_queue, n_sim, kwargs):
    while True:
        try:
            # method get() reads and removes a python object from a queue instance
            # (once removed it is not available in the queue anymore)
            sample_point = sample_queue.get(
                timeout=1
            )  # Timeout to avoid blocking indefinitely
        except queue.Empty:
            break

        # Modify kwargs with the current sample_point
        kwargs["sample_point"] = sample_point
        kwargs["n_sim"] = n_sim  # Add this line to pass n_sim

        forward_modelling(**kwargs)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    start = time.time()

    n_cpu = 2  # Number of available processes
    n_sim = 2  # Number of total simulations you want
    kwargs = {
        "folder": 0,
        "pool": True,
        "FlowTransport": True,
        "ucn": False,
        "flush": True,
        "override": True,
    }

    # Scale the sample to the appropriate bounds & Generate the sample points in the hypercube space
    l_bounds = [
        ModelParameters.Kh_aqf1_min,
        ModelParameters.Kh_aqf2_min,
        ModelParameters.Kv_from_Kh_min,
        ModelParameters.gradient_min,
        ModelParameters.por_Taqf_min,
        ModelParameters.effective_aqf_min,
        ModelParameters.volume_min,
        ModelParameters.longitudinal_min,
        ModelParameters.Twinter_min,
        ModelParameters.Tzomer_min,
        ModelParameters.Rch_min,
    ]

    u_bounds = [
        ModelParameters.Kh_aqf1_max,
        ModelParameters.Kh_aqf2_max,
        ModelParameters.Kv_from_Kh_max,
        ModelParameters.gradient_max,
        ModelParameters.por_Taqf_max,
        ModelParameters.effective_aqf_max,
        ModelParameters.volume_max,
        ModelParameters.longitudinal_max,
        ModelParameters.Twinter_max,
        ModelParameters.Tzomer_max,
        ModelParameters.Rch_max,
    ]

    sampler = qmc.LatinHypercube(d=len(l_bounds))
    sample = sampler.random(n=n_sim)

    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    # Create a queue to hold the sample points
    sample_queue = mp.Queue()
    for point in sample_scaled:
        sample_queue.put(point)

    # Create and start parallel processes
    processes = []
    for i in range(n_cpu):
        process = mp.Process(
            target=worker_process, args=(i, sample_queue, n_sim, kwargs)
        )
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    end = time.time()
    logger.info(f"TET (hours) {(end - start) / 60 / 60}")

    # print points that were not processed by the worker processes (the ones that were not even started!)
    if not sample_queue.empty():
        logger.info("Remaining unprocessed points:")
        while not sample_queue.empty():
            remaining_point = sample_queue.get()
            logger.info(remaining_point)
