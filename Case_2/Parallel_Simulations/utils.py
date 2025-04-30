# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:08:19 2023

@author: lhtas
"""

import os
import shutil
import flopy
import numpy as np
from loguru import logger
from scipy import interpolate
import scipy.integrate as integrate


def dirmaker(dird: str, erase: bool = False):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :param erase: bool: Whether to delete existing folder or not.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            if erase:
                shutil.rmtree(dird)
                os.makedirs(dird)
            return 1
    except Exception as e:
        logger.warning(e)
        return 0


def keep_essential(results_dir: str):
    """
    Deletes everything in a simulation folder except specific files.
    :param res_dir: Path to the folder containing results.
    """
    for the_file in os.listdir(results_dir):
        if (
            not the_file.endswith(".ucn")  # files we want to keep
            and not the_file.endswith(".csv")
            and not the_file.endswith(".mto")
            and not the_file.endswith(".nc")
        ):
            file_path = os.path.join(results_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(
                        file_path
                    )  # difference: does not care if directory is empty or not
            except Exception as e:
                logger.warning(e)


def axes(exe_mf: str, sim_ws: str):
    f = os.path.join(sim_ws, "Luik_2Y.nam")
    mf = flopy.modflow.Modflow.load(
        f, version="mf2005", exe_name=exe_mf, model_ws=sim_ws
    )
    XGR = mf.dis.delr.array
    YGR = mf.dis.delc.array

    y = (
        1000 - YGR.cumsum()
    )  # Correct coordinates used in modelmuse to specify grid (y-axis from top to bottom)
    x = XGR.cumsum()

    return x, y


def NewGridValues(nrow: int, ncol: int, new_value: float):
    new = np.ones((nrow, ncol)) * new_value

    return new


def Q_sine(vol: float, time_dis_sine: int, sim_time_years: int, year_days: int):
    """parameters"""
    year = year_days  # in days
    simulation_period = sim_time_years * year
    time_discretization = time_dis_sine  # in days
    storage_volume = vol  # volume injected or extracted in one season (180 days) m3

    """ get amplitude sine function based on storage volume in one season (after PySeawATES) """
    # how many periods with different flowrate per year
    PerPerYear = int(round(year / time_discretization))

    # calculate the amplitude based on a season (6 months period)
    PPY = int(PerPerYear / 2)
    SumSine = 0
    for i in range(PPY):
        # np.pi * i calculates angle in radians for the current period
        # dividing by PPY scales the angle to ensure that the sine wave completes one full cycle within half of a year (from 0 to 2pi)
        Sine = np.sin(np.pi * (i) / PPY)
        # SumSine will be used to scale the amplitude of the sine wave
        SumSine += Sine

    # calculate amplitude which will essentially be scaling factor that helps adjust the sine wave to match the desired seasonal flow variations
    amplitude = round(1 / SumSine * storage_volume / time_discretization, 0)  # in m3/d

    """calculate flow rate for every hour (m3/s) based on sine wave function"""
    time = np.arange(0, simulation_period, 1 / 24)  # in days, one value every hour
    # Set the frequency and amplitude of the sine wave
    frequency = 2 * np.pi / year
    amplitude = amplitude / (24 * 3600)  # y-axis in m3/s instead of m3/days
    # Generate the pumping rate as a sine wave (y=asin(bx); a is amplitude, b is periodicity
    pumping_rate = amplitude * np.sin(frequency * time)

    # integrate function in steps of time_discretization
    """ get flow rate input for model (36 periods of 10 days per year of 360 days)"""
    hours = np.arange(0, simulation_period * 24, 1)
    seconds = time * 24 * 3600
    fl = pumping_rate

    ds = hours // 24 // time_discretization
    max_days = int(ds.max())
    flowrate = np.zeros(max_days + 1)
    for m in range(max_days + 1):
        when = np.where(ds == m)[0]  # get indices to get data from pi for this day
        flm = fl[when]  # get flowrate at day m
        sem = seconds[
            when
        ]  # get corresponding seconds (need to integrate over this time in seconds)
        # integrate here
        funpow = interpolate.interp1d(sem, flm)
        flowint, abse = integrate.quad(funpow, sem[0], sem[-1])
        flowrate[m] = flowint / (
            time_discretization * 24 * 3600
        )  # flow rate per period

    return flowrate


def T_sine(
    Twinter: float,
    Tzomer: float,
    time_dis_sine: int,
    sim_time_years: int,
    year_days: int,
):
    # Parameters
    year = year_days  # in days
    simulation_period = sim_time_years * year
    time_discretization = time_dis_sine  # days
    amplitude = (Tzomer - Twinter) / 2  # (20-3)/2
    vertical_offset = (Twinter + Tzomer) / 2  # (20+3)/2

    time = np.arange(0, simulation_period, 1 / 24)  # in days ,one value every hour

    # Calculate temperature based on continuous sine wave
    frequency = 2 * np.pi / year  # Assuming the temperature varies yearly
    temperature = amplitude * np.sin(frequency * time) + vertical_offset

    """ get T input for model """
    hours = np.arange(0, simulation_period * 24, 1)
    seconds = time * 24 * 3600

    ds = hours // 24 // time_discretization
    max_days = int(ds.max())
    T = np.zeros(max_days + 1)
    for m in range(max_days + 1):
        when = np.where(ds == m)[0]  # get indices to get data from pi for this day
        flm = temperature[when]  # get flowrate at day m
        sem = seconds[
            when
        ]  # get corresponding seconds (need to integrate over this time in seconds)
        # integrate here
        funpow = interpolate.interp1d(sem, flm)
        T_int, abse = integrate.quad(funpow, sem[0], sem[-1])
        T[m] = T_int / (time_discretization * 24 * 3600)  # flow rate per period

    return T
