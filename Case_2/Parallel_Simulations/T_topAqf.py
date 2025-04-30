import flopy.utils.binaryfile as bf
import numpy as np
import xarray as xr
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.utils import axes
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.config import Directories


def T_TopAqf(ucn_file: str, cbc_file: str, layer_of_interest: int, gwT_natural):
    # load output files and output times
    ucn = bf.UcnFile(ucn_file)
    ucn_times = np.array(ucn.get_times())

    cbb = bf.CellBudgetFile(cbc_file)
    cbb_times = np.array(cbb.get_times())

    # get the output data
    cbb_data = cbb.get_data(text="flow lower face")  # Convert to NumPy array
    ucn_data = ucn.get_alldata(mflay=layer_of_interest)  # Convert to NumPy array

    # Initialize an empty list to store the mask indices
    # mask will contain for every time of ucn_data with which flowrate Q of cbb_data it should be paired
    mask = []

    # Iterate over the intervals cbb_times
    for i in range(len(cbb_times) - 1):
        if i == 0:
            start_time = 0
            end_time = cbb_times[i]

            # Find the indices in ucn_times that fall within this interval
            indices_within_interval = np.where(
                (ucn_times >= start_time) & (ucn_times < end_time)
            )
            if len(indices_within_interval[0]) > 0:
                mask.extend([x + i for x in indices_within_interval[0]])

        start_time = cbb_times[i]
        end_time = cbb_times[i + 1]
        indices_within_interval = np.where(
            (ucn_times > start_time) & (ucn_times <= end_time)
        )

        if len(indices_within_interval[0]) > 0:
            mask.extend([(x * 0) + i for x in indices_within_interval[0]])

    # Convert the mask list to a NumPy array
    mask = np.array(mask)
    # need for every ucn_time a cbb flowrate
    if len(mask) != len(ucn_times):
        print("Stop, something went wrong :/")
    else:
        print("Ok, continue")

    # Create Xarray for visualisation of the results (energy in every cell for every season)
    x, y = axes(exe_mf=Directories.exe_mf_dir, sim_ws=Directories.ws_dir)
    ucn_xr = xr.DataArray(
        ucn_data,
        coords={
            "time": ucn_times.tolist(),
            "y": y,
            "x": x,
        },
        dims=["time", "y", "x"],
        name="Temperature (°C)",
    )

    ucn_xr = ucn_xr.to_dataset()

    ucn_xr["Power"] = ucn_xr["Temperature (°C)"] * 0
    for i in range(len(mask)):
        ucn_xr["Power"][i] = (
            (abs(ucn_xr["Temperature (°C)"][i] - gwT_natural))
            * (cbb_data[mask[i]][layer_of_interest] * 3600)
            * 1.16
        )
        # Q in m3/h want c is in kWh/m3K

    # get for every time to which season it belongs (6 months of 30 days is a season)
    ucn_xr = ucn_xr.assign_coords(season=ucn_xr["time"] // (6 * 30 * 24 * 60 * 60))

    ucn_xr["time"] = ucn_xr["time"] / 3600
    # Create a new DataArray representing the time difference between consecutive time steps
    time_diff = ucn_xr["time"].diff(dim="time")

    # Calculate the integral of 'Power' for each season = energy in kWh
    integral_power_by_season = (
        (ucn_xr["Power"] * time_diff).groupby("season").sum(dim="time")
    )

    return integral_power_by_season, ucn_xr
