import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import xarray as xr
from scipy import interpolate
import scipy.integrate as integrate
from pyDGSA.cluster import KMedoids
from collections import Counter
from scipy.spatial.distance import pdist, squareform

from Scripts.Sine_input.config import Directories
from Scripts.Sine_input.utils import Q_sine
plt.rcParams['figure.dpi'] = 400

#made specific rounding function to round to .0 or .5, examples:
# 116.504 to 116.5: 116.05 to 116.0; 153.11 to 153.0; 153.68 to 153.5
#(rounding method for Rijkevorsel case study did not work well enough here)
def round_custom(df):
    df['Time (d)'] = df['Time (d)'] * 10
    df['Time (d)'] = df['Time (d)'].apply(math.floor)

    for r in range(len(df['Time (d)'])):
        if df['Time (d)'][r]%5==0:
            # df['Time (d)'][r]=df['Time (d)'][r]/10 (gives warning)
            df.loc[r, 'Time (d)'] = df.loc[r, 'Time (d)'] / 10
        else:
            # df['Time (d)'][r] = ((df['Time (d)'][r]-(df['Time (d)'][r]%5))/10)
            df.loc[r, 'Time (d)'] = (df.loc[r, 'Time (d)'] - (df.loc[r, 'Time (d)'] % 5)) / 10
    return df

''' load transport results '''
directory = os.path.join(Directories.output_dir,'HPC_Short_500')

prefix = 'MTO'
prefix2 = 'Parameters'
prefix3 = 'EXCHANGE'
time = np.arange(0, (720)+0.5, 0.5)   # times we saved results (days)
number_of_seasons = 4
year = 360  # in days
simulation_period = 2 # in years
time_discretization = 30  # in days

# Get a list of all directories in the output directory
directories = [os.path.join(directory, d) for d in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, d))]  # isdir to check if item is a directory

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files = [f for f in os.listdir(d) if f.startswith(prefix)]

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response = round_custom(response)
        response = response.sort_values(by = ['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates (based on indices)
            deltaT_cold = response[['DeltaT_cold']].copy()
            deltaT_cold.rename(columns={'DeltaT_cold': '0'}, inplace=True)
        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT_cold['{}'.format(directories.index(d))] = response['DeltaT_cold']

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response = round_custom(response)
        response = response.sort_values(by = ['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates 
            deltaT_warm = response[['DeltaT_warm']].copy()
            deltaT_warm.rename(columns={'DeltaT_warm': '0'}, inplace=True)
        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT_warm['{}'.format(directories.index(d))] = response['DeltaT_warm']
          
    # DeltaT can be used to calculate clusters --> use for labels for colors plot
    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response = round_custom(response)
        response = response.sort_values(by = ['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates 
            deltaT = response[['DeltaT']].copy()
            deltaT.rename(columns={'DeltaT': '0'}, inplace=True)
        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT['{}'.format(directories.index(d))] = response['DeltaT']

    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))
        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

   ''' calculate power '''
    # the flowrate changed every month
    case = deltaT_warm[['{}'.format(directories.index(d))]].copy()
    case['deltaT_warm'] = deltaT_warm['{}'.format(directories.index(d))]
    case['deltaT_cold'] = deltaT_cold['{}'.format(directories.index(d))]

    # get for every time to which season it belongs
    case['Seasons'] = case.index // 30 // 6

    case['DeltaT_inj'] = np.where(case.Seasons % 2 == 0, 15 - 10, abs(5 - 10))
    case['DeltaT_extr'] = np.where(case.Seasons % 2 == 0, case.deltaT_cold, case.deltaT_warm)

    volume = parameters['volume'][directories.index(d)]   # in m3

    # P:kW
    # 1.16 is volumetric heat capacity of the groundwater 4.18MJ/(m3 K) (which is the same as 1.16 kWh/(m3K))

    Q = Q_sine(vol=volume,
               time_dis_sine=time_discretization,
               sim_time_years=simulation_period,
               year_days=year)
    Q = abs(Q) * 3600 # m3/h
    case['month'] = case.index // 30

    #initiate these columns
    case['P_extr'] = 0
    case['P_inj'] = 0

    for q in range(len(Q)):
        for t in case.index:
            if case['month'][t] == q:
                case.loc[t, 'P_extr'] = Q[q] * 1.16 * case['DeltaT_extr'][t]  # Q in m3/h
                case.loc[t, 'P_inj'] = Q[q] * 1.16 * case['DeltaT_inj'][t]

    seasons = case['Seasons'].to_numpy()

    days = case.index.to_numpy()  # get days
    hours = days * 24

    P_extr = case["P_extr"].to_numpy()  # get P
    P_inj = case["P_inj"].to_numpy()

    max_seasons = int(case['Seasons'].max())

    # integrate injection and extraction power over time (every time the 6 months interval) to get the energy
    E_extr = np.zeros(max_seasons)
    for s in range(max_seasons):
        when = np.where(seasons == s)[0]  # when is the month m #get indices to get data from pi for this month
        pim = P_extr[when]  # get power at month m
        dam = hours[when]  # get corresponding hours (need to integrate over this time in hours)

        # integrate here
        funpow = interpolate.interp1d(dam, pim)
        powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

        E_extr[s] = powint  # (in kWh)

    E_inj = np.zeros(max_seasons)
    for s in range(max_seasons):
        when = np.where(seasons == s)[0]  # when is the month m #get indices to get data from pi for this month
        pim = P_inj[when]  # get power at month m
        dam = hours[when]  # get corresponding hours (need to integrate over this time in hours)

        # integrate here
        funpow = interpolate.interp1d(dam, pim)
        powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

        E_inj[s] = powint   # (in kWh)
      
    ''' load data to calculate energy exchange aquifer with atmosphere '''

    files3 = [p for p in os.listdir(d) if p.startswith(prefix3)]
    for file3 in files3:
        energy = xr.open_dataarray(os.path.join(d, file3))
        total_C = []
        total_W = []
        for s in range(number_of_seasons):
            zoomed_data_Cold = energy.sel(season=int('{}'.format(s)), x=slice(643, 661), y=slice(509, 491))
            zoomed_data_Warm = energy.sel(season=int('{}'.format(s)), x=slice(351, 369), y=slice(509, 491))

            # grid cells around wells have same size
            total_C.append(np.sum(zoomed_data_Cold.values))
            total_W.append(np.sum(zoomed_data_Warm.values))

        if directories.index(d) == 0:
            df_energy_W = pd.DataFrame(total_W, columns=[str(directories.index(d))], index=range(number_of_seasons))
            df_energy_C = pd.DataFrame(total_C, columns=[str(directories.index(d))], index=range(number_of_seasons))
        else:
            next_col_W = pd.DataFrame(total_W, columns=[str(directories.index(d))], index=range(number_of_seasons))
            next_col_C = pd.DataFrame(total_C, columns=[str(directories.index(d))], index=range(number_of_seasons))
            df_energy_W = pd.concat([df_energy_W, next_col_W], axis=1)
            df_energy_C = pd.concat([df_energy_C, next_col_C], axis=1)

   # make dataframe and calculations of soil energy exchange and safe Injected and extracted energy per season

    if directories.index(d) == 0:
        #initiate empte df with 250 columns and 2 rows
        column_names = [str(i) for i in range(len(directories))]
        data = [[0.0] * 500, [0.0] * 500, [0.0] * 500, [0.0] * 500]  # Initializing with zeros and ones for example
        share_exchange_WW = pd.DataFrame(data, columns=column_names, index=[0,1,2,3])
        share_exchange_CW = pd.DataFrame(data, columns=column_names, index=[0,1,2,3])
        E_inj_df = pd.DataFrame(data, columns=column_names, index = [0,1,2,3])
        E_extr_df = pd.DataFrame(data, columns=column_names, index= [0,1,2,3])

    E_inj_df[str(directories.index(d))] = E_inj
    E_extr_df[str(directories.index(d))] = E_extr

    for s in range(number_of_seasons):
        if s % 2 == 0: #zomer
            share_exchange_WW[str(directories.index(d))][s] = (df_energy_W[str(directories.index(d))][s] / E_inj[s]) * 100
            share_exchange_CW[str(directories.index(d))][s] = (df_energy_C[str(directories.index(d))][s] / E_extr[s]) * 100
        else:
            share_exchange_WW[str(directories.index(d))][s] = (df_energy_W[str(directories.index(d))][s] / E_extr[s]) * 100
            share_exchange_CW[str(directories.index(d))][s] = (df_energy_C[str(directories.index(d))][s] / E_inj[s]) * 100

parameters = parameters.drop(columns=['Unnamed: 0'])

''' Calculate efficiency '''
for m in range(len(directories)):
    # calculate thermal recovery efficiency
    for s in range(max_seasons):
        if s > 0:
            Eff = (E_extr_df[str(m)][s] / E_inj_df[str(m)][s - 1]) * 100

            if m == 0 and s == 1:
                column_names = [str(i) for i in range(len(directories))]
                data = [[0.0] * 500, [0.0] * 500, [0.0] * 500,
                        [0.0] * 500]  # Initializing with zeros and ones for example
                Eff_df = pd.DataFrame(data, columns=column_names, index=[0,1, 2, 3])

            Eff_df[str(m)][s] = Eff
