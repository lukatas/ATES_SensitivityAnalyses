import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import scipy.integrate as integrate

from scipy import interpolate
from pyDGSA.cluster import KMedoids
from collections import Counter
from pyDGSA.dgsa import dgsa
from scipy.spatial.distance import pdist, squareform

from config import Directories

plt.rcParams['figure.dpi'] = 400

''' load model results from parallel simulatons '''

directory = os.path.join(Directories.output_dir,'LH_250_correct')
prefix = 'MTO'
prefix2 = 'Parameters'
time = np.arange(0, (645)+0.5, 0.5)   # times we saved results (days)

# Get a list of all directories in the output directory
directories = [os.path.join(directory, d) for d in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, d))]  # isdir to check if item is a directory

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files = [f for f in os.listdir(d) if f.startswith(prefix)]
  
    # DeltaT (Twarm-Tcold) results can be used for clustering --> get cluster colors (labels) for plots
  
    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches
        response['Time (d)'] = response['Time (d)'].round(1)
        response = response.sort_values(by = ['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates (based on indices) duplicates = response.index.duplicated() duplicate_rows = response[duplicates]
            deltaT = response[['DeltaT']].copy()
            deltaT.rename(columns={'DeltaT': '0'}, inplace=True)

        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT['{}'.format(directories.index(d))] = response['DeltaT']
          
    # need model response of warm and cold well also seperately to calculate efficiency later on
  
    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response['Time (d)'] = response['Time (d)'].round(1)
        response = response.sort_values(by=['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates (based on indices) duplicates = response.index.duplicated() duplicate_rows = response[duplicates]
            deltaT_warm = response[['DeltaT_warm']].copy()
            deltaT_warm.rename(columns={'DeltaT_warm': '0'}, inplace=True)
        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT_warm['{}'.format(directories.index(d))] = response['DeltaT_warm']

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response['Time (d)'] = response['Time (d)'].round(1)
        response = response.sort_values(by=['Time (d)'])

        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates (based on indices) duplicates = response.index.duplicated() duplicate_rows = response[duplicates]
            deltaT_cold = response[['DeltaT_cold']].copy()
            deltaT_cold.rename(columns={'DeltaT_cold': '0'}, inplace=True)
        else:
            response = response[~response.index.duplicated()]  # remove duplicate indices
            deltaT_cold['{}'.format(directories.index(d))] = response['DeltaT_cold']

    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))
        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            # parameters = parameters.append(parameters_next, ignore_index=True)
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

    ''' calculate power '''
    # Rijkevorsel simulations: 3 'short' years, sumer season is about 3.5 months and winter season as well, Q variation per month
    
    case = deltaT_warm[['{}'.format(directories.index(d))]].copy()
    case['deltaT_warm'] = deltaT_warm['{}'.format(directories.index(d))]
    case['deltaT_cold'] = deltaT_cold['{}'.format(directories.index(d))]

    # get for every time to which season it belongs
    months_ends = [31,62,92,103,123,153,184,215,
                   246,277,307,318,338,368,399,430,
                   461,492,522,533,553,583,614,645]

    season_ends = [103,215,
                   318,430,
                   533,645]

    #initiate these columns
    case['Seasons'] = 'Nan'
    case['month'] = 'Nan'

    for month in range(len(months_ends)-1):
        if month == 0:
            case.loc[(case.index<=months_ends[month]),'month'] = month
            case.loc[(case.index > months_ends[month]) & (case.index <= months_ends[month + 1]), 'month'] = month + 1
        else:
            case.loc[(case.index > months_ends[month]) & (case.index<= months_ends[month+1]),'month'] = month + 1

    for season in range(len(season_ends)-1):
        if season == 0:
            case.loc[(case.index<=season_ends[season]),'Seasons'] = season
            case.loc[(case.index >= season_ends[season]) & (case.index <= season_ends[season + 1]), 'Seasons'] = season + 1
        else:
            case.loc[(case.index >= season_ends[season]) & (case.index <= season_ends[season + 1]),'Seasons'] = season + 1

    #every month we use the injection temperature of the 'real' system so not fixed to 5 °C
    summer_months = [0, 1, 2, 3,
                     8, 9, 10, 11,
                     16, 17, 18, 19]

    case['DeltaT_inj'] = np.where(case.month.isin(summer_months), case.deltaT_warm, case.deltaT_cold)
    case['DeltaT_extr'] = np.where(case.month.isin(summer_months), case.deltaT_cold, case.deltaT_warm)

    #initiate these columns
    case['P_extr'] = 0
    case['P_inj'] = 0

    # P:kW
    # 1.16 is volumetric heat capacity of the groundwater 4.18MJ/(m3 K) (which is the same as 1.16 kWh/(m3K))
    Q = [0.002328816,0.001254391,0.00013642,0.000200011,0.000323212,0.000451053,0.000551673,0.000948495,
             0.002328816,0.001254391,0.00013642,0.000200011,0.000323212,0.000451053,0.000551673,0.000948495,
             0.002328816,0.001254391,0.00013642,0.000200011,0.000323212,0.000451053,0.000551673,0.000948495] #total rates per stress period (3 cycles)

    Q = [rate * 3600 for rate in Q] # m3/h

    case['P_extr'] = np.array(Q)[case['month'].astype(int).values] * 1.16 * case['DeltaT_extr']
    case['P_inj'] = np.array(Q)[case['month'].astype(int).values] * 1.16 * case['DeltaT_inj']

    seasons = case['Seasons'].to_numpy()

    days = case.index.to_numpy()  # get days
    hours = days * 24

    P_extr = case["P_extr"].to_numpy()  # get P
    P_inj = case["P_inj"].to_numpy()

    max_seasons = int(case['Seasons'].max()) + 1

    # integrate injection and extraction power over time to get the energy
    E_extr = np.zeros(max_seasons)
    for s in range(max_seasons):
        when = np.where(seasons == s)[0]  #get indices to get data from power for this all months of this season
        pim = P_extr[when]  # get power at all months from season s
        dam = hours[when]  # get corresponding hours 
      
        # integrate here
        funpow = interpolate.interp1d(dam, pim)
        powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

        E_extr[s] = powint  # (in kWh)


    E_inj = np.zeros(max_seasons)
    for s in range(max_seasons):
        when = np.where(seasons == s)[0]
        pim = P_inj[when]
        dam = hours[when]

        # integrate here
        funpow = interpolate.interp1d(dam, pim)
        powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

        E_inj[s] = powint   # (in kWh)

    if directories.index(d) == 0:
        #initiate empte df with 250 columns and 2 rows
        column_names = [str(i) for i in range(len(directories))]
        data = [[0.0] * 250, [0.0] * 250, [0.0] * 250, [0.0] * 250, [0.0] * 250, [0.0] * 250]  # Initializing with zeros and ones for example
        E_inj_df = pd.DataFrame(data, columns=column_names, index = [0,1,2,3,4,5])
        E_extr_df = pd.DataFrame(data, columns=column_names, index= [0,1,2,3,4,5])

    E_inj_df[str(directories.index(d))] = E_inj # to check you can calculate this manually as well based on input ((T-Tinitial)*time (h) * flowrate (m3/h) * 1.16)
    E_extr_df[str(directories.index(d))] = E_extr

parameters = parameters.drop(columns=['Unnamed: 0'])

''' calculate efficiency '''
for m in range(len(directories)):
    # calculate thermal recovery efficiency
    for s in range(max_seasons):
        if s > 0:
            Eff = (E_extr_df[str(m)][s] / E_inj_df[str(m)][s - 1]) * 100

            if m == 0 and s == 1:
                column_names = [str(i) for i in range(len(directories))]
                data = [[0.0] * 250, [0.0] * 250, [0.0] * 250,
                        [0.0] * 250, [0.0] * 250, [0.0] * 250]  # Initializing with zeros and ones for example
                Eff_df = pd.DataFrame(data, columns=column_names, index=[0,1,2,3,4,5])

            Eff_df[str(m)][s] = Eff
