import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa
from pyDGSA.dgsa import dgsa_interactions
from pyDGSA.plot import vert_pareto_plot
from pyDGSA.plot import plot_cdf
from pyDGSA.plot import interaction_matrix
from pyDGSA.cluster import KMedoids
from sklearn.metrics import silhouette_score

import numpy as np
from collections import Counter

from Scripts.Sine_input.config import Directories
from Scripts.Sine_input.utils import T_sine

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

plt.rcParams['figure.dpi'] = 400

''' load transport results '''
directory = os.path.join(Directories.output_dir,'HPC_Short_500_Correct_Tinitial')
prefix = 'MTO'
prefix2 = 'Parameters'
time = np.arange(0, (720)+0.5, 0.5)   # times we saved results we want to plot (days)

# Get a list of all directories in the output directory
directories = [os.path.join(directory, d) for d in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, d))]  # isdir to check if item is a directory

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files = [f for f in os.listdir(d) if f.startswith(prefix)]

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        exact_matches = response[response['Time (d)'].isin(time)]

        # from here on we can skip if we don't need every 0.5 day a value
        missing_times = np.setdiff1d(time, exact_matches['Time (d)'])  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(response['Time (d)'], missing_times, side='left')
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches

        response = round_custom(response)
        response = response.sort_values(by = ['Time (d)'])

        # can skip until here
        response.set_index('Time (d)', inplace=True)

        if directories.index(d) == 0:
            response = response[~response.index.duplicated()] #to check which rows are duplicates (based on indices) duplicates = response.index.duplicated() duplicate_rows = response[duplicates]
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
            # parameters = parameters.append(parameters_next, ignore_index=True)
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

parameters = parameters.drop(columns=['Unnamed: 0'])

#%% if we want to analyse seasons seperatly
split_index1 = 360
split_index2 = 540
split_index = 180
# Split the DataFrame into two based on the index
# season1_T = deltaT.loc[deltaT.index <= split_index]
# season2_T = deltaT.loc[deltaT.index > split_index]
season1_T = deltaT[(deltaT.index >= split_index1) & (deltaT.index < split_index2)]

#%% if we want to normalize the parameters (feature scaling)
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())
parameters = parameters.apply(normalize_column)

#%%
# calculate the euclidean distances between model responses
deltaT = deltaT.T  # one row for each model instead of one column for each model
distances = pdist(deltaT, metric='euclidean')
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMeans
n_clusters = 2
# clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels, medoids = clusterer.fit_predict(distances)

#check how many models in each cluster
print(Counter(labels))

#%%
''' DGSA calculations '''

# mean sensitivy averaged across all clusters
# output = 'mean' (default)
mean_sensitivity = dgsa(parameters, labels, parameter_names=parameter_names, n_boots=5000)
print(mean_sensitivity)

''' DGSA Plots'''

fig, ax = vert_pareto_plot(mean_sensitivity,
                           np_plot='+8')
# plt.title('Kmedoids - 2 clusters')
plt.tight_layout()
plt.show()
