# -*- coding: utf-8 -*-
"""
Created on Mon July 08 09:12:04 2024

@author: lhtas
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa
from pyDGSA.dgsa import dgsa_interactions
from pyDGSA.plot import vert_pareto_plot
from pyDGSA.plot import plot_cdf
from pyDGSA.plot import interaction_matrix
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable,axes_size
from ATES_SensitivityAnalyses.Case_3.Parallel_Simulations.config import Directories

plt.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text in svg files
plt.rcParams['figure.dpi'] = 400
plt.rcParams.update({'font.size': 14})
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.labelsize'] = 13


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

directory = os.path.join(Directories.output_dir, 'LH_500_Delta5_40')
prefix = 'MTO'
prefix2 = 'Parameters'
time = np.arange(0, (1080)+1, 1)
skipped_indices = [] #get list of non-convergent models and remove them from analysis

# Get a list of all directories in the output directory
directories = [os.path.join(directory, d) for d in os.listdir(directory) if
               os.path.isdir(os.path.join(directory, d))]

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files = [f for f in os.listdir(d) if f.startswith(prefix)]

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        # a few models did not converge, filter them out

        if response['Time (d)'].iloc[-1] < 1080:
            skipped_indices.append(directories.index(d))
        # # use this to filter out non-convergent/bad models for distance = 60m,
        # # otherwise deactivate this part
        # elif (response['DeltaT_cold']>10000).any() or \
        #         (response['DeltaT_cold']<-100).any() or\
        #         (response['DeltaT_warm']>5.05).any() or (response['DeltaT_warm']<-4).any()\
        #         or (response['DeltaT_warm']<1).any()\
        #         :
        #     skipped_indices.append(directories.index(d))
        # # use this to filter out non-convergent/bad models for distance = 40m or 80m,
        # # otherwise deactivate this part
        elif (response['DeltaT_cold'] > 10000).any() or \
                (response['DeltaT_cold'] < -100).any() or \
                (response['DeltaT_warm'] > 5.1).any() \
                :
            skipped_indices.append(directories.index(d))
        else:
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
                response = response[~response.index.duplicated()]
                deltaT = response[['DeltaT']].copy()
                deltaT.rename(columns={'DeltaT': '0'}, inplace=True)

            else:
                response = response[~response.index.duplicated()]  # remove duplicate indices
                deltaT['{}'.format(directories.index(d))] = response['DeltaT']

    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))

        elif directories.index(d) in skipped_indices:
            continue
        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

parameters = parameters.drop(columns=['Unnamed: 0'])

#%% if we want to normalize the parameters (feature scaling)
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())
parameters = parameters.apply(normalize_column)

''' Clustering '''
#%%
# calculate the euclidean distances between model responses
deltaT = deltaT.T  # one row for each model
#reset index because we miss some values (skipped_indices)
deltaT = deltaT.reset_index()
deltaT = deltaT.drop(columns=['index'])
deltaT.index = deltaT.index.map(str)
distances = pdist(deltaT, metric='euclidean')
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMedoids
n_clusters = 2
clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
# clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels= clusterer.fit_predict(distances)

#check how many models in each cluster
print(Counter(labels))

''' DGSA calculations '''
#%%
parameters = parameters.to_numpy()
parameter_names = ['Kh aqf', 'Kh aqt', 'Kv aqf', 'Kv aqt', 'Grad.', 'Tot. por. aqf.',
                   'Tot. por. aqt.','Eff. por. aqf.', 'Eff. por. aqt.' ,'Long. disp.',
                   'Thick.',
                   #'DeltaT_inj',
                   'Flowrate']

# mean sensitivy averaged across all clusters
mean_sensitivity = dgsa(parameters, labels, parameter_names=parameter_names, n_boots=5000, confidence=True)
print(mean_sensitivity)
#%%
# standardized sensitivity for each individual cluster
cluster_names = ['Low ΔT','High ΔT']
cluster_colors = ['xkcd:coral', 'xkcd:blurple']
cluster_sensitivity = dgsa(parameters, labels, parameter_names=parameter_names,
                           output='cluster_avg', cluster_names=cluster_names)
print(cluster_sensitivity)


# mean sensitivity including two-way parameter interactions (sensitivity values are averaged over each cluster and bin)

mean_interact_sensitivity = dgsa_interactions(parameters, labels,
                                              parameter_names=parameter_names)
print(mean_interact_sensitivity)



''' DGSA Plots'''
#%%

fig, ax = vert_pareto_plot(mean_sensitivity,
                           np_plot='+13', confidence=True)
plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir,'temp sens 60.svg'),format='svg')
plt.show()

#%%

fig, ax = vert_pareto_plot(cluster_sensitivity, np_plot=12, fmt='cluster_avg',
                           colors=cluster_colors)
plt.tight_layout()
plt.show()
#%% class-conditional cumulative distribution function

fig, ax = plot_cdf(parameters, labels, 'Thick.', parameter_names=parameter_names,
                   cluster_names=cluster_names, colors=cluster_colors)
plt.show()
# could also plot pdf (probability distribution function)
#%%

fig, ax = vert_pareto_plot(mean_interact_sensitivity, np_plot='+8')
plt.tight_layout()
plt.show()

#%%
fig, ax = interaction_matrix(mean_interact_sensitivity, fontsize=8)
plt.savefig(os.path.join(Directories.fig_dir,'temp interaction matrix 60.svg'),format='svg')
plt.tight_layout()
plt.show()

''' plot ΔT vs time with cluster colors'''

#%%
deltaT = deltaT.T
fig,ax = plt.subplots()
models = len(directories) - len(skipped_indices)
for m in range(models):
    label = labels[m]
    color = cluster_colors[label]
    ax.plot(deltaT.index, deltaT['{}'.format(m)], label='{}'.format(m),linewidth=0.5, color=color)
    ax.set(ylim=(3.5, 10.5))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('ΔT warm and cold well (°C)')
    ax.grid(axis='y', linewidth=0.5, color='lightgray', zorder=0)
# plt.savefig(os.path.join(Directories.fig_dir, 'clusters_T_60.svg'), format='svg')
plt.show()

''' plot pairwise parameter comparison '''
#%% guarantees square plot
parameters['labels']=labels

fig, ax = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(ax)

x = parameters['Kv_aqt']
y = parameters['Kh_aqf']

#This solution is robust against figure size changes in the sense that the grid is always
# 2.8 + 0.1 + 1 = 3.9 inches wide and heigh.
# So one just needs to make sure the figure size is always large enough to host the grid.

horiz = [axes_size.Fixed(2.8), axes_size.Fixed(.1), axes_size.Fixed(1)]
vert = [axes_size.Fixed(2.8), axes_size.Fixed(.1), axes_size.Fixed(1)]
divider.set_horizontal(horiz)
divider.set_vertical(vert)

ax.scatter(x,y,color=np.array(cluster_colors)[parameters['labels']],s=10)
# ax.set_xticks(np.arange(0, 5.2E-7, 2E-7))
ax.grid(which='major',linewidth = 0.5, color='gray',zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir,'Kv aqt vs Kh aqf.svg'),format='svg')
plt.show()

''' kde plot of a single parameter (for each cluster) '''
#%%
parameters['labels']=labels
parameters_0 = parameters[parameters['labels']==0]
parameters_1 = parameters[parameters['labels']==1]
parameters_2 = parameters[parameters['labels']==2]

fig, ax = plt.subplots(figsize=(4,4))
divider = make_axes_locatable(ax)

horiz = [axes_size.Fixed(2.8), axes_size.Fixed(.1), axes_size.Fixed(1)]
vert = [axes_size.Fixed(2.8), axes_size.Fixed(.1), axes_size.Fixed(1)]
divider.set_horizontal(horiz)
divider.set_vertical(vert)

sns.kdeplot(data=parameters_0['Kh_aqf'], shade=True, bw_adjust=1.2, color=cluster_colors[0])
sns.kdeplot(data=parameters_1['Kh_aqf'], shade=True, bw_adjust=1.2, color=cluster_colors[1])
sns.kdeplot(data=parameters_2['Kh_aqf'], shade=True, bw_adjust=1.2, color=cluster_colors[2])
plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir,'kde kh_aqf.svg'),format='svg')
plt.show()
