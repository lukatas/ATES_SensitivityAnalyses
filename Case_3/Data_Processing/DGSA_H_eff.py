import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import xarray as xr
from scipy import interpolate
import scipy.integrate as integrate
from sklearn.cluster import KMeans
from pyDGSA.cluster import KMedoids
from collections import Counter
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa

from Scripts.DGSA_LatinHypercube.config import Directories
plt.rcParams['figure.dpi'] = 400
plt.rcParams.update({'font.size': 14})
# specific rounding function to round to .0 or .5, examples:
# 116.504 to 116.5: 116.05 to 116.0; 153.11 to 153.0; 153.68 to 153.5
def round_custom(df):
    df['Time (d)'] = df['Time (d)'] * 10
    df['Time (d)'] = df['Time (d)'].apply(math.floor)

    for r in range(len(df['Time (d)'])):
        if df['Time (d)'][r] % 5 == 0:
            # df['Time (d)'][r]=df['Time (d)'][r]/10 (gives warning)
            df.loc[r, 'Time (d)'] = df.loc[r, 'Time (d)'] / 10
        else:
            df.loc[r, 'Time (d)'] = (df.loc[r, 'Time (d)'] - (df.loc[r, 'Time (d)'] % 5)) / 10
    return df


def load_and_process_data(directory, prefix, prefix2, prefix3, time):
    deltaT_warm, deltaT_cold, deltaT, parameters, h_cwell, h_wwell = None, None, None, None, None, None
    directories = [os.path.join(directory, d) for d in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, d))]
    skipped_indices = []
    skipped_indices_h = []# times we saved results we want to plot (days)

    # Iterate through each directory and read all files starting with 'prefix
    for idx, d in enumerate(directories):
        files = [f for f in os.listdir(d) if f.startswith(prefix)]
        files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]
        files3 = [h for h in os.listdir(d) if h.startswith(prefix3)]

        for file in files:
            response = pd.read_csv(os.path.join(d, file))

            if response['Time (d)'].iloc[-1] < 1080:
                skipped_indices.append(directories.index(d)) #a few models did not converge, filter them out
            # for distance = 60m
            elif (response['DeltaT_cold']>10000).any() or \
              (response['DeltaT_cold']<-100).any() or\
              (response['DeltaT_warm']>5.05).any() or (response['DeltaT_warm']<-4).any()\
              or (response['DeltaT_warm']<1).any()\
                    :
                skipped_indices.append(directories.index(d))
            # for distance = 40/80 m
            elif (response['DeltaT_cold'] > 10000).any() or \
                (response['DeltaT_cold'] < -100).any() or \
                (response['DeltaT_warm'] > 5.1).any() \
                :
                skipped_indices.append(directories.index(d))
            else:
                exact_matches = response[response['Time (d)'].isin(time)]
                missing_times = np.setdiff1d(time, exact_matches['Time (d)'])

                if missing_times.size > 0:
                    closest_indices = np.searchsorted(response['Time (d)'], missing_times)
                    closest_rows = response.iloc[closest_indices]
                    exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)
                response = round_custom(exact_matches).sort_values(by=['Time (d)']).set_index('Time (d)')
                response = response[~response.index.duplicated()]

                if idx == 0:
                    deltaT_warm = response[['DeltaT_warm']].rename(columns={'DeltaT_warm': '0'})
                    deltaT_cold = response[['DeltaT_cold']].rename(columns={'DeltaT_cold': '0'})
                    deltaT = response[['DeltaT']].rename(columns={'DeltaT': '0'})

                else:
                    deltaT_warm[str(idx)] = response['DeltaT_warm']
                    deltaT_cold[str(idx)] = response['DeltaT_cold']
                    deltaT[str(idx)] = response['DeltaT']

        for file2 in files2:
            if idx == 0:
                parameters = pd.read_csv(os.path.join(d, file2))

            elif idx in skipped_indices:
                continue

            else:
                parameters_next = pd.read_csv(os.path.join(d, file2))
                parameters = pd.concat([parameters, parameters_next], ignore_index=True)

        for file3 in files3:
            response = pd.read_csv(os.path.join(d, file3))

            # if response['TOTIM'].iloc[-1] < 1080 * 24 * 3600:
            #     skipped_indices_h.append(directories.index(d))
            # elif response['Qout'].isnull().any() == True or response['Qout.1'].isnull().any() == True:
            #     skipped_indices.append(directories.index(d))
            if idx in skipped_indices:
                continue

            else:
                response.set_index('TOTIM', inplace=True)

                if directories.index(d) == 0:
                    h_cwell = response[['Qout']].copy()  # (something went wrong while writing the input, hcwell and hwwell are now in Qout columns
                    h_cwell.rename(columns={'Qout': '0'}, inplace=True)
                    h_wwell = response[['Qout.1']].copy()
                    h_wwell.rename(columns={'Qout.1': '0'}, inplace=True)

                else:
                    h_cwell['{}'.format(directories.index(d))] = response['Qout']
                    h_wwell['{}'.format(directories.index(d))] = response['Qout.1']

    # need correct index for parameters for calculations energy
    all_indices = list(range(len(directories)))
    filtered_indices = [i for i in all_indices if i not in skipped_indices]

    # Create a DataFrame using the filtered indices
    parameters.index = filtered_indices

    return deltaT_warm, deltaT_cold, deltaT, parameters, h_cwell, h_wwell, directories, skipped_indices


def calculate_energy(parameters, deltaT_warm, deltaT_cold, time_discretization, directories, skipped_indices):
    Q = parameters['flowrate']  # in m3/h
    for idx, d in enumerate(directories):
        if idx in skipped_indices:
            continue
        else:
            case = pd.DataFrame({
                'DeltaT_warm': deltaT_warm[str(idx)],
                'DeltaT_cold': deltaT_cold[str(idx)]
            })

            # get for every time to which season it belongs
            case['Seasons'] = case.index // time_discretization - ((case.index%time_discretization==0) & (case.index != 0)).astype(int) #180 otherwise wrongly assigned to season 1, True has value 1, False is 0
            max_seasons = int(case['Seasons'].max()) +1

            case['DeltaT_inj'] = np.where(case.Seasons % 2 == 0, 5, 5) #need to change to paramters[deltaT_inj][idx] when deltaT is not constant for all models
            case['DeltaT_extr'] = np.where(case.Seasons % 2 == 0, case.DeltaT_cold, case.DeltaT_warm)
            case['P_extr'] = 0
            case['P_inj'] = 0
            case['month'] = np.arange(len(deltaT_warm)) // time_discretization
            case['month'] = case.index // 30

            # P:kW
            # 1.16 is volumetric heat capacity of the groundwater 4.18MJ/(m3 K) (which is the same as 1.16 kWh/(m3K))
            case['P_extr'] = Q[idx] * 1.16 * case['DeltaT_extr']
            case['P_inj'] = Q[idx] * 1.16 * case['DeltaT_inj']

            seasons = case['Seasons'].to_numpy()
            hours = case.index.to_numpy() * 24

            P_extr = case["P_extr"].to_numpy()  # get P
            P_inj = case["P_inj"].to_numpy()

            # integrate injection and extraction power over time (every time the 6 months interval) to get the energy
            E_extr = np.zeros(max_seasons)
            for s in range(max_seasons):
                when = np.where(seasons == s)[0]  # when is season s #get indices to get data from pi for this month
                if s !=0: # need to start integrating from last datapoint previous season (otherwise you miss a day)
                    when = np.insert(when, 0, when[0]-1)
                pim = P_extr[when]  # get power at month m
                dam = hours[when]  # get corresponding hours (need to integrate over this time in hours)

                # integrate here
                funpow = interpolate.interp1d(dam, pim)
                powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

                E_extr[s] = powint  # (in kWh)

            E_inj = np.zeros(max_seasons)
            for s in range(max_seasons):
                when = np.where(seasons == s)[0]  # when is the month m #get indices to get data from pi for this month
                if s !=0: # need to start integrating from last datapoint previous season (otherwise you miss a day)
                    when = np.insert(when, 0, when[0]-1)
                pim = P_inj[when]  # get power at month m
                dam = hours[when]  # get corresponding hours (need to integrate over this time in hours)

                # integrate here
                funpow = interpolate.interp1d(dam, pim)
                powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

                E_inj[s] = powint  # (in kWh)

            if idx == 0:
                # initiate empty df with 250 columns and 2 rows
                column_names = [str(i) for i in range(len(directories))]
                data = [[0.0] * 500, [0.0] * 500, [0.0] * 500, [0.0] * 500, [0.0] * 500,
                        [0.0] * 500]  # Initializing with zeros and ones for example
                E_inj_df = pd.DataFrame(data, columns=column_names, index=[0, 1, 2, 3, 4, 5])
                E_extr_df = pd.DataFrame(data, columns=column_names, index=[0, 1, 2, 3, 4, 5])

            E_inj_df[str(idx)] = E_inj  # to check you can calculate this manually as well based on input (deltaT*time (h) * flowrate (m3/h) * 1.16)
            E_extr_df[str(idx)] = E_extr

    return E_extr_df, E_inj_df

''' load transport results '''
directory = os.path.join(Directories.output_dir, 'LH_500_Delta5_60')

prefix = 'MTO'
prefix2 = 'Parameters'
prefix3 = 'HOB'
time = np.arange(0, (1080)+1, 1)  # times we saved results we want to plot (days)
number_of_seasons = 6
time_discretization = 30*6  # in days, 6 months per season

deltaT_warm, deltaT_cold, deltaT, parameters, h_cwell, h_wwell, directories, skipped_indices = load_and_process_data(directory, prefix, prefix2, prefix3, time)
E_extr_df, E_inj_df = calculate_energy(parameters, deltaT_warm, deltaT_cold, time_discretization, directories, skipped_indices)
parameters = parameters.drop(columns=['Unnamed: 0'])

#%% get clusters deltaT
# calculate the euclidean distances between model responses
deltaT = deltaT.T  # one row for each model instead of one column for each model
# reset index because we miss some values (skipped_indices)
deltaT = deltaT.reset_index()
deltaT = deltaT.drop(columns=['index'])
deltaT.index = deltaT.index.map(str)

#%% get clusters head
# calculate the euclidean distances between model responses
head = h_cwell
head = head.T
head = head.reset_index()
head = head.drop(columns=['index'])
head.index = head.index.map(str)
# one row for each model instead of one column for each model
#%%
variable  = deltaT
distances = pdist(variable, metric='euclidean')
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMeans
n_clusters = 2
clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
# clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels = clusterer.fit_predict(distances)

# check how many models in each cluster
print(Counter(labels))
#%%
cluster_colors = ['yellow', 'green', 'orange']
cluster_colors = ['royalblue', 'limegreen', 'tomato']
#%% plot data clusters
variable = variable.T
fig,ax = plt.subplots()
models = len(directories) - len(skipped_indices)
for m in range(models):
    label = labels[m]
    color = cluster_colors[label]
    ax.plot(variable.index, variable[str(m)], label=label, color=color)
    ax.set_xlabel('Time (days)')
    ax.set(ylim=(3.5, 10.5))
    ax.set_ylabel('deltaT warm and cold well (°C)')
plt.show()


#%% Efficiency ATES
#want from 0 to new_size (without missing indices because of skipped ones)
new_size = len(directories)-len(skipped_indices)
E_extr_df = E_extr_df.replace(0,np.nan).dropna(axis='columns',how="all")
E_inj_df = E_inj_df.replace(0,np.nan).dropna(axis='columns',how="all")

new_names = [str(i) for i in range(0, new_size)]
E_extr_df = E_extr_df.set_axis(new_names, axis='columns')
E_inj_df = E_inj_df.set_axis(new_names, axis='columns')


max_seasons = 6

for m in range(new_size):
    # calculate thermal recovery efficiency
    for s in range(max_seasons):
        if s > 0:
            Eff = (E_extr_df[str(m)][s] / E_inj_df[str(m)][s - 1]) * 100

            if m == 0 and s == 1:
                data = [[0.0] * new_size, [0.0] * new_size, [0.0] * new_size,
                        [0.0] * new_size,[0.0] * new_size, [0.0] * new_size]  # Initializing with zeros and ones for example
                Eff_df = pd.DataFrame(data, columns=new_names, index=[0, 1, 2, 3, 4, 5])

            Eff_df[str(m)][s] = Eff

#%% plot thermal recovery efficiency per season
fig, ax = plt.subplots()
for m in range(new_size):
    label = labels[m]
    color = cluster_colors[label]
    ax.bar(np.arange(6), Eff_df[str(m)], edgecolor=color, color='None')
    ax.set_xlabel('Seasons (6 months)')
    ax.set_ylabel('Thermal recovery efficiency (%)')
plt.show()

# fig, ax = plt.subplots(figsize=(6, 6))
# for m in range(len(labels_60)):
#     label = labels_80[m]
#     color = cluster_colors[label]
#     ax.bar(np.arange(6), Eff_df_80[str(m)], edgecolor=color, color='None')
#     ax.set_title('80 m distance')
#     ax.set_xlabel('Seasons (6 months)')
#     ax.set_ylabel('Thermal recovery efficiency (%)')
#     ax.set_ylim(30,80)
# plt.show()

#%% plot extracted thermal energy per season
fig, ax = plt.subplots()
for m in range(new_size):
    # ax.bar(np.arange(40), Eff, label='Lanes')
    ax.bar(np.arange(6), E_extr_df[str(m)], edgecolor=cluster_colors[labels[m]], color='none')
    ax.set_xlabel('Seasons (6 months)')
    ax.set_ylabel('Extracted thermal energy (kWh)')
plt.tight_layout()
plt.show()

#%%
parameters = parameters.reset_index().drop(columns='index')
#%%  compare vs efficiency
fig, ax = plt.subplots()
for m in range(new_size):
    x = (parameters['longitudinal'][m])
    y = (Eff_df.T[5][m])
    color = cluster_colors[labels[m]]
    ax.scatter(x, y, color=color)
    ax.set_xlabel('Longitudinal dispersion (m)')
    ax.set_ylabel('Thermal recovery efficiency (%)')
# y_values = [160]
# for value in y_values:
#     ax.axhline(y=value, color=(1.0, 0.6, 0.6), linestyle='--', zorder=0)
# # #     # ax.axhline(y=response['w0'][0], color=(1.0, 0.6, 0.6), zorder=1)
# # # # ax.set_ylim(-0.1e-6, 1.75e-6)
# plt.yscale('log')
plt.tight_layout()
plt.show()

#%% plot parameter vs parameter
# cluster_colors = ['palevioletred', 'dodgerblue']
parameters['transmissivity_aqf'] = parameters['aqf_dz']*parameters['Kh_aqf']
fig, ax = plt.subplots()
plt.figure(figsize=(6,6))
for m in range(new_size):
    x = (parameters['flowrate'][m])
    y = (parameters['longitudinal'][m])
    ax.scatter(x, y, label=str(m), color=cluster_colors[labels[m]])
    ax.set_xlabel('Flowrate (m3/h)')
    ax.set_ylabel('Longitudinal dispersion (m)')
# x_values = [20/24/3600] # transmissivity campus
# for value in x_values:
#     ax.axvline(x=value, color=(1.0, 0.6, 0.6), linestyle='--', zorder=0)

plt.tight_layout()
plt.show()

#%% violinplot
parameters['label']=labels
filtered_df = parameters[parameters['label'] == 1]
filtered_df1 = parameters[parameters['label'] == 0]
filtered_df2 = parameters[parameters['label'] == 2]
unfiltered = parameters
data = [unfiltered['longitudinal'], filtered_df['longitudinal'], filtered_df1['longitudinal'],filtered_df2['longitudinal']]
colors_vplot = ['grey', 'green','yellow','orange']

fig, ax= plt.subplots()
violin = ax.violinplot(data, showmeans=False, showmedians=True)
for pc, color in zip(violin['bodies'], colors_vplot):
    pc.set_facecolor(color)
    pc.set_edgecolor(color)
# Set the color of the lines
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)

ax.set_xticks([y + 1 for y in range(len(data))],
                  labels=['prior', 'good', 'medium', 'bad'])
ax.set_ylabel('longitudinal dispersion')
ax.set_title('Parameter distribution of the prior and the classes')
plt.show()
#%% plots compared to maximum head
# Initializing with zeros and ones for example
max_head = pd.DataFrame([[0.0] * new_size], columns=new_names, index=[0])
for h in head:
    max = head[str(h)].max()
    max_head[h] = max

fig, ax = plt.subplots()
for m in range(new_size):
    x = (max_head.T[0][m])
    y = (parameters['longitudinal'][m])
    color = cluster_colors[labels[m]]
    ax.scatter(x, y, color=color)
    ax.set_xlabel('Maximum head (m)')
    ax.set_ylabel('dispersion')
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots()
bar_width = 0.2
# Indices for the bars on x-axis (for 6 months)
x = np.arange(6)

# Loop over each label (cluster)
for m in range(new_size):
    label = labels[m]
    color = cluster_colors[label]

    # Plot each Eff_df side by side by adjusting their x-position
    ax.bar(x - bar_width, Eff_df_80[str(m)], width=bar_width, edgecolor=color, color='None', label=f'Eff_df_80, Cluster {label}')
    ax.bar(x, Eff_df_60[str(m)], width=bar_width, edgecolor=color, color='None', label=f'Eff_df_60, Cluster {label}')
    ax.bar(x + bar_width, Eff_df_40[str(m)], width=bar_width, edgecolor=color, color='None', label=f'Eff_df_40, Cluster {label}')

# Labeling
ax.set_xlabel('Seasons (6 months)')
ax.set_ylabel('Thermal recovery efficiency (%)')

# Show legend
ax.legend()

# Show plot
plt.show()

#%% kde plot
parameters['labels']=labels
parameters_0 = parameters[parameters['labels']==0]
parameters_1 = parameters[parameters['labels']==1]

sns.kdeplot(data=parameters_0['por_Eaqf'], shade=True, bw_adjust=1.2, color=cluster_colors[0])
sns.kdeplot(data=parameters_1['por_Eaqf'], shade=True, bw_adjust=1.2, color=cluster_colors[1])
plt.xlabel('Effective porosity aquifer (-)')
plt.show()

#%%
Eff_df_40.T['labels'] = labels_40
Eff_df_40_0 = Eff_df_40.T[Eff_df_40.T['labels']==0]
Eff_df_40_1 = Eff_df_40.T[Eff_df_40.T['labels']==1]

Eff_df_60.T['labels'] = labels_60
Eff_df_60_0 = Eff_df_60.T[Eff_df_60.T['labels']==0]
Eff_df_60_1 = Eff_df_60.T[Eff_df_60.T['labels']==1]

Eff_df_80.T['labels'] = labels_80
Eff_df_80_0 = Eff_df_80.T[Eff_df_80.T['labels']==0]
Eff_df_80_1 = Eff_df_80.T[Eff_df_80.T['labels']==1]

column_names = ['80','60','40']
data = [[0] * 6, [0] * 6, [0] * 6, [0] * 6, [0] * 6,
        [0] * 6]  # Initializing with zeros and ones for example
df_max_0 = pd.DataFrame(data, columns=column_names, index=[0, 1, 2, 3, 4, 5])
df_max_1 = pd.DataFrame(data, columns=column_names, index=[0, 1, 2, 3, 4, 5])

for col in range(6):
    max_40_0 = Eff_df_40_0[col].max()
    max_40_1 = Eff_df_40_1[col].max()
    max_60_0 = Eff_df_60_0[col].max()
    max_60_1 = Eff_df_60_1[col].max()
    max_80_0 = Eff_df_80_0[col].max()
    max_80_1 = Eff_df_80_1[col].max()

    df_max_0['80'][col] = max_80_0
    df_max_1['80'][col] = max_80_1
    df_max_0['60'][col] = max_60_0
    df_max_1['60'][col] = max_60_1
    df_max_0['40'][col] = max_40_0
    df_max_1['40'][col] = max_40_1

# Set up the bar positions
num_seasons = len(df_max_0)  # Number of seasons
bar_width = 0.35  # Width of each bar
indices = np.arange(num_seasons)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for cluster 0 and cluster 1
for i, distance in enumerate(column_names):
    # Bar positions for each cluster
    ax.bar(indices + (i * bar_width), df_max_0[distance], bar_width, label=f'Cluster 0 - {distance}', alpha=0.7)
    ax.bar(indices + (i * bar_width) + bar_width, df_max_1[distance], bar_width, label=f'Cluster 1 - {distance}', alpha=0.7)

# Adding labels and title
ax.set_xlabel('Seasons')
ax.set_ylabel('Efficiency')
ax.set_title('Efficiency by Distance and Cluster for Each Season')
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels(['Season 1', 'Season 2', 'Season 3', 'Season 4', 'Season 5', 'Season 6'])  # Customize season names
ax.legend(title='Distance and Cluster')
ax.grid(axis='y')

# Show the plot
plt.tight_layout()
plt.show()

#%% are breaktrhough ones also the ones in orange cluster head?
breakthrough = deltaT.T.index[deltaT.T[540]<5].tolist()

fig, ax = plt.subplots()
for index in breakthrough:
    y = max_head[index]-6.77
    x = index
    ax.scatter(x,y,color='grey')
    ax.set_ylim(0,20)
plt.ylabel('Head change (m)')
plt.xlabel('Indices breakthrough')
plt.tick_params(labelbottom=False)
plt.tight_layout()
plt.show()
