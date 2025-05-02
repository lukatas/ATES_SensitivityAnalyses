import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa
from pyDGSA.plot import vert_pareto_plot
from pyDGSA.cluster import KMedoids
from collections import Counter
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.config import Directories

plt.rcParams["figure.dpi"] = 400

""" load transport results """
directory = os.path.join(Directories.output_dir, "HPC_Short_500")
prefix2 = "Parameters"
prefix3 = "EXCHANGE"
time = np.arange(0, (720) + 0.5, 0.5)  # times we saved results we want to plot (days)
number_of_seasons = 4

# Get a list of all directories in the output directory
directories = [
    os.path.join(directory, d)
    for d in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, d))
]  # isdir to check if item is a directory

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))
        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            # parameters = parameters.append(parameters_next, ignore_index=True)
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

    files3 = [p for p in os.listdir(d) if p.startswith(prefix3)]
    for file3 in files3:
        energy = xr.open_dataarray(os.path.join(d, file3))
        total_C = []
        total_W = []
        for s in range(number_of_seasons):
            zoomed_data_Cold = energy.sel(
                season=int("{}".format(s)), x=slice(642, 662), y=slice(510, 490)
            )
            zoomed_data_Warm = energy.sel(
                season=int("{}".format(s)), x=slice(350, 370), y=slice(510, 490)
            )

            # grid cells around wells have same size
            total_C.append(np.sum(zoomed_data_Cold.values))
            total_W.append(np.sum(zoomed_data_Warm.values))

        if directories.index(d) == 0:
            df_energy_W = pd.DataFrame(
                total_W,
                columns=[str(directories.index(d))],
                index=range(number_of_seasons),
            )
            df_energy_C = pd.DataFrame(
                total_C,
                columns=[str(directories.index(d))],
                index=range(number_of_seasons),
            )
        else:
            next_col_W = pd.DataFrame(
                total_W,
                columns=[str(directories.index(d))],
                index=range(number_of_seasons),
            )
            next_col_C = pd.DataFrame(
                total_C,
                columns=[str(directories.index(d))],
                index=range(number_of_seasons),
            )
            df_energy_W = pd.concat([df_energy_W, next_col_W], axis=1)
            df_energy_C = pd.concat([df_energy_C, next_col_C], axis=1)

parameters = parameters.drop(columns=["Unnamed: 0"])

""" clustering """
# %%
# calculate the euclidean distances between model responses
data = df_energy_C.T  # one row for each model instead of one column for each model
distances = pdist(data, metric="euclidean")
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMeans
n_clusters = 2
# clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
clusterer = KMedoids(
    n_clusters=n_clusters,
    max_iter=3000,
    tol=1e-4,
)

labels, medoids = clusterer.fit_predict(distances)

# check how many models in each cluster
print(Counter(labels))

""" DGSA calculation """
# %%
parameters = parameters.to_numpy()
parameter_names = [
    "Kh_aqf1",
    "Kh_aqf2",
    "Kv_aqf1",
    "Kv_aqf2",
    "gradient",
    "por_Taqf",
    "por_Eaqf",
    "volume",
    "longitudinal",
    "Twinter",
    "Tzomer",
    "Recharge",
]

# mean sensitivy averaged across all clusters
# output = 'mean' (default)
mean_sensitivity = dgsa(
    parameters, labels, parameter_names=parameter_names, n_boots=5000
)
print(mean_sensitivity)

""" DGSA Plots"""

fig, ax = vert_pareto_plot(mean_sensitivity, np_plot="+8")
# plt.title('Small area cold well')
plt.tight_layout()
plt.show()

# also cluster sensitivity, sensitivity matrix, etc. can be calculated in this way using the pyDGSA package
