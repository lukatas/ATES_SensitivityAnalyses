# -*- coding: utf-8 -*-
"""
@author: lhtas
script for distance based global sensitivity analysis (DGSA) using results from parallel simulations of flow and transport
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa
from pyDGSA.plot import vert_pareto_plot
from pyDGSA.cluster import KMedoids
from collections import Counter
from ATES_SensitivityAnalyses.Case_1.Parallel_Simulations.config import Directories


plt.rcParams["figure.dpi"] = 400

""" load transport results """

directory = os.path.join(Directories.output_dir, "LH_250")
prefix = "MTO"
prefix2 = "Parameters"
time = np.arange(0, (645) + 0.5, 0.5)  # times we saved results and want to use for DGSA

# Get a list of all directories in the output directory
directories = [
    os.path.join(directory, d)
    for d in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, d))
]  # isdir to check if item is a directory

# Iterate through each directory and read all files starting with 'MTO'
for d in directories:
    files = [f for f in os.listdir(d) if f.startswith(prefix)]

    for file in files:
        response = pd.read_csv(os.path.join(d, file))

        # times saved are not always exactly the same for each transport simulation

        exact_matches = response[response["Time (d)"].isin(time)]

        missing_times = np.setdiff1d(
            time, exact_matches["Time (d)"]
        )  # Find missing times

        if len(missing_times) > 0:
            closest_indices = np.searchsorted(
                response["Time (d)"], missing_times, side="left"
            )
            closest_rows = response.iloc[closest_indices]
            exact_matches = pd.concat([exact_matches, closest_rows], ignore_index=True)

        response = exact_matches
        response["Time (d)"] = response["Time (d)"].round(1)
        response = response.sort_values(by=["Time (d)"])

        response.set_index("Time (d)", inplace=True)

        if directories.index(d) == 0:
            response = response[
                ~response.index.duplicated()
            ]  # to check which rows are duplicates (based on indices) duplicates = response.index.duplicated() duplicate_rows = response[duplicates]
            deltaT = response[["DeltaT"]].copy()
            deltaT.rename(columns={"DeltaT": "0"}, inplace=True)

        else:
            response = response[
                ~response.index.duplicated()
            ]  # remove duplicate indices
            deltaT["{}".format(directories.index(d))] = response["DeltaT"]

    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))
        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            # parameters = parameters.append(parameters_next, ignore_index=True)
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

parameters = parameters.drop(columns=["Unnamed: 0"])

# #check if there are rows with Nan values (should not be the case)
# sum(deltaT.isnull().values.ravel())
# sum([True for idx, row in deltaT.iterrows() if any(row.isnull())])

# %% if we want to normalize the parameters (feature scaling)
# def normalize_column(column):
#     return (column - column.min()) / (column.max() - column.min())
# parameters = parameters.apply(normalize_column)

""" clustering """

# calculate the euclidean distances between model responses
deltaT = deltaT.T  # one row for each model
distances = pdist(deltaT, metric="euclidean")
distances = squareform(distances)  # create distance matrix of size models x models

# clustering of model response
n_clusters = 3
# clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels, medoids = clusterer.fit_predict(distances)

# check how many models in each cluster
print(Counter(labels))

# %%
parameters = parameters.to_numpy()
parameter_names = [
    "Kh aqf1",
    "Kh aqf2",
    "Kv aqf1",
    "Kv aqf2",
    "Grad.",
    "Tot. por.",
    "Eff. por.",
    "Long. disp.",
]
# %%

""" DGSA calculations """
parameters = parameters.to_numpy()
parameter_names = [
    "Kh aqf1",
    "Kh aqf2",
    "Kv aqf1",
    "Kv aqf2",
    "Grad.",
    "Tot. por.",
    "Eff. por.",
    "Long. disp.",
]

# mean sensitivy averaged across all clusters
mean_sensitivity = dgsa(
    parameters, labels, parameter_names=parameter_names, n_boots=5000
)
print(mean_sensitivity)

""" DGSA Plots"""

fig, ax = vert_pareto_plot(mean_sensitivity, np_plot="+8")
plt.tight_layout()
plt.show()

# also cluster sensitivity, sensitivity matrix, etc. can be calculated in this way using the pyDGSA package
