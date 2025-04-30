# -*- coding: utf-8 -*-
"""
script to do DGSA using results from parallel simulations of flow and transport
Created on Mon July 08 09:12:04 2024
@author: lhtas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.spatial.distance import pdist, squareform
from pyDGSA.dgsa import dgsa
from pyDGSA.dgsa import dgsa_interactions
from pyDGSA.plot import vert_pareto_plot
from pyDGSA.plot import plot_cdf, bubble_plot
from pyDGSA.plot import interaction_matrix
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from collections import Counter
from ATES_SensitivityAnalyses.Case_3.Parallel_Simulations.config import Directories

plt.rcParams["svg.fonttype"] = "none"  # Keep text as editable text in svg files
plt.rcParams["figure.dpi"] = 400
plt.rcParams.update({"font.size": 14})
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.labelsize"] = 13


""" load transport results """
directory = os.path.join(Directories.output_dir, "LH_500_Delta5_40")
prefix = "HOB"
prefix2 = "Parameters"
skipped_indices = []

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

        if response["TOTIM"].iloc[-1] < 1080 * 24 * 3600:
            skipped_indices.append(
                directories.index(d)
            )  # a few models did not converge, filter them out
        elif response["Qout"].isnull().any() or response["Qout.1"].isnull().any():
            skipped_indices.append(directories.index(d))

        else:
            response.set_index("TOTIM", inplace=True)

            if directories.index(d) == 0:
                h_cwell = response[
                    ["Qout"]
                ].copy()  # (something went wrong while writing the input, hcwell and hwwell are now in Qout columns
                h_cwell.rename(columns={"Qout": "0"}, inplace=True)
                h_wwell = response[["Qout.1"]].copy()
                h_wwell.rename(columns={"Qout.1": "0"}, inplace=True)

            else:
                h_cwell["{}".format(directories.index(d))] = response["Qout"]
                h_wwell["{}".format(directories.index(d))] = response["Qout.1"]

    files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]  # parameters
    for file2 in files2:
        if directories.index(d) == 0:
            parameters = pd.read_csv(os.path.join(d, file2))

        elif directories.index(d) in skipped_indices:
            continue

        else:
            parameters_next = pd.read_csv(os.path.join(d, file2))
            # parameters = parameters.append(parameters_next, ignore_index=True)
            parameters = pd.concat([parameters, parameters_next], ignore_index=True)

parameters = parameters.drop(columns=["Unnamed: 0"])


# %% if we want to normalize the parameters (feature scaling)
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())


parameters = parameters.apply(normalize_column)

# %%
""" Clustering """
# calculate the euclidean distances between model responses
head = h_cwell.T  # one row for each model instead of one column for each model
head = head.reset_index()
head = head.drop(columns=["index"])
head.index = head.index.map(str)
distances = pdist(head, metric="euclidean")
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMeans
n_clusters = 3
clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
# clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels = clusterer.fit_predict(distances)

# check how many models in each cluster
print(Counter(labels))
# %%
# check medoids and deltaT to see which is low/high/medium (medoids in 0,1,2 order)
# cluster_names = [0,1,2]
cluster_names = ["Medium ΔH", "Large ΔH", "Small ΔH"]
cluster_colors = ["xkcd:beige", "xkcd:coral", "xkcd:blurple"]

# %%

""" DGSA calculations """

parameters = parameters.to_numpy()
parameter_names = [
    "Kh aqf",
    "Kh aqt",
    "Kv aqf",
    "Kv aqt",
    "Grad.",
    "Tot. por. aqf.",
    "Tot. por. aqt.",
    "Eff. por. aqf.",
    "Eff. por. aqt",
    "Long. disp.",
    "Thick.",
    #'deltaT_inj',
    "Flowrate",
]

# mean sensitivy averaged across all clusters
# output = 'mean' (default)
mean_sensitivity = dgsa(
    parameters, labels, parameter_names=parameter_names, n_boots=5000, confidence=True
)
print(mean_sensitivity)
# %%
# standardized sensitivity for each individual cluster
cluster_sensitivity = dgsa(
    parameters,
    labels,
    parameter_names=parameter_names,
    output="cluster_avg",
    cluster_names=cluster_names,
)
print(cluster_sensitivity)

# %%

# mean sensitivity including two-way parameter interactions (sensitivity values are averaged over each cluster and bin)

mean_interact_sensitivity = dgsa_interactions(
    parameters, labels, parameter_names=parameter_names
)
print(mean_interact_sensitivity)

# specify interactions between only a few conditional parameters (here x,y confined to one of the 5 bins)
sens = dgsa_interactions(
    parameters,
    labels,
    cond_parameters=["Grad.", "Kh aqf"],
    parameter_names=parameter_names,
    n_bins=5,
)
print(sens)

# %%
# standardized sensitvity including two-way parameters interactions (st sens values on individual cluster/bin levels)

cluster_interact_sensitivity = dgsa_interactions(
    parameters,
    labels,
    parameter_names=parameter_names,
    cluster_names=cluster_names,
    output="cluster_avg",
)
print(cluster_interact_sensitivity)

# get it for each individual bin and cluster
raw_interact_sensitivity = dgsa_interactions(
    parameters,
    labels,
    parameter_names=parameter_names,
    cluster_names=cluster_names,
    output="indiv",
)
print(raw_interact_sensitivity)

# %%

""" DGSA Plots"""

fig, ax = vert_pareto_plot(mean_sensitivity, np_plot="+13", confidence=True)
plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir, "head sens 80.svg"), format="svg")
plt.show()

# %%
# Define the colors for each cluster

fig, ax = vert_pareto_plot(
    cluster_sensitivity, np_plot=8, fmt="cluster_avg", colors=cluster_colors
)
plt.savefig(os.path.join(Directories.fig_dir, "head cluster sens 40.svg"))
plt.tight_layout()
plt.show()
# %% class-conditional cumulative distribution function

fig, ax = plot_cdf(
    parameters,
    labels,
    "Thick.",
    parameter_names=parameter_names,
    cluster_names=cluster_names,
    colors=cluster_colors,
)
plt.show()
# could also plot pdf (probability distribution function)
# %%

fig, ax = vert_pareto_plot(mean_interact_sensitivity, np_plot="+8")
plt.tight_layout()
plt.show()
# %%
fig, ax = vert_pareto_plot(
    cluster_interact_sensitivity, np_plot=20, fmt="cluster_avg", colors=cluster_colors
)
plt.tight_layout()
plt.show()
# %%
# Bars with bold ouline represent those for which the sensitivity >= 1.
fig, ax = vert_pareto_plot(
    raw_interact_sensitivity, fmt="indiv", colors=cluster_colors, np_plot=8
)
plt.tight_layout()
plt.show()
# %%
fig, ax = interaction_matrix(mean_interact_sensitivity, figsize=(5, 5), fontsize=5)
plt.savefig(os.path.join(Directories.fig_dir, "head sens 80.svg"), format="svg")
plt.tight_layout()
plt.show()
# %% bubble plot

fig, ax = bubble_plot(mean_sensitivity, int_df=mean_interact_sensitivity)
plt.tight_layout()
plt.show()
# %% plot ΔT change with time with cluster colors
head = head.T
fig, ax = plt.subplots()
models = len(directories) - len(skipped_indices)
for m in range(models):
    label = labels[m]
    color = cluster_colors[label]
    ax.plot(
        head.index / (24 * 3600),
        head["{}".format(m)] - 6.77,
        label="{}".format(m),
        linewidth=0.5,
        color=color,
    )
    ax.set(ylim=(-20, 20))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Head change (m)")
    ax.grid(axis="y", linewidth=0.5, color="lightgray", zorder=0)
# plt.savefig(os.path.join(Directories.fig_dir,'clusters_H_40.svg'),format='svg')
plt.show()
# %% get svg of legend
fig, ax = plt.subplots()
beige = mpatches.Patch(color="xkcd:beige", label="Medium ΔH(t)")
coral = mpatches.Patch(color="xkcd:coral", label="Large ΔH(t)")
blurple = mpatches.Patch(color="xkcd:blurple", label="Small ΔH(t)")
plt.legend(handles=[blurple, beige, coral])
plt.savefig(os.path.join(Directories.fig_dir, "legend_head.svg"), format="svg")
plt.show()
# %% plot parameter vs parameter

parameters["transmissivity_aqf"] = parameters["aqf_dz"] * parameters["Kh_aqf"]
parameters["transmissivity_aqf_m2/d"] = (
    parameters["aqf_dz"] * parameters["Kh_aqf"] * 24 * 3600
)
new_size = len(directories) - len(skipped_indices)
new_names = [str(i) for i in range(0, new_size)]
# Initializing with zeros and ones for example
max_head = pd.DataFrame([[0.0] * new_size], columns=new_names, index=[0])

for h in head:
    max = head[str(h)].max()
    max_head[h] = max

max_headchange = max_head - 6.77
fig, ax = plt.subplots()
for m in range(len(directories) - len(skipped_indices)):
    x = parameters["transmissivity_aqf_m2/d"][m]
    y = max_headchange.T[0][m]
    ax.scatter(x, y, label=str(m), color=cluster_colors[labels[m]], s=12)
    ax.set_xlabel("Total aqf transmissivity (m²/d)")
    ax.set_ylabel("Head change (m)")
# ax.set_xticks(np.arange(0, 1.2E-6, 0.2E-6))
ax.grid(which="major", axis="y", linewidth=0.5, color="gray", zorder=0)
# x_values = [20/24/3600] # transmissivity campus
# for value in x_values:
#     ax.axvline(x=value, color=(1.0, 0.6, 0.6), linestyle='--', zorder=0)
ax.axvline(x=20, color="red")
ax.axvline(x=40, color="green")
plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir, "head vs transm.svg"), format="svg")
plt.show()

# %% whithout loop parameter vs parameter and guarantees square plot
parameters["labels"] = labels

fig, ax = plt.subplots(figsize=(4, 4))
divider = make_axes_locatable(ax)

x = parameters["Kv_aqt"]
y = parameters["Kh_aqf"]

# This solution is robust against figure size changes in the sense that the grid is always
# 2.8 + 0.1 + 1 = 3.9 inches wide and heigh.
# So one just needs to make sure the figure size is always large enough to host the grid.
horiz = [axes_size.Fixed(2.8), axes_size.Fixed(0.1), axes_size.Fixed(1)]
vert = [axes_size.Fixed(2.8), axes_size.Fixed(0.1), axes_size.Fixed(1)]
divider.set_horizontal(horiz)
divider.set_vertical(vert)

ax.scatter(x, y, color=np.array(cluster_colors)[parameters["labels"]], s=10)
# ax.set_xticks(np.arange(0, 5.2E-7, 2E-7))
ax.grid(which="major", linewidth=0.5, color="gray", zorder=0)

plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir, "Kv aqt vs Kh aqf.svg"), format="svg")
plt.show()
# %% 3D plot
fig = plt.figure()
ax = plt.axes(projection="3d")
for m in range(len(directories) - len(skipped_indices)):
    x = parameters["transmissivity_aqf"][m]
    y = max_head.T[0][m]
    z = parameters["flowrate"][m]
    ax.scatter(x, y, z, label=str(m), color=cluster_colors[labels[m]])
ax.set_xlabel("transmissivity")
ax.set_ylabel("max_head")
ax.set_zlabel("flowrate")
ax.set_title("parameter interactions")

plt.show()


# %% interactive 3d plot without for loop

indices = [i for i in range(new_size)]

x = parameters["transmissivity_aqf_m2/d"][indices]
y = max_headchange.T[0][indices]
z = parameters["flowrate"][indices]
colors = [cluster_colors[labels[i]] for i in indices]

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=5, color=colors, colorscale="Viridis", opacity=0.8),
        ),
        go.Scatter3d(
            x=[50, 50],  # x (Transmissivity) is fixed at 50
            y=[np.min(y), np.max(y)],  # y values to span the full range
            z=[np.min(z), np.min(z)],  # z values to span the full range
            mode="lines",
            line=dict(color="red", width=5),
            name="Transmissivity = 50",
        ),
        go.Scatter3d(
            x=[50, 50],  # x (Transmissivity) is fixed at 50
            y=[np.min(y), np.min(y)],  # y values to span the full range
            z=[np.min(z), np.max(z)],  # z values to span the full range
            mode="lines",
            line=dict(color="red", width=5),
            name="Transmissivity = 50",
        ),
        go.Scatter3d(
            x=[50, 50],  # x (Transmissivity) is fixed at 50
            y=[np.max(y), np.max(y)],  # y values to span the full range
            z=[np.min(z), np.max(z)],  # z values to span the full range
            mode="lines",
            line=dict(color="red", width=5),
            name="Transmissivity = 50",
        ),
        go.Scatter3d(
            x=[50, 50],  # x (Transmissivity) is fixed at 50
            y=[np.min(y), np.max(y)],  # y values to span the full range
            z=[np.max(z), np.max(z)],  # z values to span the full range
            mode="lines",
            line=dict(color="red", width=5),
            name="Transmissivity = 50",
        ),
    ]
)

fig.update_layout(
    title="Parameter Interactions",
    scene=dict(
        xaxis_title="Transmissivity (m²/d)",
        yaxis_title="Max Head (mTAW)",
        zaxis_title="Flowrate (m³/h)",
    ),
)

# Show the interactive plot
fig.write_html("first_figure_2.0.html", auto_open=True)
# %% KDE of faces 3d plot

y.index = y.index.astype("float64")
data_XY = pd.concat([x, y], axis=1)
data_XY = data_XY.rename(columns={0: "max_head_mTAW"})
sns.kdeplot(
    data=data_XY,
    x="transmissivity_aqf_m2/d",
    y="max_head_mTAW",
    fill=True,
    thresh=0,
    levels=100,
    cmap="viridis",
    bw_adjust=1.2,
)
plt.show()

data_XZ = pd.concat([x, z], axis=1)
sns.kdeplot(
    data=data_XZ,
    x="transmissivity_aqf_m2/d",
    y="flowrate",
    fill=True,
    thresh=0,
    levels=100,
    cmap="viridis",
    bw_adjust=1.2,
)
plt.show()

data_YZ = pd.concat([y, z], axis=1)
data_YZ = data_YZ.rename(columns={0: "max_head_mTAW"})
sns.kdeplot(
    data=data_YZ,
    x="max_head_mTAW",
    y="flowrate",
    fill=True,
    thresh=0,
    levels=100,
    cmap="viridis",
    bw_adjust=1.2,
)
## add datapoints to plot
# scatter= plt.scatter(
#     data=data_YZ,
#     x='max_head_mTAW',
#     y='flowrate',
#     c=x,
#     cmap='Greens',  # Color the points to make them visible
#     edgecolor='black',
#     s=15  # Adjust size of points if needed
# )
# plt.colorbar(scatter, label='Transmissivity total aquifer (m²/d)')
plt.show()

# #%% thanks chat GPT ;)
# # Step 1: Extract the specific transmissivity value
# transmissivity_value = data_XY[(data_XY['transmissivity_aqf_m2/d']>21) &
#                                (data_XY['transmissivity_aqf_m2/d']<22)].reset_index()['transmissivity_aqf_m2/d'][0]
#
# # Create a 2D grid for evaluation
# x_grid = np.linspace(data_XY['transmissivity_aqf_m2/d'].min(),
#                      data_XY['transmissivity_aqf_m2/d'].max(), 100)
# y_grid = np.linspace(data_XY['max_head_mTAW'].min(),
#                      data_XY['max_head_mTAW'].max(), 100)
# X, Y = np.meshgrid(x_grid, y_grid)
#
# # Fit KDE using the data
# kde = gaussian_kde(data_XY[['transmissivity_aqf_m2/d', 'max_head_mTAW']].T)
#
# # Evaluate the KDE on the grid
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(kde(positions), X.shape)
#
# # Extract a slice along the specific 'transmissivity_aqf_m2/d' value
# x_slice = transmissivity_value  # Adjust this to your desired value
# y_values = y_grid
# z_slice = kde(np.vstack([np.full_like(y_values, x_slice), y_values]))
#
# # Step 2: Compute the CDF for the KDE slice using cumulative trapezoidal integration
# cdf_slice = cumtrapz(z_slice, y_values, initial=0)
#
# # Step 3: Create a figure with two y-axes
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# # Plot the 2D KDE
# ax1.contourf(X, Y, Z, cmap='viridis')
# ax1.set_title("2D KDE")
# ax1.set_xlabel("Total transmissivity aquifer (m²/d)")
# ax1.set_ylabel("Head change (m)")
#
# # Plot the slice on the left y-axis
# ax2.plot(y_values, z_slice, label=f'Slice at transmissivity={x_slice.round()} m²/d', color='blue')
# ax2.set_title("1D Slice of KDE")
# ax2.set_xlabel("Head change (m)")
# ax2.set_ylabel("Density", color='blue')
#
# # Step 4: Plot the CDF on the right y-axis
# ax3 = ax2.twinx()
# ax3.plot(y_values, cdf_slice, label='CDF', color='green')
# ax3.set_ylabel("Cumulative Distribution Function (CDF)", color='green')
#
# # Step 5: Adjust layout and show the plot
# plt.tight_layout()
# plt.show()

# %% kde plot of a single parameter (for each cluster)
parameters["labels"] = labels
parameters_0 = parameters[parameters["labels"] == 0]
parameters_1 = parameters[parameters["labels"] == 1]
parameters_2 = parameters[parameters["labels"] == 2]

fig, ax = plt.subplots(figsize=(4, 4))
divider = make_axes_locatable(ax)

horiz = [axes_size.Fixed(2.8), axes_size.Fixed(0.1), axes_size.Fixed(1)]
vert = [axes_size.Fixed(2.8), axes_size.Fixed(0.1), axes_size.Fixed(1)]
divider.set_horizontal(horiz)
divider.set_vertical(vert)

sns.kdeplot(
    data=parameters_0["Kh_aqf"], shade=True, bw_adjust=1.2, color=cluster_colors[0]
)
sns.kdeplot(
    data=parameters_1["Kh_aqf"], shade=True, bw_adjust=1.2, color=cluster_colors[1]
)
sns.kdeplot(
    data=parameters_2["Kh_aqf"], shade=True, bw_adjust=1.2, color=cluster_colors[2]
)
plt.tight_layout()
plt.savefig(os.path.join(Directories.fig_dir, "kde kh_aqf.svg"), format="svg")
plt.show()

# %% 3d to 2d

# Step 1: Create a grid for the evaluation
transmissivity_grid = np.linspace(
    data_XY["transmissivity_aqf_m2/d"].min(),
    data_XY["transmissivity_aqf_m2/d"].max(),
    100,
)
head_change_grid = np.linspace(
    data_XY["max_head_mTAW"].min(), data_XY["max_head_mTAW"].max(), 100
)
flowrate_grid = np.linspace(data_YZ["flowrate"].min(), data_YZ["flowrate"].max(), 100)

# Create a 3D grid
X, Y, Z = np.meshgrid(transmissivity_grid, head_change_grid, flowrate_grid)

# Step 2: Prepare data for KDE fitting
data_combined = np.vstack(
    [data_XY["transmissivity_aqf_m2/d"], data_XY["max_head_mTAW"], data_XZ["flowrate"]]
).T

# Step 3: Fit KDE using the combined data
kde = gaussian_kde(data_combined.T)

# Step 4: Evaluate the KDE on the grid
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
density = kde(positions).reshape(X.shape)

# Step 5: Extract a slice along the specific flow rate value
flowrate_value = data_YZ[
    (data_YZ["flowrate"] > 3.4) & (data_YZ["flowrate"] < 3.5)
].reset_index()["flowrate"][0]

slice_index = (
    np.abs(flowrate_grid - flowrate_value)
).argmin()  # Get the index of the closest value
density_slice = density[
    :, :, slice_index
]  # This will give you the density for the specific flow rate


# %% constrain kde with estimate of transmissivity
transmissivity_value = data_XY[
    (data_XY["transmissivity_aqf_m2/d"] > 21)
    & (data_XY["transmissivity_aqf_m2/d"] < 22)
].reset_index()["transmissivity_aqf_m2/d"][0]
transmissivity_index = np.argmin(np.abs(transmissivity_grid - transmissivity_value))

# Extract the density values for the chosen transmissivity (column in density_slice)
density_for_chosen_transmissivity = density_slice[transmissivity_index, :]

# %% plots
# Plot the density values against max head change
plt.plot(head_change_grid, density_for_chosen_transmissivity)
plt.xlabel("Max Head Change")
plt.ylabel("Density")
plt.title(
    f"Flowrate = {flowrate_value.round(2)} m³/h, Transmissivity = {transmissivity_value.round()}"
)
plt.savefig(os.path.join(Directories.fig_dir, "kde fl 3.svg"), format="svg")
plt.tight_layout()
plt.show()

# 2D slice for a specific flow rate with indication of transmissivity estimate
fig, ax = plt.subplots(figsize=(7, 6))
contour = ax.contourf(
    transmissivity_grid, head_change_grid, density_slice.T, cmap="viridis"
)
ax.set_title(f"2D Slice at Flowrate = {flowrate_value.round(1)} m³/h")
ax.set_xlabel("Transmissivity (m²/d)")
ax.set_ylabel("Head Change (m)")
plt.savefig(
    os.path.join(Directories.fig_dir, "kde head vs trans fl 3.svg"), format="svg"
)
plt.axvline(transmissivity_value, color="white", linewidth=2)
plt.tight_layout()
plt.show()

# %% calculate probability of exceeding a certain max head change

# normalize area under curve
area_under_curve = np.trapz(density_for_chosen_transmissivity, head_change_grid)
normalized_density = density_for_chosen_transmissivity / area_under_curve

plt.plot(head_change_grid, normalized_density)
plt.xlabel("Max Head Change")
plt.ylabel("Normalized Density")
plt.title(
    f"Flowrate = {flowrate_value.round(2)} m³/h, Transmissivity = {transmissivity_value.round()}"
)
plt.tight_layout()
plt.show()

# cumulative distribution function (CDF)
cumulative_density = np.cumsum(normalized_density) * np.diff(
    head_change_grid, prepend=head_change_grid[0]
)

# Find the index where the CDF reaches or exceeds 0.5
p50_index = np.searchsorted(cumulative_density, 0.5)

# Get the corresponding head change value
p50_value = head_change_grid[p50_index]

print(f"P50 Max Head Change: {p50_value:.2f}")

## instead of getting head change for P50/P90, get probability of exceeding certain head change

# head change value of interest
head_value_of_interest = 7.5

# Find the index of the closest value in the head_change_grid
closest_index = np.argmin(np.abs(head_change_grid - head_value_of_interest))

# Retrieve the corresponding normalized density value
probability_density = normalized_density[closest_index]

# Ensure the lengths of head_change_grid and normalized_density match
if len(head_change_grid) != len(normalized_density):
    raise ValueError(
        "The lengths of head_change_grid and normalized_density do not match!"
    )

# Calculate the width of each interval in head_change_grid
interval_widths = np.diff(head_change_grid, prepend=head_change_grid[0])

# Compute the cumulative probability up to the head value
cumulative_probability = np.sum(
    normalized_density[: closest_index + 1] * interval_widths[: closest_index + 1]
)

print(
    f"Probability Density at Head Change {head_value_of_interest}: {probability_density:.4f}"
)
print(
    f"Cumulative Probability up to Head Change {head_value_of_interest}: {cumulative_probability:.4f}"
)

# %%
