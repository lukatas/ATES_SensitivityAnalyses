import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy import interpolate
import scipy.integrate as integrate
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from ATES_SensitivityAnalyses.Case_3.Parallel_Simulations.config import Directories

plt.rcParams["svg.fonttype"] = "none"  # Keep text as editable text in svg files
plt.rcParams["figure.dpi"] = 400
plt.rcParams.update({"font.size": 14})
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.labelsize"] = 13


""" define functions to load data and calculate energy """


# specific rounding function to round to .0 or .5, examples:
# 116.504 to 116.5: 116.05 to 116.0; 153.11 to 153.0; 153.68 to 153.5
def round_custom(df):
    df["Time (d)"] = df["Time (d)"] * 10
    df["Time (d)"] = df["Time (d)"].apply(math.floor)

    for r in range(len(df["Time (d)"])):
        if df["Time (d)"][r] % 5 == 0:
            # df['Time (d)'][r]=df['Time (d)'][r]/10 (gives warning)
            df.loc[r, "Time (d)"] = df.loc[r, "Time (d)"] / 10
        else:
            df.loc[r, "Time (d)"] = (
                df.loc[r, "Time (d)"] - (df.loc[r, "Time (d)"] % 5)
            ) / 10
    return df


def load_and_process_data(directory, prefix, prefix2, time):
    deltaT_warm, deltaT_cold, deltaT, parameters = None, None, None, None
    directories = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]
    skipped_indices = []  # times we saved results we want to plot (days)

    # Iterate through each directory and read all files starting with 'prefix
    for idx, d in enumerate(directories):
        files = [f for f in os.listdir(d) if f.startswith(prefix)]
        files2 = [p for p in os.listdir(d) if p.startswith(prefix2)]

        for file in files:
            response = pd.read_csv(os.path.join(d, file))

            # a few models did not converge, filter them out

            if response["Time (d)"].iloc[-1] < 1080:
                skipped_indices.append(directories.index(d))

            # # use this to filter out non-convergent/bad models for distance = 60m,
            # # otherwise deactivate this part
            # elif (response['DeltaT_cold'] > 10000).any() or \
            #         (response['DeltaT_cold'] < -100).any() or \
            #         (response['DeltaT_warm'] > 5.05).any() or (response['DeltaT_warm'] < -4).any() \
            #         or (response['DeltaT_warm'] < 1).any() \
            #         :
            #     skipped_indices.append(directories.index(d))

            # # use this to filter out non-convergent/bad models for distance = 40m/80m,
            # # otherwise deactivate this part
            elif (
                (response["DeltaT_cold"] > 10000).any()
                or (response["DeltaT_cold"] < -100).any()
                or (response["DeltaT_warm"] > 5.1).any()
            ):
                skipped_indices.append(directories.index(d))
            else:
                exact_matches = response[response["Time (d)"].isin(time)]
                missing_times = np.setdiff1d(time, exact_matches["Time (d)"])

                if len(missing_times) > 0:
                    closest_indices = np.searchsorted(
                        response["Time (d)"], missing_times, side="left"
                    )
                    closest_rows = response.iloc[closest_indices]
                    exact_matches = pd.concat(
                        [exact_matches, closest_rows], ignore_index=True
                    )

                response = exact_matches
                response = round_custom(response)
                response = response.sort_values(by=["Time (d)"])
                response.set_index("Time (d)", inplace=True)

                if idx == 0:
                    response = response[~response.index.duplicated()]
                    deltaT_warm = response[["DeltaT_warm"]].rename(
                        columns={"DeltaT_warm": "0"}
                    )
                    deltaT_cold = response[["DeltaT_cold"]].rename(
                        columns={"DeltaT_cold": "0"}
                    )
                    deltaT = response[["DeltaT"]].rename(columns={"DeltaT": "0"})

                else:
                    response = response[~response.index.duplicated()]
                    deltaT_warm[str(idx)] = response["DeltaT_warm"]
                    deltaT_cold[str(idx)] = response["DeltaT_cold"]
                    deltaT[str(idx)] = response["DeltaT"]

        for file2 in files2:
            if idx == 0:
                parameters = pd.read_csv(os.path.join(d, file2))

            elif idx in skipped_indices:
                continue

            else:
                parameters_next = pd.read_csv(os.path.join(d, file2))
                parameters = pd.concat([parameters, parameters_next], ignore_index=True)

    # need correct index for parameters for calculations energy
    all_indices = list(range(len(directories)))
    filtered_indices = [i for i in all_indices if i not in skipped_indices]

    # Create a DataFrame using the filtered indices
    parameters.index = filtered_indices

    return deltaT_warm, deltaT_cold, deltaT, parameters, directories, skipped_indices


def calculate_energy(
    parameters,
    deltaT_warm,
    deltaT_cold,
    time_discretization,
    directories,
    skipped_indices,
):
    Q = parameters["flowrate"]  # in m3/h
    for idx, d in enumerate(directories):
        if idx in skipped_indices:
            continue
        else:
            case = pd.DataFrame(
                {
                    "DeltaT_warm": deltaT_warm[str(idx)],
                    "DeltaT_cold": deltaT_cold[str(idx)],
                }
            )

            # get for every time to which season it belongs
            case["Seasons"] = case.index // time_discretization - (
                (case.index % time_discretization == 0) & (case.index != 0)
            ).astype(int)
            max_seasons = int(case["Seasons"].max()) + 1

            case["DeltaT_inj"] = np.where(case.Seasons % 2 == 0, 5, 5)
            case["DeltaT_extr"] = np.where(
                case.Seasons % 2 == 0, case.DeltaT_cold, case.DeltaT_warm
            )
            case["P_extr"] = 0
            case["P_inj"] = 0
            case["month"] = np.arange(len(deltaT_warm)) // time_discretization
            case["month"] = case.index // 30

            # P:kW
            # 1.16 is volumetric heat capacity of the groundwater 4.18MJ/(m3 K) (which is the same as 1.16 kWh/(m3K))
            case["P_extr"] = Q[idx] * 1.16 * case["DeltaT_extr"]
            case["P_inj"] = Q[idx] * 1.16 * case["DeltaT_inj"]

            seasons = case["Seasons"].to_numpy()
            hours = case.index.to_numpy() * 24

            P_extr = case["P_extr"].to_numpy()  # get P
            P_inj = case["P_inj"].to_numpy()

            # integrate injection and extraction power over time (every time the 6 months interval) to get the energy
            E_extr = np.zeros(max_seasons)
            for s in range(max_seasons):
                when = np.where(seasons == s)[0]  # when is season s
                if (
                    s != 0
                ):  # need to start integrating from last datapoint previous season (otherwise you miss a day)
                    when = np.insert(when, 0, when[0] - 1)
                pim = P_extr[when]  # get power at month m
                dam = hours[
                    when
                ]  # get corresponding hours (need to integrate over this time in hours)

                # integrate here
                funpow = interpolate.interp1d(dam, pim)
                powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

                E_extr[s] = powint  # (in kWh)

            E_inj = np.zeros(max_seasons)
            for s in range(max_seasons):
                when = np.where(seasons == s)[0]
                if s != 0:
                    when = np.insert(when, 0, when[0] - 1)
                pim = P_inj[when]
                dam = hours[when]

                # integrate here
                funpow = interpolate.interp1d(dam, pim)
                powint, abse = integrate.quad(funpow, dam[0], dam[-1], limit=1000)

                E_inj[s] = powint  # (in kWh)

            if idx == 0:
                # initiate empty df with 250 columns and 2 rows
                column_names = [str(i) for i in range(len(directories))]
                data = [
                    [0.0] * 500,
                    [0.0] * 500,
                    [0.0] * 500,
                    [0.0] * 500,
                    [0.0] * 500,
                    [0.0] * 500,
                ]  # Initializing with zeros and ones for example
                E_inj_df = pd.DataFrame(
                    data, columns=column_names, index=[0, 1, 2, 3, 4, 5]
                )
                E_extr_df = pd.DataFrame(
                    data, columns=column_names, index=[0, 1, 2, 3, 4, 5]
                )

            E_inj_df[str(idx)] = E_inj
            E_extr_df[str(idx)] = E_extr

    return E_extr_df, E_inj_df


""" load transport results """

directory = os.path.join(Directories.output_dir, "LH_500_Delta5_40")

prefix = "MTO"
prefix2 = "Parameters"
time = np.arange(0, (1080) + 1, 1)  # times we saved results we want to plot (days)
number_of_seasons = 6
time_discretization = 30 * 6  # in days, 6 months per season

deltaT_warm, deltaT_cold, deltaT, parameters, directories, skipped_indices = (
    load_and_process_data(directory, prefix, prefix2, time)
)
E_extr_df, E_inj_df = calculate_energy(
    parameters,
    deltaT_warm,
    deltaT_cold,
    time_discretization,
    directories,
    skipped_indices,
)
parameters = parameters.drop(columns=["Unnamed: 0"])

""" Clustering """
# %%
# calculate the euclidean distances between model responses
deltaT = deltaT.T  # one row for each model instead of one column for each model
# reset index because we miss some values (skipped_indices)
deltaT = deltaT.reset_index()
deltaT = deltaT.drop(columns=["index"])
deltaT.index = deltaT.index.map(str)
distances = pdist(deltaT, metric="euclidean")
distances = squareform(distances)  # create distance matrix of size models x models

# different clustering method: KMedoids
n_clusters = 2
clusterer = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
# clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)

labels = clusterer.fit_predict(distances)

# check how many models in each cluster
print(Counter(labels))

cluster_colors = ["xkcd:blurple", "xkcd:coral"]

""" plot ΔT vs time with cluster colors  """
# %%
deltaT = deltaT.T
fig, ax = plt.subplots()
models = len(directories) - len(skipped_indices)
for m in range(models):
    label = labels[m]
    color = cluster_colors[label]
    ax.plot(deltaT.index, deltaT[str(m)], label=label, color=color)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("ΔT warm and cold well (°C)")
plt.show()


""" Efficiency calculation ATES """
# %%
new_size = len(directories) - len(skipped_indices)
E_extr_df = E_extr_df.replace(0, np.nan).dropna(axis="columns", how="all")
E_inj_df = E_inj_df.replace(0, np.nan).dropna(axis="columns", how="all")

new_names = [str(i) for i in range(0, new_size)]
E_extr_df = E_extr_df.set_axis(new_names, axis="columns")
E_inj_df = E_inj_df.set_axis(new_names, axis="columns")


max_seasons = 6

for m in range(new_size):
    # calculate thermal recovery efficiency
    for s in range(max_seasons):
        if s > 0:
            Eff = (E_extr_df[str(m)][s] / E_inj_df[str(m)][s - 1]) * 100

            if m == 0 and s == 1:
                data = [
                    [0.0] * new_size,
                    [0.0] * new_size,
                    [0.0] * new_size,
                    [0.0] * new_size,
                    [0.0] * new_size,
                    [0.0] * new_size,
                ]  # Initializing with zeros and ones for example
                Eff_df = pd.DataFrame(data, columns=new_names, index=[0, 1, 2, 3, 4, 5])

            Eff_df[str(m)][s] = Eff

""" plot thermal recovery efficiency per season """
fig, ax = plt.subplots()
for m in range(new_size):
    label = labels[m]
    color = cluster_colors[label]
    ax.bar(np.arange(6), Eff_df[str(m)], edgecolor=color, color="None")
    ax.set_xlabel("Seasons (6 months)")
    ax.set_ylabel("Thermal recovery efficiency (%)")
    ax.set(ylim=(30, 80))
    ax.grid(axis="y", linewidth=0.5, color="lightgray", zorder=0)
plt.savefig(os.path.join(Directories.fig_dir, "clusters_eff_40.svg"), format="svg")
plt.show()

""" plot extracted thermal energy per season """
fig, ax = plt.subplots()
for m in range(new_size):
    # ax.bar(np.arange(40), Eff, label='Lanes')
    ax.bar(
        np.arange(6),
        E_extr_df[str(m)],
        edgecolor=cluster_colors[labels[m]],
        color="none",
    )
    ax.set_xlabel("Seasons (6 months)")
    ax.set_ylabel("Extracted thermal energy (kWh)")
plt.tight_layout()
plt.show()

# %%
""" parameter vs efficiency """
parameters = parameters.reset_index().drop(columns="index")
parameters["dispersion x advection velocity"] = (
    (parameters["gradient"] / 100 * parameters["Kh_aqf"]) / parameters["por_Eaqf"]
) * parameters["longitudinal"]
fig, ax = plt.subplots()
for m in range(new_size):
    x = Eff_df.T[5][m]
    y = parameters["dispersion x advection velocity"][m]
    color = cluster_colors[labels[m]]
    ax.scatter(x, y, color=color)
    ax.set_xlabel("Thermal recovery efficiency warm well after 3 cycles (%)")
    ax.set_ylabel("dispersion x advection velocity (1/s)")
# y_values = [160]
# for value in y_values:
#     ax.axhline(y=value, color=(1.0, 0.6, 0.6), linestyle='--', zorder=0)
# # #     # ax.axhline(y=response['w0'][0], color=(1.0, 0.6, 0.6), zorder=1)
# # # # ax.set_ylim(-0.1e-6, 1.75e-6)
# plt.yscale('log')
plt.tight_layout()
plt.show()
