![Capture6](https://github.com/user-attachments/assets/8953c781-4283-4260-96a5-b4cfb3c10ce3)

Scripts

[![DOI](https://zenodo.org/badge/830023014.svg)](https://zenodo.org/doi/10.5281/zenodo.13349120)

Datasets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13347760.svg)](https://doi.org/10.5281/zenodo.13347760)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119420.svg)](https://doi.org/10.5281/zenodo.15119420)
## 🔎 About

The scripts can be used to generate and simulate random model realizations of Aquifer Thermal Energy Storage (ATES) systems in three different hydrogeological settings. A jobscript is also provided to run the simulations in parallel (embarrasingly) on the HPC to alleviate the computational demand.

- case 1 represents a thick productive aquifer, the traditional target for ATES.
  For each simulated model realization the following parameters are sampled (Latin Hypercube) from the predefined ranges of uncertainty:

   * the hydraulic conductivity (vertical and horizontal),

   * the porosity (effective and total),

   * and the hydraulic gradient.


- case 2 represents a shallow alluvial aquifer, a more complex target for ATES.
   For each simulated model realization the following parameters are sampled (Latin Hypcercube) from the predefined ranges of uncertainty:

   * the hydraulic conductivity (vertical and horizontal),

   * the porosity (effective and total),

   * the recharge,

   * the flow rate (annual storage volume),

   * the temperature of the upper grid layer,

   * the longitudinal dispersion,

   * and the hydraulic gradient.


- case 3 represents a shallow, layered low-transmissivity aquifer, also a challenging target for ATES.
   For each simulated model realization the following parameters are sampled (Latin Hypcercube) from the predefined ranges of uncertainty:

    * the aquifer hydraulic conductivity (horizontal and vertical)

    * the aquitard hydraulic conductivity (horizontal and vertical)

    * the aquifer porosity (total and effictive)

    * the aquitard porosity (total and effective)

    * the aquifer versus aquitard layer thickness

    * the longitudinal dispersion

    * the hydraulic gradient

    * the flow rate

    * the well spacing (three scenarios: 40 m, 60 m, 80 m)


Scripts to process the output data from the random model realizations are also provided. The focus is on conducting a Distance-based Global Sensitivity Analysis (DGSA). For Case 3, the post processing additionally includes uncertainty quantification through joint probability distribution estimation.

## 📜 How to use?

To generate new model realizations the original modflow and mt3d input files from both case studies are required. These files are freely accessible on the Zenodo data repository.
Alternatively, when there is no interest in running new simulations, the choice can be made to make use of the original output for further data analysis. These are also available in the data repository.

For [`Case 1`](Case_1/Parallel_Simulations) and [`Case 2`](Case_2/Parallel_Simulations) : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13347760.svg)](https://doi.org/10.5281/zenodo.13347760)

For [`Case 3`](Case_3/Parallel_Simulations): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15119420.svg)](https://doi.org/10.5281/zenodo.15119420)

Guidelines for the scripts in this GitHub repository:

- To generate new output:

  * Scripts are available in the Parallel_Simulations folder of the case of interest (for example in [`Parallel_Simulations`](Case_1/Parallel_Simulations))
  * The file paths and uncertainty ranges for each parameter can be adjusted in the [`config.py`](Case_1/Parallel_Simulations/config.py) file that corresponds to the case of interest.
  * Output file names (and, for Case 3, well coordinates for different well spacings) should be adjusted in the corresponding FlowTransport file (for example in [`Rijkevorsel_FlowTransport.py`](Case_1/Parallel_Simulations/Rijkevorsel_FlowTransport.py)).
  * When there are issues with loading the models in flopy, guidelines are available in the FlowTransport file.
  * The simulations can also run in parallel on a desktop by adjusting the n_sim and n_cpu parameters in the corresponding [`RunParallel.py`](Case_1/Parallel_Simulations/RunParallel.py) script and running this instead of [`jobsript.pbs`](jobscript.pbs).


- To process existing output

  * Scripts are available in the Data_Processing folder of the case of interest (for example in [`Data_Processing`](Case_1/Data_Processing)).
### Software versions

The publicly available MODFLOW 2005 and MT3D-USGS were used for the groundwater flow and heat transport simulations. These were compiled for use on the HPC.

  - modflow version 1.12.00

      Harbaugh, A.W., Langevin, C.D., Hughes, J.D., Niswonger, R.N., and Konikow, L. F., 2017, MODFLOW-2005 version 1.12.00, the U.S. Geological Survey modular groundwater        model: U.S. Geological Survey Software Release, 03 February 2017, http://dx.doi.org/10.5066/F7RF5S7G

  - mt3d-usgs version 1.1.0

      Bedekar, V., Morway, E.D., Langevin, C.D., and Tonkin, M., 2016, MT3D-USGS version 1.0.0: Groundwater Solute Transport Simulator for MODFLOW: U.S. Geological Survey         Software Release, 30 September 2016, http://dx.doi.org/10.5066/F75T3HKD

## 📌 How to cite?

### The method:
- Case 1 and Case 2

Tas, L., Hartog, N., Bloemendal, M., Simpson, D., Robert, T., Thibaut, R., Zhang, L., Hermans, T. Efficiency and heat transport processes of low-temperature aquifer thermal energy storage systems: new insights from global sensitivity analyses. Geotherm Energy 13, 2 (2025). https://doi.org/10.1186/s40517-024-00326-1

- Case 3

Tas, L., Caers, J., Hermans, T. Decision-making under uncertainty for LT-ATES systems in complex subsurface settings: application to a low-transmissivity aquifer. Advances in Water Resources, 207, 105193 (2026). https://doi.org/10.1016/j.advwatres.2025.105193

### The code:
- Case 1 and 2

Tas, L. (2024). Efficiency and heat transport processes of low-temperature aquifer thermal energy storage systems: new insights from global sensitivity analyses - Supporting Code ATES_SensitivityAnalyses: version 0.1.0 [Software]. Zenodo. https://zenodo.org/doi/10.5281/zenodo.13349120

- Case 3

Tas, L. (2025). Decision-making under uncertainty for shallow geothermal systems in complex subsurface settings: application to a low-transmissivity aquifer - Supporting Code ATES_SensitivityAnalysis. [Software] Zenodo.

### The dataset:
- Case 1 and Case 2

Tas, L., Hartog, N., Bloemendal, M., Simpson, D., Robert, T., Thibaut, R., Zhang, L., Hermans, T. (2024). Efficiency and heat transport processes of low-temperature aquifer thermal energy storage systems: new insights from global sensitivity analyses - Supporting Dataset [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.13347760

- Case 3

Tas, L., Caers, J., Hermans, T. (2025) Decision-making under uncertainty for shallow geothermal systems in complex subsurface settings: application to a low-transmissivity aquifer - Supporting Dataset. [Dataset] Zenodo.https://doi.org/10.5281/zenodo.15119420
## 💭 Questions

Don't hesitate to reach out in case of questions!

[Luka Tas](https://github.com/lukatas).

## Acknowledgemets
The resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government.
