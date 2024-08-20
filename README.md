![Capture6](https://github.com/user-attachments/assets/8953c781-4283-4260-96a5-b4cfb3c10ce3)

[![DOI](https://zenodo.org/badge/830023014.svg)](https://zenodo.org/doi/10.5281/zenodo.13349120)
https://doi.org/10.5281/zenodo.13347760

## 🔎 About

The scripts can be used to generate and simulate random model realizations of Aquifer Thermal Energy Storage (ATES) systems in two different hydrogeological settings. A jobscript is also provided to run the simulations in parallel (embarrasingly) on the HPC to alleviate the computational demand.

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
  
   * the flow rate (annual storage volumue),
  
   * the temperature of the upper grid layer,
  
   * the longitudinal dispersion,
  
   * and the hydraulic gradient.

Scripts to process the output data from the random model realizations are also provided. The focus is on conducting a Distance-based Global Sensitivity Analysis (DGSA).

## 📜 How to use?

To generate new model realizations (FlowTransport.py) the original modflow and mt3d input files from both case studies are required. These files are freely accessible on the Zenodo data repository. https://doi.org/10.5281/zenodo.13347760

The file paths and uncertainty ranges for each parameter can be adjusted in the config.py file.
The simulations can also run in parallel on a desktop by adjusting the n_sim and n_cpu parameters in the RunParallel.py script and running this instead of the jobscript.

Alternatively, when there is no intreest in running new simulations, the choice can be made to make use of the original output for further data analysis. This includes output data of all simulations used for the paper “Efficiency and heat transport processes of LT-ATES systems: insights from distance-based global sensitivity analyses”. It can be consulted and downloaded from the Zenodo data repository. https://doi.org/10.5281/zenodo.13347760

### Software versions 

The publicly available MODFLOW 2005 and MT3D-USGS were used for the groundwater flow and heat transport simulations. These were compiled for use on the HPC.

  - modflow version 1.12.00
    
      Harbaugh, A.W., Langevin, C.D., Hughes, J.D., Niswonger, R.N., and Konikow, L. F., 2017, MODFLOW-2005 version 1.12.00, the U.S. Geological Survey modular groundwater        model: U.S. Geological Survey Software Release, 03 February 2017, http://dx.doi.org/10.5066/F7RF5S7G
    
  - mt3d-usgs version 1.1.0
    
      Bedekar, V., Morway, E.D., Langevin, C.D., and Tonkin, M., 2016, MT3D-USGS version 1.0.0: Groundwater Solute Transport Simulator for MODFLOW: U.S. Geological Survey         Software Release, 30 September 2016, http://dx.doi.org/10.5066/F75T3HKD

## 📌 How to cite?

### The method:

To be completed once published: Tas, L., Hartog, N., Bloemendal, M., Simpson, D., Robert, T., Thibaut, R., Zhang, L., Hermans, T. (...). Effciency and heat transport processes of LT-ATES systems: insights from distance-based global sensitivity analyses. Geothermics...

### The code:
Tas, L. (2024). Effciency and heat transport processes of LT-ATES systems: insights from distance-based global sensitivity analyses - Supporting Code ATES_SensitivityAnalyses: version 0.1.0 [Software]. Zenodo. [![DOI](https://zenodo.org/badge/830023014.svg)](https://zenodo.org/doi/10.5281/zenodo.13349120)

### The dataset:
Tas, L., Hartog, N., Bloemendal, M., Simpson, D., Robert, T., Thibaut, R., Zhang, L., Hermans, T. (2024). Effciency and heat transport processes of LT-ATES systems: insights from distance-based global sensitivity analyses - Supporting Dataset [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.13347760

## 💭 Questions

Don't hesitate to reach out in case of questions!
[Luka Tas](https://github.com/lukatas).

## Acknowledgemets
The resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government.  
