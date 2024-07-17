![png](https://github.com/user-attachments/assets/e360a33a-61ee-4769-9818-3097e63e58e8)


## About

The scripts can be used to generate and simulate random model realizations of Aquifer Thermal Energy Storage (ATES) systems in two different hydrogeological settings. A jobscript is also provided to run the simulations in parallel on the HPC to alleviate the computational demand.

- case 1 represents a thick productive aquifer, the traditional target for ATES
  For each simulated model the following parameters change (Latin Hypercube sampling) change within the predefined ranges of uncertainty:
  
  the hydraulic conductivity (vertical and horizontal),
  
  the porosity (effective and total),
  
  and the hydraulic gradient.
  
- case 2 represents a shallow alluvial aquifer, a more complex target for ATES
   For each simulated model the following parameters change (Latin Hypcercube sampling) change within the predefined ranges of uncertainty:
  
   the hydraulic conductivity (vertical and horizontal),
  
   the porosity (effective and total),
  
   the recharge,
  
   the flow rate,
  
   the temperature of the top layer,
  
   the longitudinal dispersion,
  
   and the hydraulic gradient.

Scripts to process the output data for the sensitivity analysis are also provided.

## How to use?

The output data of all simulations used for the paper “Efficiency and heat transport processes of LT-ATES systems: insights from distance-based global sensitivity analyses”. can be consulted on Zenodo.

Alternatively, the choice can be made to generate new output. Therefore, the input files of the original models can be used, also accessible on Zenodo.
The file paths and uncertainty ranges for each parameter can be adjusted in the config.py file.
The simulations can also run in parallel on a desktop by adjusting the n_sim and n_cpu parameters and simply running the RunParallel.py script.

### Software versions 

The publicly available MODFLOW 2005 and MT3D-USGS were used for the groundwater flow and heat transport simulations. These were compiled for use on the HPC.

  - modflow version 1.12.00
    
      Harbaugh, A.W., Langevin, C.D., Hughes, J.D., Niswonger, R.N., and Konikow, L. F., 2017, MODFLOW-2005 version 1.12.00, the U.S. Geological Survey modular groundwater        model: U.S. Geological Survey Software Release, 03 February 2017, http://dx.doi.org/10.5066/F7RF5S7G
    
  - mt3d-usgs version 1.1.0
    
      Bedekar, V., Morway, E.D., Langevin, C.D., and Tonkin, M., 2016, MT3D-USGS version 1.0.0: Groundwater Solute Transport Simulator for MODFLOW: U.S. Geological Survey         Software Release, 30 September 2016, http://dx.doi.org/10.5066/F75T3HKD


## How to cite?


## Acknowledgemets


## Questions

Don't hesitate to reach out in case of questions!
  link naar github profiel
