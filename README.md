# ATES senstivity analyses

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

## How to use

The output data of all simulations used for the paper can be consulted on Zenodo. 

The choice can also be made to generate new output. Therefore, the input files of the original models can be used, also accessible on Zenodo.
The file paths and uncertainty ranges for each parameter can be adjusted in the config.py file.
The simulations can also run in parallel on a desktop by adjusting the n_sim and n_cpu parameters and simply running the RunParallel.py script.

## Software versions used

The publicly available MODFLOW 2005 and MT3D-USGS were used for the groundwater flow and heat transport simulations. 
  - modflow version 1.12.00
  - mt3d-usgs version 1.1.0

## Questions

Don't hesitate to reach out in case of questions!
  link naar github profiel
