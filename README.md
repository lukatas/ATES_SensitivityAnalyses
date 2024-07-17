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

## Instructions for usage

