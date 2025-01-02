#README for sub-halo infalling satellites catalog extracted from AGORA ART-I CosmoRun simulations.

# The units for all the distances are in physical kpc
# The units for all the masses are Msun

Two csv files:

1) ARRAKIHS_Infall_Satellites.csv:

This file contains properties for all 10 galaxy satellites which have completed
their first infall (at least one pericentre and apocentre) into the main halo at z<1.25:

Columns:
## These first 10 columns are computed at the moment when the satellite first crossed the virial radius
- Halo ID -> Rockstar Halo ID
- Snapshot -> Snapshot
- Redshift -> Redshift
- uid -> Unique Consistent trees ID
- mass -> Virial mass
- Stellar_mass -> Stellar mass 
- Half_mass_radius -> Half mass radius 
- virial_radius -> Virial radius
- scale_radius -> Scale radius
- R/Rvir -> Distance to the main halo in virial radius units (should be around 1)

- Sub_tree_id -> Sub-tree ID for identifying at the attached video 
- pericentre -> Radial distance to the center of the main halo at the pericentre 
- apocentre -> Radial distance to the center of the main halo at the apocentre
- z_pericentre -> Redshift when the pericentre was reached
- z_apocentre -> Redshift when the apocentre was reached
- z_pericentre_last -> Redshift when the last pericentre was reached
- z_apocentre_last -> Redshift when the last apocentre was reached

2) Main_galaxy_z0.csv:

This file contains the properties of the main halo at z=0.

- Total_mass -> Total mass inside the virial radius
- Stellar_mass -> Stellar mass inside the virial radius.
- Total_mass_100kpc -> Total mass inside 100 kpc
- Virial_radius -> Virial radius
- Scale_radius -> Scale radius


By Ramón Rodríguez-Cardoso, any question/suggestion to: ramorodr@ucm.es