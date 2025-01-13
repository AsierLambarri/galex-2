import yt
import numpy as np
from yt.units import dimensions
from unyt import unyt_array, unyt_quantity

Z_Solar = 0.0134


def ARTI_loader(fn):
    """ART-I custom data loader: adds correctly derived X_He, X_Metals and metallicity (Zsun units) to gas cells and stellar particles. 
    
    In frontends/art/fields.py X_He is not added, and the gas metallicity is computed as X_Metals/X_Hydrogen. For stars, the metallicity
    fields is not even computed and remains separate in metals1 and metals2.

    Metalicities re-computed as:    gas, metallicity          --->   gas, correct_metallicity_Zsun = log10[ metal_density / density / Zsun(=0.0134) ]
                                  stars, particle_metallicity ---> stars, correct_metallicity_Zsun = log10[ (metallicity1 + metallicity2) / Zsun(=0.0134) ]

    additionally, metal mass fraction and total metal mass are computed using straight forward relations, and Helium mass fraction is computed for the
    gas cells as Y = 1-X-Y (it is fixed at 0.245).



    ### TO ADD: PARTICLE FILTER FOR STAR CREATION TIME AND AGE:


    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.ARTDataset
    """
    ds = yt.load(fn)

    ### GAS ###
    ds.add_field(
        ("gas", "coordinates"),
        function=lambda field, data: unyt_array(np.vstack((data['gas', 'x'].value, data['gas', 'y'].value, data['gas', 'z'].value)).T, data['gas', 'x'].units),
        sampling_type="cell",
        units='kpc',
        dimensions=dimensions.length
    )
    ds.add_field(
        ("gas", "velocity"),
        function=lambda field, data: unyt_array(np.vstack((data['gas', 'velocity_x'].value, data['gas', 'velocity_y'].value, data['gas', 'velocity_z'].value)).T, data['gas', 'velocity_z'].units),
        sampling_type="cell",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("gas", "mass"),
        function=lambda field, data: data['gas', 'cell_mass'],
        sampling_type="cell",
        units='Msun',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "index"),
        function=lambda field, data: ds.arr(np.linspace(1, data["gas", "mass"].shape[0] - 1, num=data["gas", "mass"].shape[0], dtype=int), "") + np.maximum(data["stars", "particle_index"].max(), data["darkmatter", "particle_index"].max()),
        sampling_type="cell",
        units='',
        dimensions=dimensions.dimensionless,
        force_override=False
    )
    ds.add_field(
        ("gas", "thermal_energy"),
        function=lambda field, data: data["gas", "cell_volume"] *  data["gas", "thermal_energy_density"],
        sampling_type="cell",
        units='auto',
        dimensions=dimensions.mass * dimensions.velocity**2,
    )
    ds.add_field(
        ("gas", "metal_mass_fraction"),
        function=lambda field, data: (data["gas", "metal_ii_density"] + data["gas", "metal_ia_density"]) / data["gas", "density"] ,
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("gas", "metal_mass"),
        function=lambda field, data:  data["gas", "metal_mass_fraction"] * data["gas", "mass"] ,
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "metallicity"),
        function=lambda field, data: np.log10(data["gas", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("gas", "He_mass_fraction"),
        function=lambda field, data: 1 - data['gas', 'H_mass_fraction'] - data['gas', 'metal_mass_fraction'],
        sampling_type="cell",
        units='auto',
        dimensions=dimensions.dimensionless
    )
    ds.add_field(
        ("gas", "cell_length"),
        function=lambda field, data: data["gas", "cell_volume"]**(1/3),
        sampling_type="cell",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("gas", "softening"),
        function=lambda field, data: data["gas", "cell_length"],
        sampling_type="cell",
        units='kpc',
        dimensions=dimensions.length,
    )
    

    ### STARS ###
    ds.add_field(
        ("stars", "coordinates"),
        function=lambda field, data: data["stars", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "velocity"),
        function=lambda field, data: data["stars", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("stars", "mass"),
        function=lambda field, data: data["stars", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "initial_mass"),
        function=lambda field, data: data["stars", "particle_mass_initial"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "softening"),
        function=lambda field, data: unyt_array(0.08 * np.ones_like(data["stars", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "index"),
        function=lambda field, data: data["stars", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass_fraction"),
        function=lambda field, data: data["stars", "particle_metallicity1"] + data["stars", "particle_metallicity2"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "particle_mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )


    
    ### DM ###
    ds.add_field(
        ("darkmatter", "coordinates"),
        function=lambda field, data: data["darkmatter", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "velocity"),
        function=lambda field, data: data["darkmatter", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("darkmatter", "mass"),
        function=lambda field, data: data["darkmatter", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("darkmatter", "softening"),
        function=lambda field, data: unyt_array(0.08 * np.ones_like(data["darkmatter", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "index"),
        function=lambda field, data: data["darkmatter", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )

    return ds


def GEAR_loader(fn):
    """GEAR custom data loader: Adds correcly derived X_Metals and metallicity for stars and gas particles.
    
     Metalicities re-computed as:      PT0, Metals --->   PT0, correct_metallicity_Zsun = log10[ Metals[:,9] / Zsun(=0.0134) ]
                                   PT1, StarMetals --->   PT1, correct_metallicity_Zsun = log10[ StarMetals[:,9] / Zsun(=0.0134) ]

    additionally, metal fractions and metal masses are computed.

     Star Formation Times and Ages are computed from the provided StarFormationTime, which is, the scale factor o formation, as:

                                SFT --> t_from_a(ScaleFactor)
                               Ages --> current_time - SFT

    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.GEARDataset
    """
    ds = yt.load(fn)

    ### STARS ###
    ds.add_field(
        ("stars", "coordinates"),
        function=lambda field, data: data["PartType1", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "velocity"),
        function=lambda field, data: data["PartType1", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("stars", "mass"),
        function=lambda field, data: data["PartType1", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "initial_mass"),
        function=lambda field, data: data["PartType1", "InitialMass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "softening"),
        function=lambda field, data: unyt_array(0.08 * np.ones_like(data["PartType1", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "index"),
        function=lambda field, data: data["PartType1", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "formation_time"),
        function=lambda field, data: ds.cosmology.t_from_a(data['PartType1', 'StarFormationTime']),
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("stars", "age"),
        function=lambda field, data: ds.current_time - data["stars", "formation_time"],
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("stars", "metal_mass_fraction"),
        function=lambda field, data: data["PartType1", "StarMetals"].in_units("") if len(data["PartType1", "StarMetals"].shape) == 1 else data["PartType1", "StarMetals"][:,9].in_units(""),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )
    ds.add_field(
        ("stars", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )





    ### GAS ###
    ds.add_field(
        ("gas", "coordinates"),
        function=lambda field, data: data["PartType0", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
        force_override=True
    )
    ds.add_field(
        ("gas", "velocity"),
        function=lambda field, data: data["PartType0", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity,
        force_override=True
    )
    ds.add_field(
        ("gas", "mass"),
        function=lambda field, data: data["PartType0", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "density"),
        function=lambda field, data: data["PartType0", "Density"],
        sampling_type="local",
        units='g/cm**3',
        dimensions=dimensions.mass / dimensions.length**3,
        force_override=True
    )   
    ds.add_field(
        ("gas", "softening"),
        function=lambda field, data: unyt_array(0.08 * np.ones_like(data["PartType0", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("gas", "metal_mass_fraction"),
        function=lambda field, data: data["PartType0", "Metals"].in_units("") if len(data["PartType0", "Metals"].shape) == 1 else data["PartType0", "Metals"][:,9].in_units(""),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("gas", "metal_mass"),
        function=lambda field, data: data["gas", "metal_mass_fraction"] * data["PartType0", "Masses"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "metallicity"),
        function=lambda field, data: np.log10(data["gas", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )    
    ds.add_field(
        ("gas", "thermal_energy"),
        function=lambda field, data: data["PartType0", "InternalEnergy"] * data["PartType0", "Masses"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass * dimensions.velocity**2,
    )



    
    ### DM ###
    ds.add_field(
        ("darkmatter", "coordinates"),
        function=lambda field, data: data["PartType2", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "velocity"),
        function=lambda field, data: data["PartType2", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("darkmatter", "mass"),
        function=lambda field, data: data["PartType2", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("darkmatter", "softening"),
        function=lambda field, data: unyt_array(0.08 * np.ones_like(data["PartType2", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "index"),
        function=lambda field, data: data["PartType2", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    return ds

