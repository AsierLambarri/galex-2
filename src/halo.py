#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:22:37 2024

@author: asier
"""

class HALO(BaseSimulationObject):
    def __init__(self, simulation_path, snapshots):
        super().__init__()
        self.simulation_path = simulation_path
        self.snapshots = snapshots
        self.zhalos = []

        # Default units and basis (can be changed independently)
        self.units = {'length': 'kpc', 'mass': 'Msun', 'velocity': 'km/s'}
        self.basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Create zHalo instances for each snapshot
        for snapshot in snapshots:
            zhalo = zHalo(fn=f"{simulation_path}/snapshot_{snapshot}.hdf5", center=[0, 0, 0], radius=1.0)
            zhalo.set_parent(self)  # Set parent as HALO
            self.zhalos.append(zhalo)

    def set_units(self, new_units):
        """Set units for the entire HALO class and propagate to zHalo and ptype instances."""
        self._set_units(new_units)
        for zhalo in self.zhalos:
            zhalo.set_units(new_units)  # Propagate to each zHalo instance

    def set_basis(self, new_basis):
        """Set coordinate basis for the entire HALO class and propagate to zHalo and ptype instances."""
        self._set_basis(new_basis)
        for zhalo in self.zhalos:
            zhalo.set_basis(new_basis)  # Propagate to each zHalo instance