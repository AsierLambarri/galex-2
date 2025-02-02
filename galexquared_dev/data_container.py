import yt
import warnings
import numpy as np
from tqdm import tqdm


class DataContainer:
    def __init__(self, yt_region, particle_type, particle_ids=None, mask=None, old=None):
        """Initialize a new DataContainer.
        
        Parameters
        ----------
        yt_region : object
            A yt.Region-like object from which fields are lazily loaded.
        particle_type : str
            A string identifying the particle type (or gas field) for which this container is built.
        mask : numpy.ndarray or None
            A boolean array of length N (the total number of data points in yt_region)
            that selects which indices are active. If None, all data points are active.
        old : DataContainer or None
            For filtered containers, a reference to the full (unfiltered) container.
        particle_ids : array-like or None
            Optional. If provided, only particles whose IDs are in this list will be selected.
            It is assumed that the underlying region provides a field named f"{particle_type}_ids".
            
        Behavior regarding the “source” field:
          - If particle_ids is provided, a particle filter is registered and added to the dataset so
            that a new particle field f"{particle_type}_source" is created, containing only those
            particles with IDs in particle_ids.
          - If particle_ids is None, then f"{particle_type}_source" is simply taken to be the same
            as the original field (i.e. no additional filtering is done at the yt level).
        """
        self.yt_region = yt_region
        self.ds = self.yt_region.ds
        self.particle_type = particle_type
        self._particle_source = f"{particle_type}_source"
        self._particle_ids = particle_ids
        self._mask = mask if mask is not None else None
        
        if self._particle_ids is not None:
            yt.add_particle_filter(
                self._particle_source, 
                function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], self._particle_ids), 
                filtered_type=self.particle_type, 
                requires=["index"]
    
            )
        else:
            yt.add_particle_filter(
                self._particle_source, 
                function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], data[pfilter.filtered_type, "index"]), 
                filtered_type=self.particle_type, 
                requires=["index"]
    
            )
        self.ds.add_particle_filter(self._particle_source)
        self.yt_region.clear_data()
        
        self._fields = {}
        self.old = old

    def __getitem__(self, field):
        """Retrieve a field by name. If not yet loaded, the field is loaded lazily
        from yt_region and then restricted using self._mask.
        
        Additionally, if the requested field matches the particle_type, the access is
        redirected to self._particle_source_field.
        """


        if field not in self._fields:
            full_data = self._load_field(field)
            if self._mask is None:
                self._mask = np.ones(len(full_data), dtype=bool)
                
            self._fields[field] = full_data[self._mask]
        return self._fields[field]
    
    def _load_field(self, field):
        """Load a field from the underlying yt_region. Replace the code below with
        the appropriate call to your yt API as needed.
        """
        data = self.yt_region[self._particle_source, field]
        return data

    def add_field(self, field_name, array):
        """Add a new field to the container.
        
        If this container is filtered (i.e. has an associated full container in self.old),
        then the field is also added to self.old. In that case, the new field data for the full
        dataset is built by padding with np.nan for indices that did not pass the filtering condition.
        
        Parameters
        ----------
        field_name : str
            The name of the field to add.
        array : numpy.ndarray
            The field data. Its length must equal the number of active (filtered) data points.
        """
        if self._mask is None:
            warnings.warn(f"Data length is still unknown. Make sure the array you provided has the correct length!")
        else:    
            current_length = np.sum(self._mask)
            if len(array) != current_length:
                raise ValueError(f"Length of provided array ({len(array)}) does not match "
                                 f"the current active data length ({current_length}).")

        self._fields[field_name] = array

        if self.old is not None:
            padded = np.full(len(self._mask), np.nan)
            padded[self._mask] = array
            self.old._fields[field_name] = padded

    def filter(self, condition):
        """Filter the data using the provided boolean condition and return a new DataContainer
        that only returns data for which the condition is True.
        
        The new filtered container will have an attribute 'old' that holds the full dataset.
        
        Parameters
        ----------
        condition : numpy.ndarray
            A boolean array whose length equals the current number of active data points.
        
        Returns
        -------
        DataContainer
            A new container instance containing only the data where condition is True.
        """
        if self._mask is None:
            warnings.warn(f"Data length is still unknown. Make sure the array you provided has the correct length!")
        else:    
            current_length = np.sum(self._mask)
            if len(condition) != current_length:
                raise ValueError(f"Length of provided array ({len(array)}) does not match "
                                 f"the current active data length ({current_length}).")

        active_indices = np.nonzero(self._mask)[0]
        new_mask = np.zeros_like(self._mask, dtype=bool)
        new_mask[active_indices] = condition

        full_container = self.old if self.old is not None else self

        new_container = DataContainer(
            self.yt_region,
            self.particle_type,
            particle_ids=self._particle_ids,
            mask=new_mask,
            old=full_container
        )
        new_container._particle_source = self._particle_source

        for field, data in self._fields.items():
            new_container._fields[field] = data[condition]
        
        return new_container

