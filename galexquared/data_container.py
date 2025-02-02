import yt
import numpy as np
import dask.delayed
import dask.array as da

from tqdm import tqdm

from .config import config


class Data:
    """Data Class that contains all relevant fields. Has methods for adding new fields, multiplying fields together and
    filtering all the fields according to a given condition on a given field.
    """
    def __init__(self, yt_data_container, fields, aliases=None, old=None):
        """Initializes the Data class with fields from a yt.data_container object.
        
        Args:
        yt_data_container: A yt.data container object (yt.Sphere, yt.Cylinder, yt.AllData, etc.)
        """
        self.yt_data_container = yt_data_container
        self.old = self if old is None else old
        self.fields = {}  
        self.len = np.nan
        
        if aliases is None: aliases = fields

        self.field_names = fields
        self.aliases = np.array(aliases)
        
    def load_field(self, field):
        """
        Load a field from the yt.data_container object as a numpy array.
        
        Args:
        field: The name of the field to extract.
        """
        return self.yt_data_container[field]
        
    def add_field(self, field, field_name=None):
        """
        Add a new field to the Data object.
        
        Args:
        field: The field to add (from yt data container).
        field_name: Optional name for the new field (defaults to the field name).
        """
        if field_name is None:
            field_name = field 

        if isinstance(self.yt_data_container[field], dask.array.core.Array):
             self.fields[field_name] = self.yt_data_container[field]
        else:      
            if field_name in ["coordinates", "velocity"]:
                shape = (self.len, 3)
            else:
                shape = (self.len, )
            self.fields[field_name] = da.from_delayed(
                dask.delayed(self.load_field)(field), 
                shape=shape, 
                dtype=np.float64       
            )

    def add_derived_field(self, field, field_name):
        """
        Add a new field to the Data object.
        
        Args:
        field: The field to add (from yt data container).
        field_name: Optional name for the new field (defaults to the field name).
        """
        field_returner = lambda field: field  
        
        self.fields[field_name] = da.from_delayed(
            dask.delayed(field_returner)(field), 
            shape=field.shape, 
            dtype=field.dtype
        )


    def filter_fields(self, condition):
        """
        Filter all fields according to a condition on a specific field.
        
        Args:
        field_name: The field on which to apply the filter.
        condition: A callable (e.g., lambda x: x > 10) to filter the field.
        
        Returns:
        Filtered fields as a dictionary of {field_name: filtered_dask_array}.
        """    
        for i in tqdm(range(len(self.aliases))):
            field, alias = self.field_names[i], self.aliases[i]
            
            if alias not in self.fields.keys():
                self.add_field(field, field_name=alias)
            
        filtered_fields = {name: field[condition] for name, field in self.fields.items()}

        return Data(filtered_fields, fields=list(filtered_fields.keys()), old=self.old)

        
    def __getitem__(self, field_name):
        """
        Access fields using Data[field_name].
        
        Args:
        field_name: The name of the field to access.
        
        Returns:
        The corresponding dask.array.
        """
        if field_name not in self.fields:
            self.add_field(self.field_names[np.where(self.aliases == field_name)[0][0]], field_name=field_name)
            
        return self.fields[field_name]

    
    def compute(self, field_name):
        """Compute and return the field after evaluating lazily."""
        if field_name not in self.fields:
            self.add_field(self.field_names[np.where(self.aliases == field_name)[0][0]], field_name=field_name)

        self.fields[field_name] = da.from_delayed(
                dask.delayed(lambda field: field)(self.fields[field_name].compute()), 
                shape=self.fields[field_name].compute().shape, 
                dtype=np.float64       
            )
        self.len = self[field_name].shape[0]
        return self.fields[field_name].compute()



    def __add__(self, other):
        """Concatenate all fields in self.fields and other.fields."""
        
        if not isinstance(other, Data):
            raise ValueError("Can only concatenate with another Data object")
        
        self_fields = [self.fields[field_name] for field_name in self.fields]
        other_fields = [other.fields[field_name] for field_name in other.fields]
        
        concatenated_fields = self_fields + other_fields
        concatenated_data = da.concatenate(concatenated_fields, axis=0)
        
        all_field_names = list(self.fields.keys()) + list(other.fields.keys())
        all_aliases = np.concatenate([self.aliases, other.aliases])

        return Data(self.yt_data_container, all_field_names, aliases=all_aliases, old=self.old)


