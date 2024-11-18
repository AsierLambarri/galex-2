import yt

class Config:
    """Config class that provides configuration options for pkg. Here one can set the loader, units, conversion tables
    etc. to be used.
    """
    _instance = None  # Class-level variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        """Override the __new__ method to ensure only one instance."""
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize_config()
        return cls._instance

    def _initialize_config(self):
        """Initialize the default configuration settings. It initializes the default snap loader and data parser
        (extract data with internal units, change to working units and save relevant simulation parameters such as
        cosmology, boxsize, particle numbers etc.) that may be overrun by custom user functions if whished, 
        base units of the simulation and the working units.
        The base units are not always used, e.g. when data is loaded with defaults, i.e. using the yt package, units
        are directly in place thanks to unyt.

        All attributes may be overrun by the user.
        
        List of attributes
        ------------------
        loader : func
            Loader of data
        parser : func
            Parser of data: extracts raw data, units, parameters, ...
        base_units : dict[str : str]
            Units of the simulation.
        working_units : dict[str : str]
            Units to work with.
        """
        self.loader = default_loader
        self.parser = default_parser
        self.base_units = None
        
        self.working_units =  {
            'mass': "Msun",
            'time': "Gyr",
            'length': "kpc",
            'velocity': "km/s",
            'comoving': False
        }

    def default_loader(fn):
        """Default loader. Returns yt.dataset
        """
        return yt.load(fn)

    def default_parser(dataset):
        """Default parser: extracts data from selected region with working units, and gets relevant metadata
        """
        return None


config = Config()






