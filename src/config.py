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
        """Initialize the default configuration settings."""
        self.loader = None
        self.base_units = None
        
        self.working_units =  {
            'mass': "Msun",
            'time': "Gyr",
            'length': "kpc",
            'velocity': "km/s",
            'comoving': False
        }




config = Config()






