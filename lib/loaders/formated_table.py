from astropy.table import Table

def load_ftable(fn):
    """Loads astropy tables formated with ascii.fixed_width and sep="\t". These tables are human readable but
    a bit anoying to read with astropy because of the necessary keyword arguments. This functions solves that.
    Useful for tables that need to be accessed in scripts but be readable (e.g. csv are NOT human readable).

    Equivalent to : Table.read(fn, format="ascii.fixed_width", delimiter="\t")

    Parameters
    ----------
    fn : string, required
        Path to Formated Table file

    Returns
    -------
    ftable : astropy.table
    """
    return Table.read(fn, format="ascii.fixed_width", delimiter="\t")