import re

def parse_filename(filename):
    """Finds snap numbers for format basename_number.format using regex:
    
       "(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)"

    Parameters
    ----------
    filename : str 
        Array of filenames.

    Returns
    -------
    basename, number, file_format : str, str, str
    
    """
    pattern = r"(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)"
    
    match = re.match(pattern, filename)
    
    if match:
        basename = match.group('basename')
        number = int(match.group('number'))  
        file_format = match.group('format')
        return basename, number, file_format
    else:
        raise ValueError("Filename format not recognized")


def sort_snaps(file_list):
    """Sorts files according to snapnumber using "parse_filename" function.

    Parameters
    ----------
    file_list : array[str]
        List of file names.

    Returns
    -------
    sorted_filenames : array[str]
    """
    return sorted(file_list, key=lambda x: parse_filename(x)[1])