import re

def parse_best_fit_parameters(best_fit_values):
    # Convert the object to a string
    output = str(best_fit_values)

    # Use a regular expression to extract the values after the '=' sign
    values = re.findall(r'=(\S+)', output)

    # Join the values with commas
    formatted_values = ', '.join(values)

    return formatted_values


def get_file_name(year, magnet_direction):
    for file_name in file_names:
        if str(year) in file_name and magnet_direction.upper() in file_name:
            return file_name
    return None


def get_keys_by_prefix(arrs, prefix):
    """
    Get a list of keys in the Awkward Array `arrs` that start with `prefix`.

    Args:
        arrs (ak.Array): The Awkward Array to find keys in.
        prefix (str): The prefix of the keys to find.

    Returns:
        list: A list of keys that start with `prefix`.
    """
    return [key for key in arrs.fields if key.startswith(prefix)]

def print_keys(keys):
    for i in range(0, len(keys), 5):
        print(', '.join(keys[i:i+5]))
        
