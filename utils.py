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
        


def load_mc_data(mc_base_path, mc_types, decay_tree):
    # Usage: if you want to load more MC data, call load_mc_data function with appropriate parameters.
    # mc_data += load_mc_data(mc_base_path, mc_types, decay_tree)

    # Get user input for which MC type to load
    mc_idx_list = list(map(int, input("Please enter the numbers of your desired MC types (comma separated): ").split(',')))
    selected_mc_types = [mc_types[idx-1] for idx in mc_idx_list]

    # Create a list of selected MC files for each selected MC type
    mc_data = []
    for selected_mc_type in selected_mc_types:
        selected_mc_files = [f'{selected_mc_type}_{year}.root' for year in ['15MU', '16MU', '17MU', '18MU', '15MD', '16MD', '17MD', '18MD']]
        mc_data += [f'{mc_base_path}{file}:{decay_tree}/DecayTree' for file in selected_mc_files]
    
    # Print selected decay tree and MC type
    print(f"Selected decay tree: {decay_tree}")
    for mc_type in selected_mc_types:
        print(f"Selected MC type: {mc_type}")
        
    return mc_data



def calculate_signal_efficiency_and_background_rejection(cut, signal_region, left_sideband_region, right_sideband_region):

    # Apply the cut and calculate the number of signal and background events passing the cut in the signal and sideband regions
    cut_array = eval(cut)
    signal_passing_cut_signal = np.sum(signal_weights[(B_mass > signal_region[0]) & (B_mass < signal_region[1]) & cut_array])
    background_passing_cut_signal = np.sum(signal_weights[(B_mass > signal_region[0]) & (B_mass < signal_region[1]) & ~cut_array])
    signal_passing_cut_left_sideband = np.sum(signal_weights[(B_mass > left_sideband_region[0]) & (B_mass < left_sideband_region[1]) & cut_array])
    background_passing_cut_left_sideband = np.sum(signal_weights[(B_mass > left_sideband_region[0]) & (B_mass < left_sideband_region[1]) & ~cut_array])
    signal_passing_cut_right_sideband = np.sum(signal_weights[(B_mass > right_sideband_region[0]) & (B_mass < right_sideband_region[1]) & cut_array])
    background_passing_cut_right_sideband = np.sum(signal_weights[(B_mass > right_sideband_region[0]) & (B_mass < right_sideband_region[1]) & ~cut_array])

    # Calculate the signal efficiency and background rejection
    total_signal_events = np.sum(signal_weights[(B_mass > signal_region[0]) & (B_mass < signal_region[1])])
    total_background_events = (np.sum(signal_weights[(B_mass > left_sideband_region[0]) & (B_mass < left_sideband_region[1])]) +
                               np.sum(signal_weights[(B_mass > right_sideband_region[0]) & (B_mass < right_sideband_region[1])]))
    signal_efficiency = signal_passing_cut_signal / total_signal_events
    background_rejection = (background_passing_cut_left_sideband + background_passing_cut_right_sideband) / total_background_events

    return signal_efficiency, background_rejection

