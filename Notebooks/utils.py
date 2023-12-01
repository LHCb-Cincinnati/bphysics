import re
import os
import uproot
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines



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
        

#--------------------------------------------------------------------------------------------------------------#



import os
import uproot

def load_mc_data(base_path, selected_mc_types=None, decay_tree=None, list_decay_types=False):
    all_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith(".root")]

    if list_decay_types:
        # Extract unique MC types from the file names
        mc_types = set([file.rsplit('_', 1)[0] for file in os.listdir(base_path) if file.endswith('.root')])

        # Display the list of unique MC types to the user
        print("Available MC types:")
        for idx, mc_type in enumerate(mc_types, 1):
            print(f"{idx}. {mc_type}")

        # List available decay types
        decay_types_per_file = {}
        for filename in all_files:
            file = uproot.open(filename)
            decay_types = set()
            for branch in file.keys():
                if 'DecayTree;1' in branch:
                    decay_type = branch.replace('/DecayTree;1', '')
                    decay_types.add(decay_type)
            decay_types_per_file[filename] = decay_types

        # Check if all files have the same decay types
        all_values = decay_types_per_file.values()
        if all(value == list(all_values)[0] for value in all_values):
            print('All files have the same decay types.')
            print('Available decay types:')
            for decay in list(all_values)[0]:
                print(decay)
        else:
            print('Not all files have the same decay types.')
            for file, decay_types in decay_types_per_file.items():
                print(f'File {file} has the decay types:')
                for decay_type in decay_types:
                    print(decay_type)
        return

    if selected_mc_types is not None:
        all_files = [os.path.join(base_path, selected_mc_type, f"{selected_mc_type}_{year}.root") for selected_mc_type in selected_mc_types for year in ['15MU', '16MU', '17MU', '18MU', '15MD', '16MD', '17MD', '18MD']]

    data = []
    for file_path in all_files:
        with uproot.open(file_path) as root_file:
            print(f"Keys in {file_path}: {root_file.keys()}")  # Add this line
            for key in root_file.keys():
                tree_name = key.split(';')[0]
                if decay_tree is None or decay_tree in tree_name:
                    # Append '/DecayTree' only if not present in the tree_name
                    suffix = "/DecayTree" if not tree_name.endswith("/DecayTree") else ""
                    data.append(f"{file_path}:{tree_name}{suffix}")
                    break  # Break the loop as soon as a match is found
    return data



#--------------------------------------------------------------------------------------------------------------#


def load_rd_data(decay_type=None, base_path=None, help=False, list_decay_types=False):
    if help:
        print("Usage: load_data(decay_type=None, base_path=None, help=False, list_decay_types=False)")
        print("decay_type: The specific decay type to load, or None to load all decay types.")
        print("base_path: The base path where the data files are located.")
        print("help: Set to True to print this help message.")
        print("list_decay_types: Set to True to list available decay types in the specified location.")
        return

    if base_path is None:
        raise ValueError("Please provide a base path for the data files.")

    all_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith(".root")]
    

    if list_decay_types:
        decay_types_per_file = {}
        for filename in all_files:
            file = uproot.open(filename)
            decay_types = set()
            for branch in file.keys():
                if 'DecayTree;1' in branch:
                    decay_type = branch.replace('/DecayTree;1', '')
                    decay_types.add(decay_type)
            decay_types_per_file[filename] = decay_types

        # Check if all files have the same decay types
        all_values = decay_types_per_file.values()
        if all(value == list(all_values)[0] for value in all_values):
            print('All files have the same decay types.')
            print('Available decay types:')
            for decay in list(all_values)[0]:
                print(decay)
        else:
            print('Not all files have the same decay types.')
            for file, decay_types in decay_types_per_file.items():
                print(f'File {file} has the decay types:')
                for decay_type in decay_types:
                    print(decay_type)
        return

    data = []
    for file_path in all_files:
        with uproot.open(file_path) as root_file:
            for key in root_file.keys():
                tree_name = key.split(';')[0]
                if decay_type is None or tree_name == decay_type:
                    data.append(f"{file_path}:{tree_name}/DecayTree")

    return data



#--------------------------------------------------------------------------------------------------------------#


def calculate_efficiency_rejection(cut, signal_region, left_sideband_region, right_sideband_region):
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

    return cut, signal_efficiency, background_rejection

#--------------------------------------------------------------------------------------------------------------#

def list_decay_types(base_path):
    """
    List all unique decay types in the ROOT files at the given base path.

    base_path: str, path to the ROOT data files
    """
    all_files = [f for f in os.listdir(base_path) if f.endswith('.root')]
    
    decay_types = set()
    
    for filename in all_files:
        file = uproot.open(os.path.join(base_path, filename))
        # Assuming that the decay type is represented as a branch in the ROOT file
        for branch in file.keys():
            # Assuming that the branch name represents the decay type
            decay_types.add(branch)
    
    print('Available decay types:')
    for decay in decay_types:
        print(decay)

#--------------------------------------------------------------------------------------------------------------#
def load_data(base_path='', data_type=None, decay_type=None, mc_type=None, help=False, list_decay_types=False):
    """
    Load data from ROOT files.

    base_path: str, path to the data files
    data_type: str, type of data ('rd' or 'mc')
    decay_type: str, specific decay type to load
    mc_type: str, specific MC type to load
    help: bool, if True, print this docstring
    list_decay_types: bool, if True, list all available decay types in the data files
    """
    if help:
        print(load_data.__doc__)
        return
    
    # If base_path is a file, wrap it in a list
    if os.path.isfile(base_path):
        all_files = [base_path]
    # Else, if base_path is a directory, get all files in it
    elif os.path.isdir(base_path):
        all_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.root')]
    else:
        print(f'The specified path does not exist: {base_path}')
        return

    if list_decay_types:
        decay_types_per_file = {}
        for filename in all_files:
            file = uproot.open(filename)
            decay_types = set()
            for branch in file.keys():
                if 'DecayTree;1' in branch:
                    decay_type = branch.replace('/DecayTree;1', '')
                    decay_types.add(decay_type)
            decay_types_per_file[filename] = decay_types

        # Check if all files have the same decay types
        all_values = decay_types_per_file.values()
        if all(value == list(all_values)[0] for value in all_values):
            print('All files have the same decay types.')
            print('Available decay types:')
            for decay in list(all_values)[0]:
                print(decay)
        else:
            print('Not all files have the same decay types.')
            for file, decay_types in decay_types_per_file.items():
                print(f'File {file} has the decay types:')
                for decay_type in decay_types:
                    print(decay_type)
        return  
        
    if data_type not in ['rd', 'mc']:
        print(f"Invalid data type: {data_type}. Data type should be either 'rd' or 'mc'.")
        return
    
    loaded_files = []
    for file in all_files:
        file_data_type = os.path.basename(file).split('_')[0]
        file_decay_type = os.path.basename(file).split('_')[1]
        file_mc_type = os.path.basename(file).rsplit('_', 1)[0]

        if (data_type and file_data_type != data_type) or \
           (decay_type and file_decay_type != decay_type) or \
           (mc_type and file_mc_type != mc_type):
            continue
        loaded_files.append(file)
    
    return loaded_files

def load_mc(mc_base_path, mc_types, decay_tree):
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


#--------------------------------------------------------------------------------------------------------------#


def subplot(from_ax, to_ax):
    to_ax.set_xlim(from_ax.get_xlim())
    to_ax.set_ylim(from_ax.get_ylim())
    to_ax.set_title(from_ax.get_title())
    to_ax.set_xlabel(from_ax.get_xlabel())
    to_ax.set_ylabel(from_ax.get_ylabel())
    
    for line in from_ax.get_lines():
        new_line = mlines.Line2D(line.get_xdata(), line.get_ydata(), color=line.get_color())
        to_ax.add_line(new_line)

    for patch in from_ax.patches:
        new_patch = mpatches.Rectangle((patch.get_x(), patch.get_y()), patch.get_width(), patch.get_height(), fill=patch.get_fill(), color=patch.get_facecolor())
        to_ax.add_patch(new_patch)
    
    handles, labels = from_ax.get_legend_handles_labels()
    to_ax.legend(handles, labels)

    to_ax.label_outer() 
