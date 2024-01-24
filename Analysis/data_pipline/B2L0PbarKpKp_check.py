import os
import uproot
import pandas as pd
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager, Lock
from tqdm.contrib.concurrent import process_map

# Function to check file integrity
def check_file_integrity(file_path, required_tree):
    try:
        with uproot.open(file_path) as file:
            if required_tree in file:
                tree = file[required_tree]

                # Check for empty tree
                if tree.num_entries == 0:
                    return "Tree has no entries"

                # Attempt to read data from each branch
                for branch in tree.keys():
                    try:
                        data = tree[branch].array()  # This might throw an error if data is corrupted
                        # Perform any additional checks on the data
                    except Exception as e:
                        return f"Error reading branch {branch}: {str(e)}"
            else:
                return "Required tree not found"
    except Exception as e:
        return f"File read error: {str(e)}"

    return None  # No issues found



# Function to check the keys in a file
def check_file_keys(file_path, required_tree, missing_tree_files):
    try:
        with uproot.open(file_path) as file:
            if required_tree not in file:
                missing_tree_files.append(os.path.basename(file_path))
                return None  # Tree not found, skip key check
            keys = file.keys()
            return keys
    except Exception as e:
        return str(e)

def process_file(args):
    file_path, required_tree, missing_tree_files = args
    key_check_result = check_file_keys(file_path, required_tree, missing_tree_files)
    integrity_result = check_file_integrity(file_path, required_tree)
    errors = []
    if isinstance(key_check_result, str):
        errors.append({'file': os.path.basename(file_path), 'error': key_check_result})
    elif isinstance(key_check_result, list):
        for key in key_check_result:
            try:
                with uproot.open(file_path) as file:
                    file[key]
            except Exception as e:
                errors.append({'file': os.path.basename(file_path), 'key': key, 'error': str(e)})

    if integrity_result:
        errors.append({'file': os.path.basename(file_path), 'error': integrity_result})

    return errors

def print_summary(error_files, missing_tree_files, error_summary, required_tree):
    print("Summary of Errors:")
    print(error_summary)

    print("\nNumber of files with errors:", len(error_files))
    if error_files:
        print("List of files with errors:\n" + "\n".join(error_files))

    print("\nNumber of files missing the tree '{}': {}".format(required_tree, len(missing_tree_files)))
    if missing_tree_files:
        print("List of files missing the tree:\n" + "\n".join(missing_tree_files))

def check_files(data_folder_path, decay_trees, year=None, magnet=None, ll_or_dd=None):
    years = [year] if year else ['2016', '2017']
    magnets = [magnet] if magnet else ['MagDown', 'MagUp']
    ll_or_dds = [ll_or_dd] if ll_or_dd else ['LL', 'DD']
    config_errors = {}  # Dictionary to hold error details per configuration

    with Pool(processes=cpu_count()) as pool:
        for decay_tree_base in decay_trees:
            for year in years:
                for magnet in magnets:
                    for ll_or_dd in ll_or_dds:
                        config_key = (decay_tree_base, year, magnet, ll_or_dd)
                        config_errors[config_key] = {'error_files': [], 'error_details': [], 'missing_tree_files': []}

                        output_folder = os.path.join(data_folder_path, decay_tree_base, year, magnet, ll_or_dd)
                        output_files = [f for f in os.listdir(output_folder) if f.endswith('.root')]
                        required_tree = f"{decay_tree_base}_{ll_or_dd}/DecayTree"

                        tasks = [(os.path.join(output_folder, filename), required_tree, config_errors[config_key]['missing_tree_files']) for filename in output_files]
                        chunksize = max(1, len(tasks) // (10 * cpu_count()))

                        with tqdm(total=len(tasks), desc=f"Processing {config_key}") as pbar:
                            for result in pool.imap(process_file, tasks, chunksize=chunksize):
                                if isinstance(result, str):
                                    config_errors[config_key]['error_files'].append(result)
                                    config_errors[config_key]['error_details'].append({'file': result, 'error': result})
                                elif result is None:
                                    continue
                                elif isinstance(result, list):
                                    for error in result:
                                        config_errors[config_key]['error_files'].append(error['file'])
                                        config_errors[config_key]['error_details'].append(error)
                                pbar.update(1)

    # Display statistics for each configuration
    for config, data in config_errors.items():
        print(f"\nConfiguration: {config}")

        # Create DataFrame from list of dictionaries
        error_summary = pd.DataFrame(data['error_details'])

        # Print the DataFrame
        print("Summary of Errors:")
        print(error_summary.to_string())  # Updated line
        print("\nNumber of files with errors:", len(data['error_files']))
        if data['error_files']:
            print("List of files with errors:\n" + "\n".join(data['error_files']))

        print("\nNumber of files missing the tree '{}': {}".format(required_tree, len(data['missing_tree_files'])))
        if data['missing_tree_files']:
            print("List of files missing the tree:\n" + "\n".join(data['missing_tree_files']))


# usage
data_folder_path = "/afs/cern.ch/work/m/melashri/public/bphysics/data/"
decay_trees = ['B2L0PbarKpKp']
#year = '2016'
#magnet = 'MagDown'
#ll_or_dd = 'LL'
#check_files(data_folder_path, decay_trees, year, magnet, ll_or_dd)
check_files(data_folder_path, decay_trees)