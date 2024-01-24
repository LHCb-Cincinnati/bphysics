import os
import time
import subprocess
import uproot
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def is_file_corrupt(file_path):
    try:
        with uproot.open(file_path) as file:
            file.keys()
        return False
    except:
        return True

def process_file(args):
    filename, input_folder, output_folder, decay_tree_line, branches_to_keep = args
    input_file_path = os.path.join(input_folder, filename)
    output_file_path = os.path.join(output_folder, filename.replace('.root', '_reduced.root'))

    if os.path.exists(output_file_path):
        return f"Skipped existing file: {output_file_path}"
    if is_file_corrupt(input_file_path):
        return f"File {filename} seems to be corrupt. Skipping."

    try:
        with uproot.open(input_file_path) as file:
            if decay_tree_line not in file:
                return f"Tree {decay_tree_line} not found in {filename}. Skipping."

            tree = file[decay_tree_line]
            data = tree.arrays(expressions=branches_to_keep, library="np")
            # Number of events before pre-selection
            num_events_before = len(data['Bu_MM'])
            # Calculations for pre-selection
            Delta_Z = data['L0_ENDVERTEX_Z'] - data['Bu_ENDVERTEX_Z']
            Delta_X = data['L0_ENDVERTEX_X'] - data['Bu_ENDVERTEX_X']
            Delta_Y = data['L0_ENDVERTEX_Y'] - data['Bu_ENDVERTEX_Y']

            Delta_X_ERR = np.sqrt(np.square(data['Bu_ENDVERTEX_XERR']) + np.square(data['L0_ENDVERTEX_XERR']))
            Delta_Y_ERR = np.sqrt(np.square(data['Bu_ENDVERTEX_YERR']) + np.square(data['L0_ENDVERTEX_YERR']))
            Delta_Z_ERR = np.sqrt(np.square(data['Bu_ENDVERTEX_ZERR']) + np.square(data['L0_ENDVERTEX_ZERR']))

            delta_x = np.divide(Delta_X, Delta_X_ERR, out=np.zeros_like(Delta_X), where=Delta_X_ERR!=0)
            delta_y = np.divide(Delta_Y, Delta_Y_ERR, out=np.zeros_like(Delta_Y), where=Delta_Y_ERR!=0)
            delta_z = np.divide(Delta_Z, Delta_Z_ERR, out=np.zeros_like(Delta_Z), where=Delta_Z_ERR!=0)

            L0_FD_CHISQ = np.square(delta_x) + np.square(delta_y) + np.square(delta_z)

            prodProbKK = np.multiply(data['h1_ProbNNk'], data['h2_ProbNNk'])

            # Apply pre-selection
            pre_select = (data['Bu_FDCHI2_OWNPV'] > 175) & (Delta_Z > 2.5) & \
                         (data['Lp_ProbNNp'] > 0.05) & (data['p_ProbNNp'] > 0.05) & \
                         (prodProbKK > 0.10) & (data['Bu_DTF_chi2'] < 30) & (data['Bu_PT'] > 3000)
            data = {branch: values[pre_select] for branch, values in data.items()}

            # Number of events after pre-selection
            num_events_after = len(data['Bu_MM'])
            # Percentage of events remaining
            percent_remaining = (num_events_after / num_events_before) * 100 if num_events_before > 0 else 0

            with uproot.recreate(output_file_path) as new_file:
                new_file[decay_tree_line] = data

            return f"Processed {filename}: {num_events_before} events before, {num_events_after} after ({percent_remaining:.2f}% remaining)"
    except Exception as e:
        return f"Error processing {filename}: {e}"
def main(year, magnet, decay_tree, ll_or_dd, num_files=None):
    # Extract the base name from the decay tree (e.g., "B2L0barPKpKm" from "B2L0barPKpKm_LL")
    decay_tree_base = decay_tree.split('_')[0]

    # Check for consistency between decay_tree and ll_or_dd arguments
    if ('_LL/' in decay_tree and ll_or_dd != 'LL') or ('_DD/' in decay_tree and ll_or_dd != 'DD'):
        raise ValueError("Inconsistency in decay_tree and ll_or_dd. Both should be LL or DD.")

    input_folder = f"/eos/lhcb/wg/BnoC/Bu2LambdaPPP/RD/restripped.data/reduced/{year}{magnet}/"
    output_folder = f"/afs/cern.ch/work/m/melashri/public/bphysics/data/{decay_tree_base}/{year}/{magnet}/{ll_or_dd}"

    # Print the output folder location
    print(f"Output files will be saved in: {output_folder}")
    # Check if the output folder exists, otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # The list of branches to keep
    branches_to_keep = ['Bu_MM', 'Bu_MMERR', 'Bu_ID', 'Bu_P', 'Bu_PT', 'Bu_PE', 'Bu_PX', 'Bu_PY', 'Bu_PZ',
                    'h1_P', 'h1_PT','h1_PE', 'h1_PX', 'h1_PY', 'h1_PZ',
                    'h1_ID', 'h1_TRACK_Type', 'h1_IPCHI2_OWNPV', 'h1_PIDK', 'h1_PIDp',
                    'h2_P', 'h2_PT', 'h2_PE', 'h2_PX','h2_PY', 'h2_PZ',
                    'h2_ID', 'h2_TRACK_Type', 'h2_IPCHI2_OWNPV', 'h2_PIDK', 'h2_PIDp',
                    'p_P', 'p_PT', 'p_PE', 'p_PX', 'p_PY', 'p_PZ',
                    'p_ID', 'p_TRACK_Type', 'p_IPCHI2_OWNPV', 'p_PIDK', 'p_PIDp',
                    'Lp_P', 'Lp_PT', 'Lp_PE', 'Lp_PX', 'Lp_PY', 'Lp_PZ', 'Lp_ID', 'Lp_TRACK_Type', 'Lp_PIDK', 'Lp_PIDp',
                    'L0_P',
                    'L0_PT', 'L0_PE', 'L0_PX', 'L0_PY', 'L0_PZ', 'L0_ID', 'Lpi_P', 'Lpi_PT', 'Lpi_PE', 'Lpi_PX', 'Lpi_PY',
                    'Lpi_PZ', 'Lpi_ID', 'Lpi_TRACK_Type', 'p_ProbNNp', 'Lpi_ProbNNpi', 'Lp_ProbNNp', 'h1_ProbNNk', 'h2_ProbNNk',
                    'Bu_FDCHI2_OWNPV', 'Bu_IPCHI2_OWNPV', 'Bu_ENDVERTEX_X', 'Bu_ENDVERTEX_Y', 'Bu_ENDVERTEX_Z', 'Bu_ENDVERTEX_XERR',
                    'Bu_ENDVERTEX_YERR', 'Bu_ENDVERTEX_ZERR', 'Bu_DTF_decayLength', 'Bu_DTF_decayLengthErr', 'Bu_DTF_ctau',
                    'Bu_DTF_ctauErr', 'Bu_DTF_status', 'Bu_DTFL0_M', 'Bu_DTFL0_MERR', 'Bu_DTFL0_ctau', 'Bu_DTFL0_ctauErr', 'Bu_DTF_nPV',
                    'Bu_DTF_chi2', 'Bu_DTF_nDOF', 'L0_ENDVERTEX_X', 'L0_ENDVERTEX_Y', 'L0_ENDVERTEX_Z', 'L0_ENDVERTEX_XERR',
                    'L0_ENDVERTEX_YERR', 'L0_ENDVERTEX_ZERR', 'L0_MM', 'L0_OWNPV_Z', 'L0_OWNPV_ZERR', 'L0_FD_OWNPV', 'L0_FDCHI2_OWNPV',
                    'Bu_L0Global_TIS', 'Bu_L0HadronDecision_TOS', 'Bu_Hlt1Global_TIS', 'Bu_Hlt1TrackMVADecision_TOS',
                    'Bu_Hlt1TwoTrackMVADecision_TOS', 'Bu_Hlt2Topo2BodyDecision_TOS', 'Bu_Hlt2Topo3BodyDecision_TOS', 'Bu_Hlt2Topo4BodyDecision_TOS',
                    'Bu_Hlt2Topo2BodyBBDTDecision_TOS', 'Bu_Hlt2Topo3BodyBBDTDecision_TOS', 'Bu_Hlt2Topo4BodyBBDTDecision_TOS',
                    'h1_PIDK', 'h1_PIDp', 'h2_PIDK', 'h2_PIDp', 'p_PIDK', 'p_PIDp',
                    'Bu_DTFL0_chi2', 'Bu_DTFL0_nDOF', 'Bu_DTFL0_status',
                    'nCandidate','totCandidates', 'EventInSequence', 'runNumber', 'eventNumber','GpsTime',
                    'Polarity',
                    'Bu_DTFL0_Lambda0_M', 'Bu_DTFL0_Lambda0_ctau', 'Bu_DTFL0_Lambda0_ctauErr',
                    'Bu_DTFL0_Lambda0_decayLength', 'Bu_DTFL0_Lambda0_decayLengthErr',
                    'Lp_ProbNNghost', 'Lpi_ProbNNghost', 'p_ProbNNghost', 'h1_ProbNNghost', 'h2_ProbNNghost',
                       ]
    # Go through all files the ends with .root
    root_files = [f for f in os.listdir(input_folder) if f.endswith('.root')]
    # Go through only the first couple files that passed by num_files
    if num_files is not None:
        root_files = root_files[:num_files]
    # Arguments for the main function
    args = [(filename, input_folder, output_folder, decay_tree, branches_to_keep) for filename in root_files]

    # Use multiprocessing with 4 cores (max on SWAN)
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, args), total=len(root_files), desc="Processing files"))

        for result in results:
            print(result)

# Possible configurations
decays = ['B2L0PbarKpKp']
decay_tree_lines = [f'{name}_{ending}' for name in decays for ending in ['LL', 'DD']]
years = ['2016', '2017']
magnet_polarizations = ['MagDown', 'MagUp']

print(f"The decay trees being processed are the following",decay_tree_lines)


# Set to a specific number for testing, or None for processing all files
num_files_for_testing = None  # or set to 16 for testing

# Capture the start time
start_time = time.time()

# Iterate through all configurations and call main
for year in years:
    for magnet in magnet_polarizations:
        for decay_tree_line in decay_tree_lines:
            ll_or_dd = 'LL' if '_LL' in decay_tree_line else 'DD'
            decay_tree = f"{decay_tree_line}/DecayTree"

            # Call the main function with the current configuration
            # If num_files_for_testing is None, all files will be processed
            print(f"Processing: Year: {year}, Magnet: {magnet}, Decay Tree: {decay_tree}, Limit: {'All files' if num_files_for_testing is None else f'{num_files_for_testing} files'}")
            main(year, magnet, decay_tree, ll_or_dd, num_files=num_files_for_testing)
# Capture the end time
end_time = time.time()

# Calculate and print the total execution time
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")