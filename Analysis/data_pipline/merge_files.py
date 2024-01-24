import os
import subprocess
import uproot
import numpy as np
import awkward as ak
import gc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def concatenate_file(args):
    file_path, branches, decay_tree_line = args
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        fname = f"{file_path}:{decay_tree_line}/DecayTree"
        return uproot.concatenate(fname, branches)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def concatenate_files(file_paths, branches, decay_tree_line, use_parallel):
    """
    Concatenates files either in parallel or serially based on `use_parallel`.

    :param file_paths: List of file paths to concatenate.
    :param branches: Branches to read.
    :param decay_tree_line: Decay tree line name.
    :param use_parallel: Boolean to determine if parallel processing should be used.
    :return: Concatenated array.
    """
    if use_parallel:
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(concatenate_file, [(path, branches, decay_tree_line) for path in file_paths]),
                                total=len(file_paths), desc="Concatenating files"))
    else:
        results = [concatenate_file((path, branches, decay_tree_line)) for path in tqdm(file_paths, desc="Concatenating files")]

    # Filter out None results and concatenate
    arrays = [arr for arr in results if arr is not None]
    return ak.concatenate(arrays, axis=0) if arrays else None

def parallel_concatenate_files(file_paths, branches, decay_tree_line):
    """
    Concatenates files in parallel using multiprocessing.
    """
    with Pool(cpu_count()) as pool:
        # Wrap 'tqdm' around the 'pool.imap' for a progress bar
        results = list(tqdm(pool.imap(concatenate_file, [(path, branches, decay_tree_line) for path in file_paths]),
                            total=len(file_paths), desc="Concatenating files"))

    # Filter out None results and concatenate
    arrays = [arr for arr in results if arr is not None]
    return ak.concatenate(arrays, axis=0) if arrays else None

def get_file_paths_for_config(base_dir, decay_tree, year, magnet, ll_dd):
    """
    Generate a list of file paths for a specific configuration.

    :param base_dir: Base directory containing the data.
    :param decay_tree: Name of the decay tree.
    :param year: Specific year.
    :param magnet: Magnet configuration ('MagUp', 'MagDown').
    :param ll_dd: 'LL' or 'DD'.
    :return: List of file paths.
    """
    dir_path = os.path.join(base_dir, decay_tree, year, magnet, ll_dd)
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.root')]

def process_and_save_files_for_config(input_files, decay_tree_line, output_file, use_parallel):
    """
    Process and save files for a specific configuration.

    :param input_files: List of .root files to process.
    :param decay_tree_line: Decay tree line name.
    :param output_file: Output file path.
    :param use_parallel: Boolean to determine if parallel processing should be used.
    """
    if not input_files:
        print(f"No files to process for {output_file}")
        return

    branches_to_read = ['Bu_MM', 'Bu_MMERR', 'Bu_ID', 'Bu_P', 'Bu_PT', 'Bu_PE', 'Bu_PX', 'Bu_PY', 'Bu_PZ',
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

    concatenated_array = concatenate_files(input_files, branches_to_read, decay_tree_line, use_parallel)

    if concatenated_array is None or len(concatenated_array) == 0:
        print(f"No data to process for {output_file}")
        return

    processed_data = process_data(concatenated_array)
    save_to_root(output_file, processed_data)

    del concatenated_array
    gc.collect()

def process_data(arrs):
        ## K+ in B+ --> Lambda_bar,p,K+,K-
        h1_P   = arrs['h1_P']
        h1_PT  = arrs['h1_PT']
        h1_PE  = arrs['h1_PE']
        h1_PX  = arrs['h1_PX']
        h1_PY  = arrs['h1_PY']
        h1_PZ  = arrs['h1_PZ']
        h1_ID  = arrs["h1_ID"]
        h1_PIDK          = arrs['h1_PIDK']
        h1_PIDp          = arrs['h1_PIDp']
        h1_ProbNNk       = arrs["h1_ProbNNk"]
        h1_TRACK_Type    = arrs["h1_TRACK_Type"]
        h1_IPCHI2_OWNPV  = arrs['h1_IPCHI2_OWNPV']
        h1_ProbNNghost   = arrs['h1_ProbNNghost']
        ## K- in B+ --> Lambda_bar,p,K+,K-

        h2_P   = arrs['h2_P']
        h2_PT  = arrs['h2_PT']
        h2_PE  = arrs['h2_PE']
        h2_PX  = arrs['h2_PX']
        h2_PY  = arrs['h2_PY']
        h2_PZ  = arrs['h2_PZ']
        h2_ID  = arrs["h2_ID"]
        h2_PIDK         = arrs['h2_PIDK']
        h2_PIDp         = arrs['h2_PIDp']
        h2_ProbNNk      = arrs["h2_ProbNNk"]
        h2_TRACK_Type   = arrs["h2_TRACK_Type"]
        h2_IPCHI2_OWNPV = arrs['h2_IPCHI2_OWNPV']
        h2_ProbNNghost  = arrs['h2_ProbNNghost']
        ## p in B+ --> Lambda_bar,p,K+,K-

        p_P   = arrs['p_P']
        p_PT  = arrs['p_PT']
        p_PE  = arrs['p_PE']
        p_PX  = arrs['p_PX']
        p_PY  = arrs['p_PY']
        p_PZ  = arrs['p_PZ']
        p_ID  = arrs["p_ID"]
        p_PIDK         = arrs['p_PIDK']
        p_PIDp         = arrs['p_PIDp']
        p_TRACK_Type   = arrs["p_TRACK_Type"]
        p_ProbNNp      = arrs['p_ProbNNp']
        p_IPCHI2_OWNPV = arrs['p_IPCHI2_OWNPV']
        p_ProbNNghost  = arrs['p_ProbNNghost']
        ## proton in the Lambda_bar in B+ --> Lambda_bar,p,K+,K-

        Lp_P   = arrs['Lp_P']
        Lp_PT  = arrs['Lp_PT']
        Lp_PE  = arrs['Lp_PE']
        Lp_PX  = arrs['Lp_PX']
        Lp_PY  = arrs['Lp_PY']
        Lp_PZ  = arrs['Lp_PZ']
        Lp_ID  = arrs["Lp_ID"]
        Lp_PIDK        = arrs['Lp_PIDK']
        Lp_PIDp        = arrs['Lp_PIDp']
        Lp_TRACK_Type  = arrs["Lp_TRACK_Type"]
        Lp_ProbNNp     = arrs['Lp_ProbNNp']
        Lp_ProbNNghost = arrs['Lp_ProbNNghost']
        LL = (3 == Lp_TRACK_Type)
        DD = (5 == Lp_TRACK_Type)
        ## pion in the Lambda_bar in B+ --> Lambda_bar,p,K+,K-

        Lpi_P   = arrs['Lpi_P']
        Lpi_PT  = arrs['Lpi_PT']
        Lpi_PE  = arrs['Lpi_PE']
        Lpi_PX  = arrs['Lpi_PX']
        Lpi_PY  = arrs['Lpi_PY']
        Lpi_PZ  = arrs['Lpi_PZ']
        Lpi_ID  = arrs["Lpi_ID"]
        Lpi_TRACK_Type  = arrs["Lpi_TRACK_Type"]
        Lpi_ProbNNpi    = arrs['Lpi_ProbNNpi']
        Lpi_ProbNNghost = arrs['Lpi_ProbNNghost']


        ## the Lambda_bar in B+ --> Lambda_bar,p,K+,K-
        L0_P   = arrs['L0_P']
        L0_PT  = arrs['L0_PT']
        L0_PE  = arrs['L0_PE']
        L0_PX  = arrs['L0_PX']
        L0_PY  = arrs['L0_PY']
        L0_PZ  = arrs['L0_PZ']
        L0_ID  = arrs["L0_ID"]
        L0_MM  = arrs['L0_MM']
        ##L0_DOCA12 = arrs["L0_DOCA12"]
        Bu_FDCHI2_OWNPV = arrs['Bu_FDCHI2_OWNPV']
        L0_ENDVERTEX_X    = arrs["L0_ENDVERTEX_X"]
        L0_ENDVERTEX_Y    = arrs["L0_ENDVERTEX_Y"]
        L0_ENDVERTEX_Z    = arrs["L0_ENDVERTEX_Z"]
        L0_ENDVERTEX_XERR = arrs["L0_ENDVERTEX_XERR"]
        L0_ENDVERTEX_YERR = arrs["L0_ENDVERTEX_YERR"]
        L0_ENDVERTEX_ZERR = arrs["L0_ENDVERTEX_ZERR"]
        L0_OWNPV_Z = arrs["L0_OWNPV_Z"]
        L0_OWNPV_ZERR = arrs["L0_OWNPV_ZERR"]

        L0_FD_OWNPV = arrs["L0_FD_OWNPV"]
        L0_FDCHI2_OWNPV = arrs["L0_FDCHI2_OWNPV"]
        Bu_ENDVERTEX_X     = arrs["Bu_ENDVERTEX_X"]
        Bu_ENDVERTEX_Y     = arrs["Bu_ENDVERTEX_Y"]
        Bu_ENDVERTEX_Z     = arrs["Bu_ENDVERTEX_Z"]
        Bu_ENDVERTEX_XERR  = arrs["Bu_ENDVERTEX_XERR"]
        Bu_ENDVERTEX_YERR  = arrs["Bu_ENDVERTEX_YERR"]
        Bu_ENDVERTEX_ZERR  = arrs["Bu_ENDVERTEX_YERR"]
        Bu_IPCHI2_OWNPV    = arrs["Bu_IPCHI2_OWNPV"]
        Bu_MM              = arrs['Bu_MM']
        Bu_MMERR           = arrs['Bu_MMERR']
        Bu_ID              = arrs['Bu_ID']
        Bu_P               = arrs['Bu_P']
        Bu_PT              = arrs['Bu_PT']
        Bu_PE              = arrs['Bu_PE']
        Bu_PX              = arrs['Bu_PX']
        Bu_PY              = arrs['Bu_PY']
        Bu_PZ              = arrs['Bu_PZ']
        Bu_DTF_nPV            = arrs['Bu_DTF_nPV'] ## not indexed

        Bu_DTF_decayLength    = arrs['Bu_DTF_decayLength']
        Bu_DTF_decayLength    = Bu_DTF_decayLength[0:,0]

        Bu_DTF_decayLengthErr = arrs['Bu_DTF_decayLengthErr']
        Bu_DTF_decayLengthErr = Bu_DTF_decayLengthErr[0:,0]

        Bu_DTF_ctau           = arrs['Bu_DTF_ctau']
        Bu_DTF_ctau           = Bu_DTF_ctau[0:,0]

        Bu_DTF_ctauErr        = arrs['Bu_DTF_ctauErr']
        Bu_DTF_ctauErr        = Bu_DTF_ctauErr[0:,0]

        Bu_DTF_status         = arrs['Bu_DTF_status']
        Bu_DTF_status         = Bu_DTF_status[0:,0]

        Bu_DTF_chi2           = arrs['Bu_DTF_chi2']
        Bu_DTF_chi2           = Bu_DTF_chi2[0:,0]

        Bu_DTF_nDOF           = arrs['Bu_DTF_nDOF']
        Bu_DTF_nDOF           = Bu_DTF_nDOF[0:,0]

        Bu_DTFL0_M            = arrs['Bu_DTFL0_M']
        Bu_DTFL0_M            = Bu_DTFL0_M[0:,0]

        Bu_DTFL0_MERR         = arrs['Bu_DTFL0_MERR']
        Bu_DTFL0_MERR         = Bu_DTFL0_MERR[0:,0]

        Bu_DTFL0_ctau         = arrs['Bu_DTFL0_ctau']
        Bu_DTFL0_ctau         = Bu_DTFL0_ctau[0:,0]

        Bu_DTFL0_ctauErr      = arrs['Bu_DTFL0_ctauErr']
        Bu_DTFL0_ctauErr      = Bu_DTFL0_ctauErr[0:,0]

        Bu_DTFL0_chi2         = arrs['Bu_DTFL0_chi2']
        Bu_DTFL0_chi2         = Bu_DTFL0_chi2[0:,0]

        Bu_DTFL0_nDOF         = arrs['Bu_DTFL0_nDOF']
        Bu_DTFL0_nDOF         = Bu_DTFL0_nDOF[0:,0]

        Bu_DTFL0_status       = arrs['Bu_DTFL0_status']
        Bu_DTFL0_status       = Bu_DTFL0_status[0:,0]

        Bu_DTFL0_Lambda0_M    = arrs['Bu_DTFL0_Lambda0_M']
        Bu_DTFL0_Lambda0_M    = Bu_DTFL0_Lambda0_M[0:,0]

        Bu_DTFL0_Lambda0_ctau = arrs['Bu_DTFL0_Lambda0_ctau']
        Bu_DTFL0_Lambda0_ctau = Bu_DTFL0_Lambda0_ctau[0:,0]

        Bu_DTFL0_Lambda0_ctauErr = arrs['Bu_DTFL0_Lambda0_ctauErr']
        Bu_DTFL0_Lambda0_ctauErr = Bu_DTFL0_Lambda0_ctauErr[0:,0]

        Bu_DTFL0_Lambda0_decayLength = arrs['Bu_DTFL0_Lambda0_decayLength']
        Bu_DTFL0_Lambda0_decayLength = Bu_DTFL0_Lambda0_decayLength[0:,0]

        Bu_DTFL0_Lambda0_decayLengthErr = arrs['Bu_DTFL0_Lambda0_decayLengthErr']
        Bu_DTFL0_Lambda0_decayLengthErr = Bu_DTFL0_Lambda0_decayLengthErr[0:,0]

        Bu_L0Global_TIS                        = arrs['Bu_L0Global_TIS']
        Bu_L0HadronDecision_TOS                = arrs['Bu_L0HadronDecision_TOS']
        Bu_Hlt1Global_TIS                      = arrs['Bu_Hlt1Global_TIS']
        Bu_Hlt1TrackMVADecision_TOS            = arrs['Bu_Hlt1TrackMVADecision_TOS']
        Bu_Hlt1TwoTrackMVADecision_TOS         = arrs['Bu_Hlt1TwoTrackMVADecision_TOS']
        Bu_Hlt2Topo2BodyDecision_TOS           = arrs['Bu_Hlt2Topo2BodyDecision_TOS']
        Bu_Hlt2Topo3BodyDecision_TOS           = arrs['Bu_Hlt2Topo3BodyDecision_TOS']
        Bu_Hlt2Topo4BodyDecision_TOS           = arrs['Bu_Hlt2Topo4BodyDecision_TOS']
        Bu_Hlt2Topo2BodyBBDTDecision_TOS       = arrs['Bu_Hlt2Topo2BodyBBDTDecision_TOS']
        Bu_Hlt2Topo3BodyBBDTDecision_TOS       = arrs['Bu_Hlt2Topo3BodyBBDTDecision_TOS']
        Bu_Hlt2Topo4BodyBBDTDecision_TOS       = arrs['Bu_Hlt2Topo4BodyBBDTDecision_TOS']
        nCandidate                             = arrs['nCandidate']
        totCandidates                          = arrs['totCandidates']
        EventInSequence                        = arrs['EventInSequence']
        runNumber                              = arrs['runNumber']
        eventNumber                            = arrs['eventNumber']
        GpsTime                                = arrs['GpsTime']
        Polarity                               = arrs['Polarity']

    # Apply any transformations or calculations here
        Delta_Z = L0_ENDVERTEX_Z - Bu_ENDVERTEX_Z
        Delta_X = L0_ENDVERTEX_X - Bu_ENDVERTEX_X
        Delta_Y = L0_ENDVERTEX_Y - Bu_ENDVERTEX_Y
        Delta_X_ERR = np.sqrt(np.square(Bu_ENDVERTEX_XERR)+np.square(L0_ENDVERTEX_XERR))
        Delta_Y_ERR = np.sqrt(np.square(Bu_ENDVERTEX_YERR)+np.square(L0_ENDVERTEX_YERR))
        Delta_Z_ERR = np.sqrt(np.square(Bu_ENDVERTEX_ZERR)+np.square(L0_ENDVERTEX_ZERR))

        delta_x = np.divide(Delta_X,Delta_X_ERR)
        delta_y = np.divide(Delta_Y,Delta_Y_ERR)
        delta_z = np.divide(Delta_Z,Delta_Z_ERR)
        L0_FD_CHISQ = np.square(delta_x) + np.square(delta_y) + np.square(delta_z)


        # Apply pre-selection
        pre_select = (arrs['Bu_MM'] > 4500)


        processed_data = {'Bu_MM': Bu_MM[pre_select],
                            'Bu_MMERR': Bu_MMERR[pre_select],
                            'Bu_ID':  Bu_ID[pre_select],
                            'Bu_P':   Bu_P[pre_select],
                            'Bu_PT':  Bu_PT[pre_select],
                            'Bu_PE':  Bu_PE[pre_select],
                            'Bu_PX':  Bu_PX[pre_select],
                            'Bu_PY':  Bu_PY[pre_select],
                            'Bu_PZ':  Bu_PZ[pre_select],
                            'h1_P':  h1_P[pre_select],
                            'h1_PT': h1_PT[pre_select],
                            'h1_PE': h1_PE[pre_select],
                            'h1_PX': h1_PX[pre_select],
                            'h1_PY': h1_PY[pre_select],
                            'h1_PZ': h1_PZ[pre_select],
                            'h1_ID': h1_ID[pre_select],
                            'h1_TRACK_Type': h1_TRACK_Type[pre_select],
                            'h1_IPCHI2_OWNPV': h1_IPCHI2_OWNPV[pre_select],
                            'h1_PIDK':  h1_PIDK[pre_select],
                            'h1_PIDp':  h1_PIDp[pre_select],
                            'h2_P':  h2_P[pre_select],
                            'h2_PT': h2_PT[pre_select],
                            'h2_PE': h2_PE[pre_select],
                            'h2_PX': h2_PX[pre_select],
                            'h2_PY': h2_PY[pre_select],
                            'h2_PZ': h2_PZ[pre_select],
                            'h2_ID': h2_ID[pre_select],
                            'h2_TRACK_Type': h2_TRACK_Type[pre_select],
                            'h2_IPCHI2_OWNPV': h2_IPCHI2_OWNPV[pre_select],
                            'h2_PIDK':  h2_PIDK[pre_select],
                            'h2_PIDp':  h2_PIDp[pre_select],
                            'p_P':  p_P[pre_select],
                            'p_PT': p_PT[pre_select],
                            'p_PE': p_PE[pre_select],
                            'p_PX': p_PX[pre_select],
                            'p_PY': p_PY[pre_select],
                            'p_PZ': p_PZ[pre_select],
                            'p_ID': p_ID[pre_select],
                            'p_TRACK_Type': p_TRACK_Type[pre_select],
                            'p_IPCHI2_OWNPV': p_IPCHI2_OWNPV[pre_select],
                            'p_PIDK':  p_PIDK[pre_select],
                            'p_PIDp':  p_PIDp[pre_select],
                            'Lp_P':  Lp_P[pre_select],
                            'Lp_PT': Lp_PT[pre_select],
                            'Lp_PE': Lp_PE[pre_select],
                            'Lp_PX': Lp_PX[pre_select],
                            'Lp_PY': Lp_PY[pre_select],
                            'Lp_PZ': Lp_PZ[pre_select],
                            'Lp_ID': Lp_ID[pre_select],
                            'Lp_TRACK_Type': Lp_TRACK_Type[pre_select],
                            'Lp_PIDK':  Lp_PIDK[pre_select],
                            'Lp_PIDp':  Lp_PIDp[pre_select],
                            'L0_P':  L0_P[pre_select],
                            'L0_PT': L0_PT[pre_select],
                            'L0_PE': L0_PE[pre_select],
                            'L0_PX': L0_PX[pre_select],
                            'L0_PY': L0_PY[pre_select],
                            'L0_PZ': L0_PZ[pre_select],
                            'L0_ID': L0_ID[pre_select],
                            'Lpi_P':  Lpi_P[pre_select],
                            'Lpi_PT': Lpi_PT[pre_select],
                            'Lpi_PE': Lpi_PE[pre_select],
                            'Lpi_PX': Lpi_PX[pre_select],
                            'Lpi_PY': Lpi_PY[pre_select],
                            'Lpi_PZ': Lpi_PZ[pre_select],
                            'Lpi_ID': Lpi_ID[pre_select],
                            'Lpi_TRACK_Type': Lpi_TRACK_Type[pre_select],
                            'p_ProbNNp':      p_ProbNNp[pre_select],
                            'Lpi_ProbNNpi':   Lpi_ProbNNpi[pre_select],
                            'Lp_ProbNNp':     Lp_ProbNNp[pre_select],
                            'h1_ProbNNk':     h1_ProbNNk[pre_select],
                            'h2_ProbNNk':     h2_ProbNNk[pre_select],
                            'p_ProbNNghost':      p_ProbNNghost[pre_select],
                            'Lpi_ProbNNghost':     Lpi_ProbNNghost[pre_select],
                            'Lp_ProbNNghost':      Lp_ProbNNghost[pre_select],
                            'h1_ProbNNghost':      h1_ProbNNghost[pre_select],
                            'h2_ProbNNghost':      h2_ProbNNghost[pre_select],
                            'Bu_FDCHI2_OWNPV':         Bu_FDCHI2_OWNPV[pre_select],
                            'Bu_IPCHI2_OWNPV':         Bu_IPCHI2_OWNPV[pre_select],
                            'Bu_ENDVERTEX_X':          Bu_ENDVERTEX_X[pre_select],
                            'Bu_ENDVERTEX_Y':          Bu_ENDVERTEX_Y[pre_select],
                            'Bu_ENDVERTEX_Z':          Bu_ENDVERTEX_Z[pre_select],
                            'Bu_ENDVERTEX_XERR':       Bu_ENDVERTEX_XERR[pre_select],
                            'Bu_ENDVERTEX_YERR':       Bu_ENDVERTEX_YERR[pre_select],
                            'Bu_ENDVERTEX_ZERR':       Bu_ENDVERTEX_ZERR[pre_select],
                            'Bu_DTF_decayLength':      Bu_DTF_decayLength[pre_select],
                            'Bu_DTF_decayLengthErr':   Bu_DTF_decayLengthErr[pre_select],
                            'Bu_DTF_ctau':             Bu_DTF_ctau[pre_select],
                            'Bu_DTF_ctauErr':          Bu_DTF_ctauErr[pre_select],
                            'Bu_DTF_status':           Bu_DTF_status[pre_select],
                            'Bu_DTF_nPV':              Bu_DTF_nPV[pre_select],
                            'Bu_DTF_chi2':             Bu_DTF_chi2[pre_select],
                            'Bu_DTF_nDOF':             Bu_DTF_nDOF[pre_select],
                            'Bu_DTFL0_M':              Bu_DTFL0_M[pre_select],
                            'Bu_DTFL0_MERR':           Bu_DTFL0_MERR[pre_select],
                            'Bu_DTFL0_ctau':           Bu_DTFL0_ctau[pre_select],
                            'Bu_DTFL0_ctauErr':        Bu_DTFL0_ctauErr[pre_select],
                            'Bu_DTFL0_chi2':           Bu_DTFL0_chi2[pre_select],
                            'Bu_DTFL0_nDOF':           Bu_DTFL0_nDOF[pre_select],
                            'Bu_DTFL0_status':         Bu_DTFL0_status[pre_select],
                            'Bu_DTFL0_Lambda0_M':      Bu_DTFL0_Lambda0_M[pre_select],
                            'Bu_DTFL0_Lambda0_ctau':   Bu_DTFL0_Lambda0_ctau[pre_select],
                            'Bu_DTFL0_Lambda0_ctauErr':        Bu_DTFL0_Lambda0_ctauErr[pre_select],
                            'Bu_DTFL0_Lambda0_decayLength':    Bu_DTFL0_Lambda0_decayLength[pre_select],
                            'Bu_DTFL0_Lambda0_decayLengthErr': Bu_DTFL0_Lambda0_decayLengthErr[pre_select],
                            'L0_ENDVERTEX_X':          L0_ENDVERTEX_X[pre_select],
                            'L0_ENDVERTEX_Y':          L0_ENDVERTEX_Y[pre_select],
                            'L0_ENDVERTEX_Z':          L0_ENDVERTEX_Z[pre_select],
                            'L0_ENDVERTEX_XERR':       L0_ENDVERTEX_XERR[pre_select],
                            'L0_ENDVERTEX_YERR':       L0_ENDVERTEX_YERR[pre_select],
                            'L0_ENDVERTEX_ZERR':       L0_ENDVERTEX_ZERR[pre_select],
                            'L0_MM':                   L0_MM[pre_select],
                            'L0_OWNPV_Z':              L0_OWNPV_Z[pre_select],
                            'L0_OWNPV_ZERR':           L0_OWNPV_ZERR[pre_select],
                            'L0_FD_OWNPV':             L0_FD_OWNPV[pre_select],
                            'L0_FDCHI2_OWNPV':         L0_FDCHI2_OWNPV[pre_select],
                            'Bu_L0Global_TIS':         Bu_L0Global_TIS[pre_select],
                            'Bu_L0HadronDecision_TOS': Bu_L0HadronDecision_TOS[pre_select],
                            'Bu_Hlt1Global_TIS':       Bu_Hlt1Global_TIS[pre_select],
                            'Bu_Hlt1TrackMVADecision_TOS':      Bu_Hlt1TrackMVADecision_TOS[pre_select],
                            'Bu_Hlt1TwoTrackMVADecision_TOS':   Bu_Hlt1TwoTrackMVADecision_TOS[pre_select],
                            'Bu_Hlt2Topo2BodyDecision_TOS':     Bu_Hlt2Topo2BodyDecision_TOS[pre_select],
                            'Bu_Hlt2Topo3BodyDecision_TOS':     Bu_Hlt2Topo3BodyDecision_TOS[pre_select],
                            'Bu_Hlt2Topo4BodyDecision_TOS':     Bu_Hlt2Topo4BodyDecision_TOS[pre_select],
                            'Bu_Hlt2Topo2BodyBBDTDecision_TOS': Bu_Hlt2Topo2BodyBBDTDecision_TOS[pre_select],
                            'Bu_Hlt2Topo3BodyBBDTDecision_TOS': Bu_Hlt2Topo3BodyBBDTDecision_TOS[pre_select],
                            'Bu_Hlt2Topo4BodyBBDTDecision_TOS': Bu_Hlt2Topo4BodyBBDTDecision_TOS[pre_select],
                            'nCandidate':                       nCandidate[pre_select],
                            'totCandidates':                    totCandidates[pre_select],
                            'EventInSequence':                  EventInSequence[pre_select],
                            'eventNumber':                      eventNumber[pre_select],
                            'runNumber':                        runNumber[pre_select],
                            'GpsTime':                          GpsTime[pre_select],
                            'Polarity':                         Polarity[pre_select],
        }
        return processed_data

def save_to_root(output_file, data):
    with uproot.recreate(output_file) as outFile:
        outFile["DecayTree"] = data
    print("Data saved to", output_file)

def main(base_dir, decay_trees, years, magnets, ll_or_dds, use_parallel):
    for decay_tree in decay_trees:
        for year in years:
            for magnet in magnets:
                for ll_dd in ll_or_dds:
                    input_files = get_file_paths_for_config(base_dir, decay_tree, year, magnet, ll_dd)
                    output_file = f"{base_dir}/{decay_tree}_{year}_{magnet}_{ll_dd}.root"
                    print(f"Processing: {decay_tree}, {year}, {magnet}, {ll_dd}")
                    process_and_save_files_for_config(input_files, f"{decay_tree}_{ll_dd}", output_file, use_parallel)
                    # After processing each config, call garbage collector
                    gc.collect()

if __name__ == "__main__":
    base_dir = "/afs/cern.ch/work/m/melashri/public/bphysics/data"
    decay_trees = ["B2L0PbarKpKp", "B2L0barPKpKm"]
    years = ["2016", "2017"]
    magnets = ["MagDown", "MagUp"]
    ll_or_dds = ["DD", "LL"]
    use_parallel = False  # Set to False to run in serial mode
    main(base_dir, decay_trees, years, magnets, ll_or_dds, use_parallel)