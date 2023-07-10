import uproot
import awkward as ak
import numpy as np
from scipy.stats import norm, expon
import iminuit
import matplotlib.pyplot as plt
import os

base_path = '/share/lazy/Bu2LambdaPHH/NTuples/RD/'
#plots_folder = '../../../plots/B2L0pbarKpKp/lambda/'
plots_folder = '/data/home/melashri/BPhysics/Mohamed/plots/B2L0pbarKpKp/lambda'

file_names = ['L0phh_18MU.root', 'L0phh_18MD.root', 'L0phh_17MU.root', 'L0phh_17MD.root', 
              'L0phh_16MU.root', 'L0phh_16MD.root', 'L0phh_15MU.root', 'L0phh_15MD.root']

# Define a function to create a subplot
def create_subplot(ax, x, y_bp, err_bp, fit_xs, fit_ys, title, x_label, y_label):
    ax.errorbar(x, y_bp, yerr=err_bp, fmt='.', color='black')
    ax.plot(fit_xs, fit_ys, color='red')
    ax.title.set_text(title)
    ax.xaxis.set_label_text(x_label)
    ax.yaxis.set_label_text('candidates per 2 MeV')
    ax.set_ylim(0)
    ax.set_xlim(1100, 1135)
    ax.grid(True)

# To store results for LL and DD for each file
results = {}
for year in ['18', '17', '16', '15']:
    results[year] = {'MU': {'LL': None, 'DD': None}, 'MD': {'LL': None, 'DD': None}}
    
        
def process_file(filename):
    full_path = os.path.join(base_path, filename) 
    f = uproot.open(full_path)
    #print(*f.keys(), sep='\n')
    print(" ---------")
    print("       ")

    t = f['B2L0pbarKpKp/DecayTree']

    arrs = t.arrays(library="ak")
    print(f"Number of events: {len(arrs)}")
    h1_P = arrs['h1_P']
    h1_PT = arrs['h1_PT']
    h1_PE = arrs['h1_PE']
    h1_PX = arrs['h1_PX']
    h1_PY = arrs['h1_PY']
    h1_PZ = arrs['h1_PZ']
    h1_ID = arrs['h1_ID']
    h1_TRACK_Type = arrs['h1_TRACK_Type']


    h2_P = arrs['h2_P']
    h2_PT = arrs['h2_PT']
    h2_PE = arrs['h2_PE']
    h2_PX = arrs['h2_PX']
    h2_PY = arrs['h2_PY']
    h2_PZ = arrs['h2_PZ']
    h2_ID = arrs['h2_ID']
    h2_TRACK_Type = arrs['h2_TRACK_Type']

    p_P = arrs['p_P']
    p_PT = arrs['p_PT']
    p_PE = arrs['p_PE']
    p_PX = arrs['p_PX']
    p_PY = arrs['p_PY']
    p_PZ = arrs['p_PZ']
    p_ID = arrs['p_ID']
    p_TRACK_Type = arrs['p_TRACK_Type']



    Lp_P = arrs['Lp_P']
    Lp_PT = arrs['Lp_PT']
    Lp_PE = arrs['Lp_PE']
    Lp_PX = arrs['Lp_PX']
    Lp_PY = arrs['Lp_PY']
    Lp_PZ = arrs['Lp_PZ']
    Lp_ID = arrs['Lp_ID']
    Lp_TRACK_Type = arrs['Lp_TRACK_Type']
    Lp_ProbNNp = arrs['Lp_ProbNNp']

    LL = (3 == Lp_TRACK_Type)
    DD = (5 == Lp_TRACK_Type)
    Lpi_P = arrs['Lpi_P']
    Lpi_PT = arrs['Lpi_PT']
    Lpi_PE = arrs['Lpi_PE']
    Lpi_PX = arrs['Lpi_PX']
    Lpi_PY = arrs['Lpi_PY']
    Lpi_PZ = arrs['Lpi_PZ']
    Lpi_ID = arrs['Lpi_ID']
    Lpi_TRACK_Type = arrs['Lpi_TRACK_Type']
    Lpi_ProbNNpi = arrs['Lpi_ProbNNpi']



    L0_P = arrs['L0_P']
    L0_PT = arrs['L0_PT']
    L0_PE = arrs['L0_PE']
    L0_PX = arrs['L0_PX']
    L0_PY = arrs['L0_PY']
    L0_PZ = arrs['L0_PZ']
    L0_ID = arrs['L0_ID']
    L0_MM = arrs['L0_MM']
    L0_DOCA12 = arrs['L0_DOCA12']


    L0_ENDVERTEX_X = arrs['L0_ENDVERTEX_X'] 
    L0_ENDVERTEX_Y = arrs['L0_ENDVERTEX_Y'] 
    L0_ENDVERTEX_Z = arrs['L0_ENDVERTEX_Z'] 
    L0_ENDVERTEX_XERR = arrs['L0_ENDVERTEX_XERR'] 
    L0_ENDVERTEX_YERR = arrs['L0_ENDVERTEX_YERR'] 
    L0_ENDVERTEX_ZERR = arrs['L0_ENDVERTEX_ZERR'] 
    L0_OWNPV_Z = arrs['L0_OWNPV_Z'] 
    L0_OWNPV_ZERR = arrs['L0_OWNPV_ZERR'] 

    L0_FD_OWNPV = arrs['L0_FD_OWNPV'] 
    L0_FDCHI2_OWNPV = arrs['L0_FDCHI2_OWNPV'] 


    Lp_ProbNNk = arrs['Lp_ProbNNk']

    good_L0_MM_LL = L0_MM[(Lp_ProbNNp>0.5) & LL & (L0_MM>1100) & (L0_MM<1165) ]
    good_L0_MM_DD = L0_MM[(Lp_ProbNNp>0.5) & DD & (L0_MM>1100) & (L0_MM<1165) ]

    def pdf(m, n_s, n_b, mu, sigma, a):
        return n_s*norm.pdf(m, mu, sigma) + n_b*expon.pdf(m, loc=1100, scale=a)


    y_l0_ll, edges_ll = np.histogram(good_L0_MM_LL, range=(1000, 1165), bins=60)
    y_l0_ll = np.array(y_l0_ll)
    edge_ll = np.array(edges_ll)
    
    xs_ll = 0.5*(edges_ll[1:]+edges_ll[:-1])
    widths_ll = 0.5*(edges_ll[1:]-edges_ll[:-1]) 
    err_l0_ll = np.sqrt(y_l0_ll)
    
    def chi2_l0_ll(n_s, n_b, mu, sigma, a):
        mask = err_l0_ll != 0
        ressq = (y_l0_ll[mask] - widths_ll[mask]*pdf(xs_ll[mask], n_s, n_b, mu, sigma, a))**2/err_l0_ll[mask]**2
        #return ressq.sum()
        return ak.sum(ressq)


    init_pars = [
        y_l0_ll.sum()*0.95, 
        y_l0_ll.sum()*0.05,
        1115., 
        11.5, 
        400. 
    ]
    minimizer_ll = iminuit.Minuit(chi2_l0_ll, *init_pars)
    limits = [
        (0, np.inf), 
        (0, np.inf),
        (1000, 1165), 
        (1., 200.), 
        (1., np.inf) 
    ]
       
    minimizer_ll.limits = limits


    minimizer_ll.migrad()
    
    
    fit_ll_xs = np.linspace(1100, 1135, 80)
    fit_ll_ys = pdf(fit_ll_xs, *minimizer_ll.values)
    
    #results_ll.append((xs_ll, y_l0_ll, err_l0_ll, fit_ll_xs, fit_ll_ys, filename.split('_')[1].split('.')[0]))
    #results['LL'][filename.split('_')[1].split('.')[0]] = (xs_ll, y_l0_ll, err_l0_ll, fit_ll_xs, fit_ll_ys)
    y_l0_dd, edges_dd = np.histogram(good_L0_MM_DD, range=(1000, 1165), bins=60)
    y_l0_dd = np.array(y_l0_dd)
    edge_dd = np.array(edges_dd)
    
    xs_dd = 0.5*(edges_dd[1:]+edges_dd[:-1])
    widths_dd = 0.5*(edges_dd[1:]-edges_dd[:-1]) 
    err_l0_dd = np.sqrt(y_l0_dd)
    
    def chi2_l0_dd(n_s, n_b, mu, sigma, a):
        mask = err_l0_dd != 0
        ressq = (y_l0_dd[mask] - widths_dd[mask]*pdf(xs_dd[mask], n_s, n_b, mu, sigma, a))**2/err_l0_dd[mask]**2
        #return ressq.sum()
        return ak.sum(ressq)


    init_pars = [
        y_l0_dd.sum()*0.95, 
        y_l0_dd.sum()*0.05,
        1115., 
        11.5, 
        400. 
    ]
    minimizer_dd = iminuit.Minuit(chi2_l0_dd, *init_pars)
    limits = [
        (0, np.inf), 
        (0, np.inf),
        (1000, 1165), 
        (1., 200.), 
        (1., np.inf) 
    ]
       
    minimizer_dd.limits = limits


    minimizer_dd.migrad()
    fit_dd_xs = np.linspace(1100, 1135, 80)
    fit_dd_ys = pdf(fit_dd_xs, *minimizer_dd.values)
    #results_dd.append((xs_dd, y_l0_dd, err_l0_dd, fit_dd_xs, fit_dd_ys, filename.split('_')[1].split('.')[0]))
    #results['DD'][filename.split('_')[1].split('.')[0]] = (xs_dd, y_l0_dd, err_l0_dd, fit_dd_xs, fit_dd_ys)
    year_magnet = filename.split('_')[1].split('.')[0]  # This should give '18MU' or '17MD' etc.
    year = year_magnet[:2]  # first 2 characters: the year
    magnet = year_magnet[2:]  # rest of the characters: the magnet direction
    results[year][magnet]['LL'] = (xs_ll, y_l0_ll, err_l0_ll, fit_ll_xs, fit_ll_ys)
    results[year][magnet]['DD'] = (xs_dd, y_l0_dd, err_l0_dd, fit_dd_xs, fit_dd_ys)    
# Process all the files
for file_name in file_names:
    process_file(file_name)
    
# Now create the plots
years = ['18', '17', '16', '15']  # add or remove years based on the actual files available

for year, mag_dirs in results.items():
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    for i, (mag_dir, particles) in enumerate(mag_dirs.items()):
        for j, (particle, result) in enumerate(particles.items()):
            create_subplot(
                ax[i, j], 
                *result,  
                f'Lambda {particle} mass distribution, good_L0_MM_{particle} fit ({year} {mag_dir})',
                'mass [MeV]', 
                'Entries'
            )
    plt.tight_layout()
    plt.savefig(f'plots/B2L0pbarKpKp/lambda/Lambda_mass_distribution_{year}.pdf')
