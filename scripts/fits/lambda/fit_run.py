import uproot
import awkward as ak
import numpy as np
from scipy.stats import norm, expon
import iminuit
import matplotlib.pyplot as plt
import os

base_path = '/share/lazy/Bu2LambdaPHH/NTuples/RD/'

file_names = ['L0phh_18MU.root', 'L0phh_18MD.root', 'L0phh_17MU.root', 'L0phh_17MD.root', 
              'L0phh_16MU.root', 'L0phh_16MD.root', 'L0phh_15MU.root', 'L0phh_15MD.root']

def process_file(filename):
    full_path = os.path.join(base_path, filename) # Add the base_path here.
    f = uproot.open(full_path)
    print(*f.keys(), sep='\n')
    print(" ---------")
    print("       ")

    t = f['B2L0pbarKpKp/DecayTree']

    arrs = t.arrays(library="ak")

    print("Arrays are ready:")
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

    Bu_ENDVERTEX_X = arrs['Bu_ENDVERTEX_X'] 
    Bu_ENDVERTEX_Y = arrs['Bu_ENDVERTEX_Y'] 
    Bu_ENDVERTEX_Z = arrs['Bu_ENDVERTEX_Z'] 
    Bu_ENDVERTEX_XERR = arrs['Bu_ENDVERTEX_XERR'] 
    Bu_ENDVERTEX_YERR = arrs['Bu_ENDVERTEX_YERR'] 
    Bu_ENDVERTEX_ZERR = arrs['Bu_ENDVERTEX_ZERR'] 
    Bu_IPCHI2_OWNPV = arrs['Bu_IPCHI2_OWNPV'] 
    Bu_MM = arrs['Bu_MM'] 
    Bu_DOCA12 = arrs['Bu_DOCA12'] 



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
    Lp_ProbNNk = arrs['Lp_ProbNNk']

    good_L0_MM_LL = L0_MM[(Lp_ProbNNp>0.5) & LL & (L0_MM>1100) & (L0_MM<1165) ]
    good_L0_MM_DD = L0_MM[(Lp_ProbNNp>0.5) & DD & (L0_MM>1100) & (L0_MM<1165) ]

    def pdf(m, n_s, n_b, mu, sigma, a):
        return n_s*norm.pdf(m, mu, sigma) + n_b*expon.pdf(m, loc=1100, scale=a)



    y_bp, edges = np.histogram(good_L0_MM_LL, range=(1000, 1165), bins=60)
    y_bp = np.array(y_bp)
    edges = np.array(edges)

    x = 0.5*(edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    err_bp = np.sqrt(y_bp)

    def chi2_bp(n_s, n_b, mu, sigma, a):
        mask = err_bp != 0
        ressq = (y_bp[mask] - widths[mask]*pdf(x[mask], n_s, n_b, mu, sigma, a))**2/err_bp[mask]**2
        return ressq.sum()

    init_pars = [
        y_bp.sum()*0.95, 
        y_bp.sum()*0.05,
        1115., 
        11.5, 
        400. 
    ]

    minimizer_bp = iminuit.Minuit(chi2_bp, *init_pars)
    limits = [
        (0, np.inf), 
        (0, np.inf),
        (1000, 1165), 
        (1., 200.), 
        (1., np.inf) 
    ]
        
    minimizer_bp.limits = limits


    minimizer_bp.migrad()



    fit_xs = np.linspace(1100, 1135, 80)
    fit_ys = pdf(fit_xs, *minimizer_bp.values)

    plt.clf()
    ax = plt.gca()
    ax.errorbar(x, y_bp, yerr=err_bp, fmt='.', color='black')

    ax.plot(fit_xs, widths[0]*fit_ys, color='red')
    ax.set_ylim(0)


    
    file_tag = filename.split('_')[1].split('.')[0]  

    plt.clf()
    ax = plt.gca()
    ax.errorbar(x, y_bp, yerr=err_bp, fmt='.', color='black')

    ax.plot(fit_xs, widths[0]*fit_ys, color='red')
    ax.set_ylim(0)

    ax.title.set_text(f'Lambda LL mass distribution, good_L0_MM_LL fit ({file_tag})')
    ax.yaxis.set_label_text('candidates per 2 MeV')
    ax.xaxis.set_label_text('mass [MeV]')
    ax.set_xlim(1100, 1135)
    ax.grid(True)
    plt.savefig(f'plots/B2L0pbarKpKp/lambda/Lambda_LL_mass_distribution_fit_{file_tag}.png')
    plt.show()


    y_bp, edges = np.histogram(good_L0_MM_DD, range=(1000, 1165), bins=60)
    y_bp = np.array(y_bp)
    edges = np.array(edges)

    x = 0.5*(edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    err_bp = np.sqrt(y_bp)


    init_pars = [
        y_bp.sum()*0.95, 
        y_bp.sum()*0.05,
        1115., 
        11.5, 
        400. 
    ]

    minimizer_bp = iminuit.Minuit(chi2_bp, *init_pars)
    limits = [
        (0, np.inf), 
        (0, np.inf),
        (1000, 1165), 
        (1., 200.), 
        (1., np.inf) 
    ]
        
    minimizer_bp.limits = limits


    minimizer_bp.migrad()



    fit_xs = np.linspace(1100, 1135, 80)
    fit_ys = pdf(fit_xs, *minimizer_bp.values)


    # And also for the second plot
    plt.clf()
    ax = plt.gca()
    ax.errorbar(x, y_bp, yerr=err_bp, fmt='.', color='black')

    ax.plot(fit_xs, widths[0]*fit_ys, color='red')
    ax.set_ylim(0)

    ax.title.set_text(f'Lambda DD mass distribution, good_L0_MM_DD fit ({file_tag})')
    ax.yaxis.set_label_text('candidates per 2 MeV')
    ax.xaxis.set_label_text('mass [MeV]')
    ax.set_xlim(1100, 1135)
    ax.grid(True)
    plt.savefig(f'plots/lB2L0pbarKpKp/ambda/Lambda_DD_mass_distribution_fit_{file_tag}.png')
    plt.show()

# Now, process each file with a simple loop:
for filename in file_names:
    process_file(filename)
