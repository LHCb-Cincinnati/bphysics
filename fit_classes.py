import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad


## -------------------------- Fit Classes -------------------------- ##


class Gaussian_plus_Exp:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}
        
    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        print('xMin, xMax = ',xMin,xMax)
        bin_width = bins[1]-bins[0]
        print('bin_width = ',bin_width)
        #x_vals = bins[0:len(bins)-1] + 0.5*bin_width
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        print("x_vals[0:10] = ",x_vals[0:10])
        print("x_vals shape = ",x_vals.shape)
        y_vals = nC
        print("y_vals[0:10] = ",y_vals[0:10])
        print("y_vals shape = ",y_vals.shape)
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs        
    
    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x-mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    def expA(self, x, A, b):
        integral = (A/b) * (1.0 - np.exp(-b*(self.xMax - self.xMin)))
        norm = 1./integral
        return self.bin_width * norm * A * np.exp(-b * (x - self.xMin))
    

    def Gaussian_plus_ExpA(self, x_vals, n_s, n_b, mu, sigma, A, b):
        return n_s * self.gaussianA(x_vals, mu, sigma) + n_b * self.expA(x_vals, A, b)


    def chi_squared(self, n_s, n_b, mu, sigma, A, b):
        mask = (0 != self.y_errs)
        prediction = self.Gaussian_plus_ExpA(self.x_vals[mask],n_s, n_b, mu, sigma, A, b)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()
    
    

    def fit(self, init_pars, minuit_limits):
        m = Minuit(self.chi_squared, n_s=init_pars[0], n_b=init_pars[1], mu=init_pars[2], sigma=init_pars[3], A=init_pars[4], b=init_pars[5])
        
        m.limits["b"] = minuit_limits["b"]
        m.limits["A"] = minuit_limits["A"]
        m.limits["n_s"] = minuit_limits["n_s"]
        m.limits["n_b"] = minuit_limits["n_b"]
        
        m.migrad()
        return m
    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.Gaussian_plus_ExpA(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax   
   
class DoubleGaussian_plus_Exp:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}
                
    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        #print('xMin, xMax = ',xMin,xMax)
        bin_width = bins[1]-bins[0]
        #print('bin_width = ',bin_width)
        #x_vals = bins[0:len(bins)-1] + 0.5*bin_width
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        #print("x_vals[0:10] = ",x_vals[0:10])
        #print("x_vals shape = ",x_vals.shape)
        y_vals = nC
        #print("y_vals[0:10] = ",y_vals[0:10])
        #print("y_vals shape = ",y_vals.shape)
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs        

    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x-mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    def expA(self, x, A, b):
        integral = (A/b) * (1.0 - np.exp(-b*(self.xMax - self.xMin)))
        norm = 1./integral
        return self.bin_width * norm * A * np.exp(-b * (x - self.xMin))


    def DoubleGaussian_plus_ExpA(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, A, b):
        n_s1 = n_s*f
        n_s2 = n_s*(1-f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.expA(x_vals, A, b)


    def chi_squared(self, n_s, f, n_b, mu1, mu2, sigma1, sigma2, A, b):
        mask = (0 != self.y_errs)
        #print("mask shape = ",mask.shape)
        prediction = self.DoubleGaussian_plus_ExpA(self.x_vals[mask],n_s, f, n_b, mu1, mu2, sigma1, sigma2, A, b)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()

    def fit(self, init_pars):
        m = Minuit(self.chi_squared, n_s=init_pars[0], f=init_pars[1], n_b=init_pars[2], mu1=init_pars[3], mu2=init_pars[4], sigma1=init_pars[5], sigma2=init_pars[6], A=init_pars[7], b=init_pars[8])
        
        m.limits["n_s"] = self.minuit_limits.get("n_s", None)
        m.limits["f"] = self.minuit_limits.get("f", None)
        m.limits["n_b"] = self.minuit_limits.get("n_b", None)
        m.limits["mu1"] = self.minuit_limits.get("mu1", None)
        m.limits["mu2"] = self.minuit_limits.get("mu2", None)
        m.limits["sigma1"] = self.minuit_limits.get("sigma1", None)
        m.limits["sigma2"] = self.minuit_limits.get("sigma2", None)
        m.limits["A"] = self.minuit_limits.get("A", None)
        m.limits["b"] = self.minuit_limits.get("b", None)
        
        m.migrad()
        return m    
    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.DoubleGaussian_plus_ExpA(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax
    
class DoubleGaussian_plus_Parabola:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}
                        
    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        print('xMin, xMax = ',xMin,xMax)
        bin_width = bins[1]-bins[0]
        print('bin_width = ',bin_width)
        #x_vals = bins[0:len(bins)-1] + 0.5*bin_width
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        print("x_vals[0:10] = ",x_vals[0:10])
        print("x_vals shape = ",x_vals.shape)
        y_vals = nC
        print("y_vals[0:10] = ",y_vals[0:10])
        print("y_vals shape = ",y_vals.shape)
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs        

    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x-mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    def parabola(self, x, a, b, c):
        return self.bin_width * a * (x**2) + b * x + c

    def DoubleGaussian_plus_Parabola(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, a, b, c):
        n_s1 = n_s*f
        n_s2 = n_s*(1-f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.parabola(x_vals, a, b, c)

    def chi_squared(self, n_s, f, n_b, mu1, mu2, sigma1, sigma2, a, b, c):
        mask = (0 != self.y_errs)
        prediction = self.DoubleGaussian_plus_Parabola(self.x_vals[mask],n_s, f, n_b, mu1, mu2, sigma1, sigma2, a, b, c)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()
        
    def fit(self, init_pars):
        m = Minuit(self.chi_squared, n_s=init_pars[0], f=init_pars[1], n_b=init_pars[2], mu1=init_pars[3], mu2=init_pars[4], sigma1=init_pars[5], sigma2=init_pars[6], a=init_pars[7], b=init_pars[8], c=init_pars[9])
        
        m.limits["n_s"] = self.minuit_limits.get("n_s", None)
        m.limits["f"] = self.minuit_limits.get("f", None)
        m.limits["n_b"] = self.minuit_limits.get("n_b", None)
        m.limits["mu1"] = self.minuit_limits.get("mu1", None)
        m.limits["mu2"] = self.minuit_limits.get("mu2", None)
        m.limits["sigma1"] = self.minuit_limits.get("sigma1", None)
        m.limits["sigma2"] = self.minuit_limits.get("sigma2", None)
        m.limits["a"] = self.minuit_limits.get("a", None)
        m.limits["b"] = self.minuit_limits.get("b", None)
        m.limits["c"] = self.minuit_limits.get("c", None)
        
        m.migrad()
        return m
        
    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.DoubleGaussian_plus_Parabola(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax

class DoubleGaussian_plus_Linear:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}
                
    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        print('xMin, xMax = ',xMin,xMax)
        bin_width = bins[1]-bins[0]
        print('bin_width = ',bin_width)
        #x_vals = bins[0:len(bins)-1] + 0.5*bin_width
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        print("x_vals[0:10] = ",x_vals[0:10])
        print("x_vals shape = ",x_vals.shape)
        y_vals = nC
        print("y_vals[0:10] = ",y_vals[0:10])
        print("y_vals shape = ",y_vals.shape)
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs        
    
    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x-mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    def linear(self, x, m, b):
        integral = m * (self.xMax - self.xMin) + 0.5 * b * (self.xMax**2 - self.xMin**2)
        norm = 1. / integral
        return self.bin_width * norm * (m + b * (x - self.xMin))
    def DoubleGaussian_plus_Linear(self, x_vals, n_s, f, n_b, mu1, mu2, sigma1, sigma2, m, b):
        n_s1 = n_s*f
        n_s2 = n_s*(1-f)
        return n_s1 * self.gaussian(x_vals, mu1, sigma1) + n_s2 * self.gaussian(x_vals, mu2, sigma2) + n_b * self.linear(x_vals, m, b)

    def chi_squared(self, n_s, f, n_b, mu1, mu2, sigma1, sigma2, m, b):
        mask = (0 != self.y_errs)
        prediction = self.DoubleGaussian_plus_Linear(self.x_vals[mask],n_s, f, n_b, mu1, mu2, sigma1, sigma2, m, b)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()
    
    def fit(self, init_pars):
        m = Minuit(self.chi_squared, n_s=init_pars[0], f=init_pars[1], n_b=init_pars[2], mu1=init_pars[3], mu2=init_pars[4], sigma1=init_pars[5], sigma2=init_pars[6], m=init_pars[7], b=init_pars[8])
        
        m.limits["n_s"] = self.minuit_limits.get("n_s", None)
        m.limits["f"] = self.minuit_limits.get("f", None)
        m.limits["n_b"] = self.minuit_limits.get("n_b", None)
        m.limits["mu1"] = self.minuit_limits.get("mu1", None)
        m.limits["mu2"] = self.minuit_limits.get("mu2", None)
        m.limits["sigma1"] = self.minuit_limits.get("sigma1", None)
        m.limits["sigma2"] = self.minuit_limits.get("sigma2", None)
        m.limits["m"] = self.minuit_limits.get("m", None)
        m.limits["b"] = self.minuit_limits.get("b", None)
        
        m.migrad()
        return m
    
    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.DoubleGaussian_plus_Linear(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax        
        

class Gaussian_plus_Argus:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}

    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        print('xMin, xMax = ',xMin,xMax)
        bin_width = bins[1]-bins[0]
        print('bin_width = ',bin_width)
        #x_vals = bins[0:len(bins)-1] + 0.5*bin_width
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        print("x_vals[0:10] = ",x_vals[0:10])
        print("x_vals shape = ",x_vals.shape)
        y_vals = nC
        print("y_vals[0:10] = ",y_vals[0:10])
        print("y_vals shape = ",y_vals.shape)
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs        

    def gaussian(self, x, mu, sigma):
        return self.bin_width * np.exp(-(x-mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

    def argus_bg_integral(self, m0, c, p):
        def integrand(x):
            z = 1 - (x / m0)**2
            return x * np.sqrt(z) * np.exp(c * z**p) if z > 0 else 0

        integral, _ = quad(integrand, 0, m0, limit=10000)
        return integral

    def argus_bg(self, x, m0, c, p):
        normalization = self.argus_bg_integral(m0, c, p)
        z = 1 - (x / m0)**2
        return np.where(z > 0, (x * np.sqrt(z) * np.exp(c * z**p)) / normalization, 0)
        
    def Gaussian_plus_Argus(self, x_vals, n_s, n_b, mu, sigma, m0, c, p):
        return n_s * self.gaussian(x_vals, mu, sigma) + n_b * self.argus_bg(x_vals, m0, c, p)

    def chi_squared(self, n_s, n_b, mu, sigma, m0, c, p):
        mask = (0 != self.y_errs)
        prediction = self.Gaussian_plus_Argus(self.x_vals[mask], n_s, n_b, mu, sigma, m0, c, p)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()

    def fit(self, init_pars, minuit_limits):
        m = Minuit(self.chi_squared, n_s=init_pars[0], n_b=init_pars[1], mu=init_pars[2], sigma=init_pars[3], m0=init_pars[4], c=init_pars[5], p=init_pars[6])

        for key, value in minuit_limits.items():
            m.limits[key] = value

        m.migrad()
        return m
    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.Gaussian_plus_Argus(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()
        
        if show_plot:
            plt.show()
        return fig, ax   
    
class BreitWigner_plus_Exp:
    def __init__(self, bins, nC, minuit_limits=None):
        self.xMin, self.xMax, self.bin_width, self.x_vals, self.y_vals, self.y_errs = self.Fit_Setup(nC, bins)
        self.minuit_limits = minuit_limits or {}

    def Fit_Setup(self, nC, bins):
        xMin = bins[0]
        xMax = bins[-1]
        bin_width = bins[1]-bins[0]
        x_vals = 0.5 * (bins[:-1] + bins[1:])
        y_vals = nC
        y_errs = np.sqrt(nC)
        return xMin, xMax, bin_width, x_vals, y_vals, y_errs

    def breit_wigner(self, x, M, Gamma):
        return (1 / np.pi) * (0.5 * Gamma) / ((x - M)**2 + (0.5 * Gamma)**2)

    def expA(self, x, A, b):
        integral = (A/b) * (1.0 - np.exp(-b*(self.xMax - self.xMin)))
        norm = 1./integral
        return self.bin_width * norm * A * np.exp(-b * (x - self.xMin))

    def BreitWigner_plus_ExpA(self, x_vals, n_s, n_b, M, Gamma, A, b):
        return n_s * self.breit_wigner(x_vals, M, Gamma) + n_b * self.expA(x_vals, A, b)

    def chi_squared(self, n_s, n_b, M, Gamma, A, b):
        mask = (0 != self.y_errs)
        prediction = self.BreitWigner_plus_ExpA(self.x_vals[mask], n_s, n_b, M, Gamma, A, b)
        ressq = (self.y_vals[mask] - prediction)**2 / np.square(self.y_errs[mask])
        return ressq.sum()

    def fit(self, init_pars):
        m = Minuit(self.chi_squared, n_s=init_pars[0], n_b=init_pars[1], M=init_pars[2], Gamma=init_pars[3], A=init_pars[4], b=init_pars[5])

        m.limits["n_s"] = self.minuit_limits.get("n_s", None)
        m.limits["n_b"] = self.minuit_limits.get("n_b", None)
        m.limits["M"] = self.minuit_limits.get("M", None)
        m.limits["Gamma"] = self.minuit_limits.get("Gamma", None)
        m.limits["A"] = self.minuit_limits.get("A", None)
        m.limits["b"] = self.minuit_limits.get("b", None)

        m.migrad()
        return m

    def plot(self, m, bins, nC, title='Plot', xlabel='X', ylabel='Y', vlines=None, show_plot=True):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.hist(bins[:-1], bins=bins, weights=nC, label='Data')
        mask = (0 != self.y_errs)
        predictions = self.BreitWigner_plus_ExpA(self.x_vals[mask], *m.values)
        ax.plot(self.x_vals[mask], predictions, 'r-', label='Fit')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis([self.xMin, self.xMax, 0, 1.15*max(nC)])
        ax.grid(True)
        if vlines:
            for vl in vlines:
                ax.vlines(vl, 0., 0.6*max(nC), colors='yellow')
        ax.legend()

        if show_plot:
            plt.show()
        return fig, ax



         
### ------------------------------- Helper Class ----------------------------- ##        

        
class RandomSearch:
    def __init__(self, bins, nC, fit_class, search_ranges, num_searches=1000):
        self.bins = bins
        self.nC = nC
        self.fit_class = fit_class
        self.search_ranges = search_ranges
        self.num_searches = num_searches        

    def perform_search(self):
        best_score = np.inf
        best_params = None
        best_fit = None
        
        for _ in range(self.num_searches):
            random_params = self.generate_random_params()
            fit = self.fit_class(self.bins, self.nC)  # removed random_params from class instantiation
            
            try:
                score = fit.chi_squared(**random_params)  # pass random_params to the chi_squared
                
                if score < best_score:
                    best_score = score
                    best_params = random_params
                    best_fit = fit
                    
            except ValueError:
                continue
        
        return best_score, best_params, best_fit
    
    def generate_random_params(self):
        params = {}
        
        for param, (min_val, max_val) in self.search_ranges.items():
            params[param] = np.random.uniform(min_val, max_val)
        
        return params
