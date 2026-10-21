import numpy as np
import os
from astropy import units as u
from tqdm    import tqdm

class readpars:
    def __init__(self, inputfile):
         if os.path.exists(inputfile):
            with open(inputfile, 'r') as f:
               lines = f.readlines()
               for line in tqdm(lines, ncols=0, desc=f"Reading parameters from {inputfile}"):
                  if line.startswith('#'):
                      continue
                  key, value = line.split('=')
                  key = key.strip()
                  value = value.strip()
                  try:
                      value = eval(value)
                  except:
                      pass
                  setattr(self, key, value)

        ###########################################################################
#R = float(((f.readline()).split())[0])       # Resolving power of spectra
#dvres   = 299792.458/(2.35*R)                # Correspoding velocity resolution
#dwres   = 1300.0/(2.35*R)                    # Corresponding wavelength resolution
#snrlim  = float(((f.readline()).split())[0]) # Lower limit on S/N
#snrulim = float(((f.readline()).split())[0]) # Upper limit on S/N
#minlox  = float(((f.readline()).split())[0]) # Minimum LOX for atomic species
#nsigma  = float(((f.readline()).split())[0]) # Minimum absorption detection threshold
#samplefile1 = (f.readline().split())[0]      # Sample file 1
#samplefile0 = (f.readline().split())[0]      # Sample file 0
#ncomp      = int(((f.readline()).split())[0])  # Number of NMF components
#dwave      = float(((f.readline()).split())[0]) # Angstroms - sampling for NMF decomp
#sigclip    = float(((f.readline()).split())[0])   # Sigma-clippig for NMF
#compfile   = '../../nmf_components_'+str(ncomp)+'_SNR_'+str(snrlim)+'_sigma_'+str(sigclip)
#box1       = int(((f.readline()).split())[0])
#box2       = int(((f.readline()).split())[0])
#nsmth      = int(((f.readline()).split())[0]) # Smoothing factor x (2803.5315 - 2796.3543)/self.dwave; Smoothing factor for determining initial Legendre order
#donmf      = True
#if (((f.readline()).split())[0] == 'False'): donmf     = False
#dofit      = True
#if (((f.readline()).split())[0] == 'False'): dofit     = False
#absonly    = True
#if (((f.readline()).split())[0] == 'False'): absonly   = False
#dosearch   = True
#if (((f.readline()).split())[0] == 'False'): dosearch  = False
#specid   = True
#if (((f.readline()).split())[0] == 'False'): specid  = False
#firsttpcf  = True
#if (((f.readline()).split())[0] == 'False'): firsttpcf = False
#docluster  = True
#if (((f.readline()).split())[0] == 'False'): docluster = False
        ###########################################################################
        # Data directory
         print("Defining program parameters")
         try:
            print(f"Base data path = {self.basedir}")
         except AttributeError:
            print(f"Base data path not defined in {inputfile}")
            self.basedir = "/mnt/data/QANVAS_data"

        ###########################################################################
        # Program flow
         try:
            print(f"Verbose = {self.verbose}")
         except AttributeError:
            print(f"Verbose not defined in {inputfile}")
            self.verbose = True

         try:
            print(f"ned_photometry = {self.ned_photometry}")
         except AttributeError:
            print(f"ned_photometry not defined in {inputfile}")
            self.ned_photometry = True

         try:
            print(f"sdss_composite = {self.sdss_composite}")
         except AttributeError:
            print(f"sdss_composite not defined in {inputfile}")
            self.sdss_composite = True

         try:
            print(f"plot_freq = {self.splot_freq}")
         except AttributeError:
            print(f"plot_freq not defined in {inputfile}")
            self.plot_freq = 100

         try:
            print(f"nproc = {self.nproc}")
         except AttributeError:
            print(f"nproc not defined in {inputfile}")
            self.nproc = 1

        ###########################################################################
        # Sample parameters
         try:
            print(f"Nsamp = {self.nsamp}")
         except AttributeError:
            print(f"Nsamp not defined in {inputfile}")
            self.nsamp = -1

         try:
            print(f"snr1300min = {self.snr1300min}")
         except AttributeError:
            print(f"snr1300min not defined in {inputfile}")
            self.snr1300min = -1

         try:
            print(f"object_filter = {self.object_filter}")
         except AttributeError:
            print(f"object_filter not defined in {inputfile}")
            self.object_filter = np.array([])

         try:
            print(f"R = {self.R}")
         except AttributeError:
            print(f"R not defined in {inputfile}")
            self.R = 1300.


        ###########################################################################
        # Clustering parameters
         try:
            print(f"clustering = {self.clustering}")
         except AttributeError:
            print(f"clustering not defined in {inputfile}")
            self.clustering = False

         try:
            print(f"cluster_nsigma = {self.cluster_nsigma}")
         except AttributeError:
            print(f"cluster_nsigma not defined in {inputfile}")
            self.cluster_nsigma = 3.0

         try:
            print(f"cluster_tol = {self.cluster_tol}")
         except AttributeError:
            print(f"cluster_tol not defined in {inputfile}")
            self.cluster_tol = 1.0e-5

        ###########################################################################
        # NMF parameters
         try:
            print(f"nmf = {self.nmf}")
         except AttributeError:
            print(f"nmf not defined in {inputfile}")
            self.nmf = False

         try:
            print(f"nmf_ncomp = {self.nmf_ncomp}")
         except AttributeError:
            print(f"nmf_ncomp not defined in {inputfile}")
            self.nmf_ncomp = 10

         try:
            print(f"nmf_smooth_fac = {self.nmf_smooth_fac}")
         except AttributeError:
            print(f"nmf_smooth_fac not defined in {inputfile}")
            self.nmf_smooth_fac = np.array([24,12])

         try:
            print(f"nmf_smooth_sigma = {self.nmf_smooth_sigma}")
         except AttributeError:
            print(f"nmf_smooth_sigma not defined in {inputfile}")
            self.nmf_smooth_sigma = 1.0

        ###########################################################################
        # Absorption Catalog parameters
         try:
            print(f"abscat = {self.abscat}")
         except AttributeError:
            print(f"abscat not defined in {inputfile}")
            self.abscat = False

         try:
            print(f"lohiip = {self.lohiip}")
         except AttributeError:
            print(f"lohiip not defined in {inputfile}")
            self.lohiip = np.array([20.0])

         try:
            self.abs_vlo *= (u.km/u.s)
            print(f"abs_vlo = {self.abs_vlo}")
         except AttributeError:
            print(f"abs_vlo not defined in {inputfile}")
            self.abs_vlo = -25000 * (u.km/u.s)

         try:
            self.abs_vhi *= (u.km/u.s)
            print(f"abs_vhi = {self.abs_vhi}")
         except AttributeError:
            print(f"abs_vhi not defined in {inputfile}")
            self.abs_vhi = 5000 * (u.km/u.s)
