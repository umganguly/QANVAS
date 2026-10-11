import os
import numpy             as np
import matplotlib.pyplot as plt
#import scipy             as sp
import sys
#import time

#from scipy                   import special
from scipy.cluster.hierarchy import dendrogram
#from scipy.interpolate       import CubicSpline
#from scipy.optimize          import leastsq
from scipy.optimize          import root
from astropy                 import constants               as const
from astropy.coordinates     import SkyCoord
from astropy                 import units                   as u
from astropy.io              import fits
#from astropy.table           import QTable,Table,Column
from astropy.table           import Table
from astropy.cosmology       import FlatLambdaCDM
from astroquery.ipac.ned     import Ned
#from sklearn.decomposition   import NMF
from sklearn.cluster         import AgglomerativeClustering as AG
from tqdm                    import tqdm

#from doppler                 import calcvel, calcwave
#from atomic                  import atomic
from spectrum                import spectrum


###############################################################################################
class sdsscat:
    ###############################################################################################
    def __init__(self, 
                 mypars,
                 ntabs = 0
                 ):
        self.flambda_unit = u.erg / u.cm**2 / u.s / u.Angstrom
        self.fefu = 1.0e-15 * self.flambda_unit
        self.sdssfu = 0.01 * self.fefu

        self.mypars = mypars

        if self.mypars.nsamp == -1:
            samplefile0 = self.mypars.basedir+"/qanvas_fullsamp.fits"
            samplefile1 = self.mypars.basedir+"/qanvas_fullsamp_nodups.fits"
            specmatfile = self.mypars.basedir+"/qanvas_specmat.fits"
        else:
            samplefile0 = self.mypars.basedir+f"/qanvas_n{self.mypars.nsamp}.fits"
            samplefile1 = self.mypars.basedir+f"/qanvas_n{self.mypars.nsamp}_nodups.fits"
            specmatfile = self.mypars.basedir+f"/qanvas_specmat_n{self.mypars.nsamp}.fits"

        print("\t"*ntabs + f"Looking for {samplefile1}")
        if not os.path.exists(samplefile1):
            print("\t"*(ntabs+1) + "Not found...")
            print("\t"*ntabs + f"Looking for {samplefile0}")
            if not os.path.exists(samplefile0):
                print("\t"*(ntabs+1) + "Not found...")
                self._fetch_sample(samplefile0,
                                   ntabs = ntabs+1
                                   )
            self._read_sample(samplefile0, 
                              ntabs = ntabs+1
                              )

            self._filter_duplicates(samplefile1,
                                    ntabs = ntabs+1
                                    )

        self._read_sample(samplefile1,
                          ntabs = ntabs+1
                          )
        print("\t"*ntabs + f"Sample has {np.unique(self.qnum).size} unique objects out of {self.qnum.size}")

        print("\t"*ntabs + f"Looking for {specmatfile}")
        if not os.path.exists(specmatfile):
            print("\t"*(ntabs+1) + "Not found...")
            speclist = self._speclist(ntabs = ntabs+1)
            rest_wavelength,fluxmat,ivarmat = self._the_matrix(speclist,
                                                               ntabs = ntabs+1
                                                               )

            The_Matrix = Table(data=[rest_wavelength,fluxmat.T,ivarmat.T],
                               names=["rest_wave","fluxmat","ivarmat"]
                               )
            The_Matrix.write(specmatfile,
                             format="fits",
                             overwrite=True
                             )

        print("\t"*(ntabs+1) + "Reading in wavelength-normalized flux matrix")
        The_Matrix = Table.read(specmatfile,
                                format="fits"
                                )
        self.rest_wavelength = The_Matrix['rest_wave'] * u.Angstrom
        self.fluxmat = The_Matrix['fluxmat'].T * self.sdssfu
        self.ivarmat = The_Matrix['ivarmat'].T * 1/self.sdssfu**2

        if self.mypars.snr1300min > 0:
            dw1300 = np.fabs(self.rest_wavelength - 1300 * u.Angstrom)
            w1300_mask = (dw1300 == np.min(dw1300))
            snr1300 = np.squeeze(self.fluxmat[:,w1300_mask] * np.sqrt(self.ivarmat[:,w1300_mask]))
            snr1300_mask = (snr1300 >= self.mypars.snr1300min)

            if np.sum(snr1300_mask) > 0:
                print("\t"*ntabs + f"Filtering spectra down to {np.sum(snr1300_mask)} with S/N_1300 > {self.mypars.snr1300min}")
                self.fluxmat = self.fluxmat[snr1300_mask,:]
                self.ivarmat = self.ivarmat[snr1300_mask,:]

        if self.mypars.ned_photometry:
            self._ned_photometry_sample(ntabs = ntabs+1)

        if self.mypars.sdss_composite or self.mypars.clustering:
            self.composite_flux = self._composite_spectrum(np.arange(self.fluxmat.shape[0],
                                                                     dtype = np.int16
                                                                     ),
                                                           ntabs = ntabs+1
                                                           )

        if self.mypars.clustering:
            self._clustering(ntabs = ntabs+1)

        input("paused")

    ###############################################################################################
    def _clustering(self,
                    p = 10,
                    ntabs = 0
                    ):

        #qmask = np.arange(self.fluxmat.shape[0], dtype=np.int16) != 73
        #self.fluxmat = self.fluxmat[qmask,:]

        new_fluxmat = np.copy(self.fluxmat)
        for i in tqdm(range(new_fluxmat.shape[1]), ncols=0, desc="\t"*ntabs + f"Substituting composite flux in fluxmat"):
            zero_mask = new_fluxmat[:,i] < sys.float_info.epsilon * self.sdssfu
            if np.sum(zero_mask) > 0:
                new_fluxmat[zero_mask,i] = self.composite_flux[i]

        model = AG(n_clusters=None,
                   distance_threshold=0
                   ).fit(new_fluxmat/self.sdssfu)

        old_distance_average = np.zeros(new_fluxmat.shape[0]-1)
        nit = 0
        num_groups = 5
        done = False
        while not done:
            nit += 1
            random_sample = self._random_sdss_sample(idx = nit,
                                                     ntabs = ntabs+1
                                                     )
            random_model = AG(n_clusters=None,
                            distance_threshold=0
                            ).fit(random_sample/self.sdssfu)
            try:
                rand_distance_array = np.vstack((rand_distance_array, 
                                                random_model.distances_
                                                ))
            except:
                rand_distance_array = np.copy(random_model.distances_)

            rand_distance_average = np.average(rand_distance_array,
                                                axis = 0
                                                )
            rand_distance_stddev  = np.std(rand_distance_array,
                                            axis = 0
                                            )

            # Better idea than this... we should do a root find on the interpolation of f(x) = model.distances_ - random_model.distances_
            breaking_point = model.distances_ > rand_distance_average + self.mypars.cluster_nsigma * rand_distance_stddev
            res = root(fun=lambda x: np.interp(x, 
                                                np.arange(model.distances_.size), 
                                                model.distances_ - (random_model.distances_ + self.mypars.cluster_nsigma * rand_distance_stddev)
                                                ),
                        x0 = np.min(model.distances_[breaking_point])
                        )
            if res.success:
                breaking_distance = np.interp(res.x, 
                                                np.arange(model.distances_.size), 
                                                model.distances_
                                                )[0]
            else:
                breaking_distance = np.min(model.distances_[breaking_point])

            num_groups = 1 + np.sum(model.distances_ > breaking_distance)

            fracerr = np.fabs(rand_distance_average-old_distance_average)/(old_distance_average+sys.float_info.epsilon)
            if np.all(fracerr < self.mypars.cluster_tol):
                done = True
            else:
                old_distance_average = rand_distance_average
                print("\t"*(ntabs+1) + f"Haven't coverged yet: {np.max(fracerr):e} {np.sum(fracerr >= self.mypars.cluster_tol)} {breaking_distance} {num_groups}")

            if nit % num_groups == num_groups-1:
                plt.clf()
                plt.subplot(211)
                plt.title(f"{nit}: {num_groups}")
                self._clustering_plot_linkage_distance(model,
                                                       rand_distance_average,
                                                       rand_distance_stddev,
                                                       breaking_distance
                                                       )

                plt.subplot(212)
                self._clustering_plot_dendrogram(model, 
                                                truncate_mode="level", 
                                                p=p
                                                )
                xlo,xhi = plt.xlim()
                plt.plot([xlo,xhi],
                        [breaking_distance,breaking_distance],
                        'k--'
                        )
                plt.xlabel("Number of quasars in node (or index of quasar if no parenthesis).")
                plt.ylabel("Linkage Distance")

                plt.show(block=False)
                plt.pause(0.01)

        plt.clf()
        plt.subplot(211)
        self._clustering_plot_dendrogram(model, 
                                        truncate_mode="level", 
                                        p=p
                                        )
        xlo,xhi = plt.xlim()
        plt.plot([xlo,xhi],
                [breaking_distance,breaking_distance],
                'k--'
                )
        plt.xlabel("Number of quasars in node (or index of quasar if no parenthesis).")
        plt.ylabel("Linkage Distance")

        plt.subplot(212)
        print("\t" * (ntabs+1) + f"Grabbing composites for {num_groups} groups with linkage distance < {breaking_distance}")
        wave_mask = ((self.rest_wavelength < 1750 * u.Angstrom) | (self.rest_wavelength > 1802 * u.Angstrom)) & \
                    ((self.rest_wavelength < 2809 * u.Angstrom) | (self.rest_wavelength > 2959 * u.Angstrom))
        qdx = self._clustering_find_group_membership(model,
                                                        breaking_distance
                                                        )

        offset = 0
        fmax = 0
        for group_number in np.unique(qdx):
            group_members = np.arange(model.distances_.shape[0]+1)[qdx == group_number]
            print("\t" * (ntabs+2) + f"Group {group_number} membership: {' '.join(map(str, group_members))}")
            if group_members.size > 1:
                group_composite = self._composite_spectrum(group_members,
                                                        ntabs = ntabs+3
                                                        )
            else:
                group_composite = np.copy(new_fluxmat[group_number,:])
            plt.step(self.rest_wavelength,
                    group_composite/self.sdssfu + offset,
                    label = f"{group_number} ({group_members.size})")
            if np.max(group_composite[wave_mask]/self.sdssfu) + offset > fmax:
                fmax = np.max(group_composite[wave_mask]/self.sdssfu) + offset
                print("\t" * (ntabs+2) + f"New fmax = {fmax}")
            offset += 2
        plt.ylim(top = fmax)
        plt.xlabel('Rest Wavelenghth [Angstrom]')
        plt.ylabel('Group Composite Flux')
        plt.legend(ncols = 1 + num_groups//6)

        plt.show(block=False)
        plt.pause(0.01)

    ###############################################################################################
    def _clustering_find_group_membership(self,
                               model,
                               breaking_distance
                               ):

        qdx = np.arange(model.distances_.shape[0]+1, 
                        dtype=np.int16
                        )

        nit = np.sum(model.distances_ <= breaking_distance)

        for i in range(nit):
            m1,m2 = model.children_[i,:]

            m1_mask = (qdx == m1)
            m2_mask = (qdx == m2)

            new_group = np.max(qdx) + 1

            qdx[m1_mask] = new_group
            qdx[m2_mask] = new_group

        return qdx

    ###############################################################################################
    def _clustering_plot_dendrogram(self,
                                    model, 
                                    **kwargs
                                    ):
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    ###############################################################################################
    def _clustering_plot_linkage_distance(self,
                                          model,
                                          rand_distance_average,
                                          rand_distance_stddev,
                                          breaking_distance
                                          ):
        plt.fill_between(np.arange(model.distances_.size),
                            rand_distance_average + rand_distance_stddev,
                            rand_distance_average - rand_distance_stddev,
                            alpha=0.5
                            )
        plt.step(np.arange(model.distances_.size),
                rand_distance_average,
                label="Random Linkage distance"
                )
        plt.step(np.arange(model.distances_.size),
                model.distances_,
                label="SDSS Linkage distance"
                )
        plt.plot([0,model.distances_.size],
                [breaking_distance,breaking_distance],
                'k--'
                )
        plt.yscale('log')
        plt.ylabel("Linkage Distance")
        plt.legend()

    ###############################################################################################
    def _composite_spectrum(self,
                            qdx,
                            plotspec = False,
                            ntabs = 0
                            ):
        composite_flux = np.zeros(self.rest_wavelength.size) * self.sdssfu
        stddev_flux    = np.zeros(self.rest_wavelength.size) * self.sdssfu
        for i in tqdm(range(self.rest_wavelength.size), ncols=0, desc="\t"*ntabs + "Computing composite spectrum"):
            non_zero_mask = self.fluxmat[qdx,i] > 0
            if np.sum(non_zero_mask) > 0:
                composite_flux[i] = 10.0**np.average(np.log10(self.fluxmat[qdx[non_zero_mask],i].to(self.sdssfu).value)) * self.sdssfu
                stddev_flux[i]    = 10.0**np.std(np.log10(self.fluxmat[qdx[non_zero_mask],i].to(self.sdssfu).value)) * self.sdssfu

        if plotspec:
            fig = plt.gcf()
            fig.clf()
            plt.fill_between(self.rest_wavelength.value,
                            (composite_flux+stddev_flux).value,
                            (composite_flux-stddev_flux).value,
                            alpha=0.2
                            )
            plt.step(self.rest_wavelength,
                    composite_flux
                    )
            plt.xlabel("Rest Wavelength [Angstrom]")
            plt.ylabel(f"Flux density [{self.flambda_unit}]")
            plt.show(block=False)

        return composite_flux

    ###############################################################################################
    def _fetch_sample(self,
                      samplefile0,
                      ntabs = 0
                      ):
        print("\t"*ntabs + "Reading in original download_rsync.txt...\n")
        data = Table.read(self.mypars.basedir+"/download_rsync.txt",
                        format="ascii.fast_no_header"
                        )
        g = np.array(data['col1'])
        self.fitsfile = np.array([], 
                                dtype='U30'
                                )
        self.fileroot = np.array([], 
                                dtype='U30'
                                )
        self.ra       = np.array([])
        self.dec      = np.array([])
        self.zqso     = np.array([])

        plt.ion()
        for line in tqdm(g, ncols=0, desc="\t"*ntabs + f"Looking for fits files in {self.mypars.basedir}"):
            fitsfile = line.rstrip()
            if os.path.exists(self.mypars.basedir+fitsfile):
                self.fitsfile = np.append(self.fitsfile, fitsfile)
                self.fileroot = np.append(self.fileroot, np.array(self.fitsfile[-1][self.fitsfile[-1].rfind("/")+1:self.fitsfile[-1].rfind(".f")], 
                                                                dtype='U30'
                                                                )
                                        )
                ra,dec,zqso = self._get_fits_info(self.fitsfile[-1])
                self.ra   = np.append(self.ra,   ra)
                self.dec  = np.append(self.dec,  dec)
                self.zqso = np.append(self.zqso, zqso)

                self.qnum = np.arange(self.ra.size,dtype=np.int32)
                if self.ra.size % 100 == 2:
                    self._plot_dist()
                    plt.pause(0.01)

            if self.ra.size >= self.mypars.nsamp and self.mypars.nsamp > 0:
                break

        self._write_sample(samplefile0)

    ###############################################################################################
    def _filter_duplicates(self,
                           samplefile1,
                           ntabs = 0
                           ):
        newq      = np.copy(self.qnum)
        sdsssupercat = SkyCoord(ra=self.ra*u.degree,dec=self.dec*u.degree)
        for q in tqdm(range(self.ra.size), ncols=0, desc="\t"*ntabs + "Looking for duplicates"):
            if newq[q] == self.qnum[q]: # Try this to speed things up.. don't need to repeat things
                s          = SkyCoord(ra=self.ra[q]*u.degree,dec=self.dec[q]*u.degree)
                ss         = s.separation(sdsssupercat).arcsecond
                ssdx       = np.extract(ss < 1.0, np.arange(self.qnum.size, dtype=np.int32))
                if ssdx.size > 1:
                    print("\t"*ntabs + "Duplicates for ",q," found in ",ssdx)
                newq[ssdx] = np.min(self.qnum[ssdx])
        self.qnum      = np.copy(newq)
        self._write_sample(samplefile1)

    ###############################################################################################
    # The shape of self.fluxmat will be (number of quasars, number of wavelength bins)
    def _generate_mock_sdss(self,
                            ntabs = 0
                            ):

        try:
            dum = self.fluxmat_sorted[0,0]
        except:
            self.fluxmat_sorted = np.copy(self.fluxmat)
            self._gms_qdxs = np.zeros((self.fluxmat.shape[1],2), dtype=np.int16)
            for i in range(self.fluxmat.shape[1]):
                self.fluxmat_sorted[:,i] = np.sort(self.fluxmat[:,i])
                non_zero_mask = (self.fluxmat_sorted[:,i] > sys.float_info.epsilon * self.sdssfu)
                if np.sum(non_zero_mask) > 0:
                    qdxs = np.arange(self.fluxmat.shape[0], dtype=np.int16)[non_zero_mask]
                    self._gms_qdxs[i,0] = np.min(qdxs)
                    self._gms_qdxs[i,1] = np.max(qdxs)
                else:
                    print("\t"*ntabs + f"No fluxes at wavelength {self.rest_wavelength[i]}")

        rng = np.random.default_rng()

        # Generate a 1D array of 5 random integers between 10 (inclusive) and 50 (exclusive)
        mock_spectrum = np.zeros_like(self.fluxmat[0,:])
        for i in range(self.fluxmat.shape[1]):
            if self._gms_qdxs[i,0] < self._gms_qdxs[i,1]:
                rdx = rng.integers(self._gms_qdxs[i,0],
                                self._gms_qdxs[i,1]
                                )
            else:
                rdx = self._gms_qdxs[i,0]
            mock_spectrum[i] = self.fluxmat_sorted[rdx,i]

        return mock_spectrum
    
    ###############################################################################################
    def _get_fits_info(self,
                       fitsfile
                       ):
        with fits.open(self.mypars.basedir+fitsfile) as hdul:
            ra   = hdul[0].header['plug_ra']
            dec  = hdul[0].header['plug_dec']
            zqso = hdul[2].data['z'][0]

        return ra,dec,zqso

    ###############################################################################################
    def _ned_photometry(self,
                        qskycoord
                        ):
        result_table = Ned.query_region(qskycoord, 
                                        radius = 5 * u.arcsec
                                        )
        redshift_mask = result_table['Redshift Points'] == np.max(result_table['Redshift Points'])
        photometry_table = Ned.get_table(result_table[redshift_mask]['Object Name'])
        broad_band_mask = (photometry_table['Frequency Mode'] == "Broad-band measurement") & \
                            (photometry_table['Flux Density'] > 0)
        #photometry_table[broad_band_mask]['Frequency', 'Flux Density', 'Upper limit of uncertainty', 'Lower limit of uncertainty', 'Upper limit of Flux Density', 'Lower limit of Flux Density', 'NED Uncertainty', 'NED Units'].pprint()

        return photometry_table[broad_band_mask]['Frequency', 'Flux Density', 'Upper limit of uncertainty', 'Lower limit of uncertainty']


    ###############################################################################################
    def _ned_photometry_sample(self,
                               ntabs = 0
                               ):

        flambda_unit = u.erg / u.cm**2 / u.s / u.Angstrom
        fefu = 1.0e-15 * flambda_unit
        sdssfu = 0.01 * fefu
        hc = (const.h * const.c).to(u.eV * u.nm).value

        qunique, quidx = np.unique(self.qnum,
                                   return_index=True
                                   )

        energy_range = np.zeros((2,qunique.size)) * u.eV

        if not os.path.exists(self.mypars.basedir+"/NED_photometry"):
            os.mkdir(self.mypars.basedir+"/NED_photometry")
        for i in tqdm(range(quidx.size), ncols=0, desc="\t"*ntabs+"Fetching NED photometry"):
            qskycoord = SkyCoord(self.ra[quidx[i]] * u.deg,
                                 self.dec[quidx[i]] * u.deg
                                 )
            nedphotfile = self.mypars.basedir+f"/NED_photometry/{self.fileroot[quidx[i]]}.fits"
            if os.path.exists(nedphotfile):
                photometry_table = Table.read(nedphotfile,
                                              format='fits'
                                              )
            else:
                photometry_table = self._ned_photometry(qskycoord)
                if 'description' in photometry_table.meta:
                    del photometry_table.meta['description']
                photometry_table.write(nedphotfile,
                                       format="fits"
                                       )

            energy_range[:,i] = np.array([(const.h * np.min(photometry_table['Frequency']) * u.Hz).to(u.eV).value,
                                          (const.h * np.max(photometry_table['Frequency']) * u.Hz).to(u.eV).value
                                          ]) * u.eV

            fig = plt.gcf()
            fig.clf()
            axs = fig.add_subplot(1,1,1)
            non_zero_mask = energy_range[0,:] > 0
            axs.step(np.sort(energy_range[0,non_zero_mask]),
                     np.arange(np.sum(non_zero_mask))
                     )
            non_zero_mask = energy_range[1,:] > 0
            axs.step(np.sort(energy_range[1,non_zero_mask]),
                     np.arange(np.sum(non_zero_mask), 0, -1)
                     )

            xlo = np.min(energy_range[0,:])
            xhi = np.max(energy_range[1,:])

            energy_radio = (const.h * (5 * u.GHz)).to(u.eV)
            energy_xray  = 2000 * u.eV

            (ylo,yhi) = axs.get_ylim()
            for e in [energy_radio,energy_xray]:
                if (e > xlo) & (e < xhi):
                    axs.plot([e.value,e.value],
                             [ylo,yhi],
                             'k--'
                             )
                    axs.text(e.value * 1.0001,
                             ylo + 0.1*(yhi-ylo),
                             f"{e:.2e}",
                             rotation=90)


            axs.set_xscale("log")
            axs.set_xlabel("Energy (eV)")
            secaxs = axs.secondary_xaxis('top',
                                         functions=(lambda e: hc / e,
                                                    lambda w: hc / w
                                                    )
                                         )
            secaxs.set_xlabel('Wavelength [nm]')
            plt.show(block=False)
            plt.pause(0.01)

            #non_zero_mask = self.fluxmat[i,:] > 0

            #if np.sum(non_zero_mask) > 0:
                #lumdist = FlatLambdaCDM(H0=70,
                #                        Om0=0.3
                #                        ).luminosity_distance(self.zqso[quidx[i]]).to(u.cm)
                #photo_restwave = photometry_table['Frequency'].to(u.Angstrom, equivalencies=u.spectral())  / (1 + self.zqso[quidx[i]])
                #photo_nuFnu = photometry_table['Flux Density'].to(u.erg/u.cm**2/u.s/u.Hz) * photometry_table['Frequency']
                #photo_Flambda = (photometry_table['Flux Density'].to(u.erg/u.cm**2/u.s/u.Hz) * photometry_table['Frequency'] * photometry_table['Frequency'] / const.c.cgs).to(flambda_unit)
                #ned_mask = (photo_restwave > np.min(self.rest_wavelength[non_zero_mask])) & \
                #           (photo_restwave < np.max(self.rest_wavelength[non_zero_mask]))
                #scales = photo_Flambda[ned_mask] / np.interp(photo_restwave[ned_mask], self.rest_wavelength[non_zero_mask], self.fluxmat[i,non_zero_mask])
                #avg_scale = np.average(scales)

                #plt.clf()
                #plt.title(f"{self.fileroot[quidx[i]]}")
                #plt.xscale("log")
                #plt.yscale("log")
                #plt.xlabel("Energy (eV)")
                #plt.ylabel("Luminosity (erg/s)")

                # Plot SDSS spectrum
                #plt.step((const.h * const.c / self.rest_wavelength[non_zero_mask]).to(u.eV),
                #         avg_scale * self.fluxmat[i,non_zero_mask] * self.rest_wavelength[non_zero_mask] * (1 + self.zqso[quidx[i]]) * 4 * np.pi * lumdist**2
                #         )

                # Plot NED photometry
                #plt.scatter((const.h * const.c / photo_restwave[ned_mask]).to(u.eV),
                #            photo_nuFnu[ned_mask] * 4 * np.pi * lumdist**2,
                #            c='r'
                #            )
                #plt.scatter((const.h * const.c / photo_restwave[~ned_mask]).to(u.eV),
                #            photo_nuFnu[~ned_mask] * 4 * np.pi * lumdist**2,
                #            c='g'
                #            )

                #xlo = np.min((const.h * const.c / photo_restwave[~ned_mask]).to(u.eV))
                #xhi = np.max((const.h * const.c / photo_restwave[~ned_mask]).to(u.eV))

                #energy_radio = (const.h * (5 * u.GHz)).to(u.eV)
                #energy_xray  = 2000 * u.eV

                #(ylo,yhi) = plt.ylim()
                #for e in [energy_radio,energy_xray]:
                #    if (e > xlo) & (e < xhi):
                #        plt.plot([e.value,e.value],
                #                 [ylo,yhi],
                #                 'k--'
                #                 )
                #        plt.text(e.value * 1.0001,
                #                 ylo + 0.1*(yhi-ylo),
                #                 f"{e:.2e}",
                #                 rotation=90)

                #plt.show(block=False)
                #plt.pause(0.001)


    ###############################################################################################
    def _plot_dist(self):
        fig = plt.gcf()
        fig.clf()
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        #fig.delaxes(ax1)
        ax1 = fig.add_subplot(1, 2, 1, projection='aitoff')
        ax1.scatter(self.ra, 
                    self.dec, 
                    s = self.zqso, 
                    color = 'red', 
                    marker = 'o'
                    )
        ax1.grid(True)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.step(np.sort(self.zqso),
                 np.arange(self.zqso.size)/self.zqso.size
                 )
        ax2.plot([np.min(self.zqso),np.max(self.zqso)],
                 [0,1],
                 "g--")
        plt.show(block=False)
        plt.pause(0.001)

    ###############################################################################################
    def _random_sdss_sample(self,
                            idx = 0,
                            ntabs = 0
                            ):

        random_sample = np.zeros_like(self.fluxmat)

        #fig = plt.gcf()
        for i in tqdm(range(self.fluxmat.shape[0]), ncols=0, desc="\t"*ntabs+f"Generating mock sample {idx}"):
            random_sample[i,:] = self._generate_mock_sdss()
            #fig.clf()
            #axs = fig.add_subplot(1,1,1)
            #axs.step(self.rest_wavelength,
            #         random_sample[i,:]
            #         )
            #plt.show(block=False)
            #plt.pause(0.1)

        return random_sample

    ###############################################################################################
    def _read_sample(self,
                      samplefile,
                      ntabs = 0
                      ):
        print("\t"*ntabs + f"Reading in {samplefile}")
        sample = Table.read(samplefile,
                            format="fits"
                            )
        self.qnum     = sample['qnum']
        self.ra       = sample['RA']
        self.dec      = sample['Dec']
        self.zqso     = sample['zqso']
        self.fileroot = sample['fileroot']
        self.fitsfile = sample['fitsfile']

    ###############################################################################################
    def _speclist(self,
                  ntabs = 0
                  ):
        speclist = []
        for qnum in tqdm(np.unique(self.qnum), ncols=0, desc="\t"*ntabs + "Reading in quasar fits spectra"):
            qnum_mask = self.qnum == qnum
            fitsfiles = []
            for f in self.fitsfile[qnum_mask]:
                fitsfiles.append(f)

            speclist.append(spectrum(self.mypars,
                                        fitsfiles,
                                        qnum,
                                        self.zqso[qnum_mask][0],
                                        ntabs = ntabs+1
                                        )
                            )
        return speclist

    ###############################################################################################
    # The shape of self.fluxmat will be (number of quasars, number of wavelength bins)
    def _the_matrix(self,
                    speclist,
                    ntabs = 0
                    ):
        fig = plt.gcf()
        plt.ion()
        fig.clf()
        for ax in fig.get_axes():
            ax.remove()
        ax = fig.add_axes((0.125, 0.1, 0.775, 0.8))
        ax.set_axis_on()
        plt.ion()

        rest_wavelength = np.array([]) * u.Angstrom
        for spec in tqdm(speclist, ncols=0, desc="\t"*ntabs+"Entering The Matrix"):
            if rest_wavelength.size == 0:
                rest_wavelength = np.copy(spec.rest_wavelength)
                nqso = np.ones(rest_wavelength.size)
                fluxmat = np.copy(spec.flux)
                ivarmat = np.copy(spec.ivar)
            else:
                fluxmat = np.vstack((fluxmat, 
                                        np.zeros(rest_wavelength.size))
                                    )
                ivarmat = np.vstack((ivarmat, 
                                        np.zeros(rest_wavelength.size))
                                    )
                between_mask = (rest_wavelength > np.min(spec.rest_wavelength)) & \
                                (rest_wavelength < np.max(spec.rest_wavelength))
                nqso[between_mask] += 1
                fluxmat[-1,between_mask] = np.interp(rest_wavelength[between_mask],
                                                        spec.rest_wavelength,
                                                        spec.flux
                                                        )
                ivarmat[-1,between_mask] = np.interp(rest_wavelength[between_mask],
                                                        spec.rest_wavelength,
                                                        spec.ivar
                                                        )

                prepend_mask = spec.rest_wavelength < np.min(rest_wavelength)
                append_mask  = spec.rest_wavelength > np.max(rest_wavelength)

                if np.sum(prepend_mask):
                    rest_wavelength = np.append(spec.rest_wavelength[prepend_mask], rest_wavelength)
                    nqso = np.append(np.ones(np.sum(prepend_mask)), nqso)
                    fluxmat = np.append(np.zeros((fluxmat.shape[0],
                                                    np.sum(prepend_mask)
                                                    )),
                                        fluxmat,
                                        axis=1
                                        )
                    ivarmat = np.append(np.zeros((ivarmat.shape[0],
                                                    np.sum(prepend_mask)
                                                    )),
                                        ivarmat,
                                        axis=1
                                        )

                    fluxmat[-1,:np.sum(prepend_mask)] = np.copy(spec.flux[prepend_mask])
                    ivarmat[-1,:np.sum(prepend_mask)] = np.copy(spec.ivar[prepend_mask])
                if np.sum(append_mask):
                    rest_wavelength = np.append(rest_wavelength, spec.rest_wavelength[append_mask])
                    nqso = np.append(nqso, np.ones(np.sum(append_mask)))
                    fluxmat = np.append(fluxmat,
                                        np.zeros((fluxmat.shape[0],
                                                  np.sum(append_mask),
                                                  )),
                                        axis=1
                                        )
                    ivarmat = np.append(ivarmat,
                                        np.zeros((ivarmat.shape[0],
                                                  np.sum(append_mask),
                                                  )),
                                        axis=1
                                        )
                    fluxmat[-1,-np.sum(append_mask):] = np.copy(spec.flux[append_mask])
                    ivarmat[-1,-np.sum(append_mask):] = np.copy(spec.ivar[append_mask])

                ax.cla()
                ax.step(rest_wavelength,
                        nqso)
                ax.step(rest_wavelength,
                        fluxmat[-1,:] * np.max(nqso) / np.max(fluxmat[-1,:]))
                plt.pause(0.001)

        return rest_wavelength,fluxmat,ivarmat

    ###############################################################################################
    def _write_sample(self,
                      samplefile
                      ):
        sample = Table(data=[self.qnum,self.ra,self.dec,self.zqso,self.fileroot,self.fitsfile],
                    names=["qnum", "RA", "Dec", "zqso", "fileroot", "fitsfile"]
                    )
        sample.write(samplefile,
                    format="fits",
                    overwrite=True
                    )

"""
            self.qnum     = np.array([], dtype=np.int32)
            self.snr      = np.array([])
            if readfits:
                print("Reading in original download_rsync.txt...\n")
                data = ascii.read(basedir+"/download_rsync.txt",format="no_header")
                with open(samplefile, "w") as f:
                    for line in g:
                        self.fitsfile = np.append(self.fitsfile, np.array(self.basedir+line.rstrip(), dtype='U30'))
                        self.fileroot = np.append(self.fileroot, np.array(self.fitsfile[-1][self.fitsfile[-1].rfind("/")+1:self.fitsfile[-1].rfind(".f")], dtype='U30'))
                        with fits.open(self.fitsfile[-1]) as hdul:
                            npix = hdul[1].header['naxis2']
                            wavelength = 10.0**hdul[1].data['loglam'] / (1.0+hdul[2].data['z'][0])
                            self.ra   = np.append(self.ra,   hdul[0].header['plug_ra'])
                            self.dec  = np.append(self.dec,  hdul[0].header['plug_dec'])
                            self.zqso = np.append(self.zqso, hdul[2].data['z'][0])
                            wdx = np.extract( (wavelength > calcwave(np.array([-5000]),np.array([1238.821]))[0] ) & (wavelength < calcwave(np.array([5000]),np.array([1548.195]))[0] ), range(wavelength.size) )
                            flux = hdul[1].data['flux']
                            ivar = hdul[1].data['ivar']
                            self.snr  = np.append(self.snr, np.sum( np.multiply( flux[wdx], np.sqrt(ivar[wdx]) ) ) )
                            self.qnum = np.arange(self.ra.size,dtype=np.int32)
                        f.write(("{:5d} {:10.5f} {:10.6f} {:8.5f} {:20s} {:s}").format(self.qnum[-1], self.ra[-1], self.dec[-1], self.zqso[-1], self.fileroot[-1], line))
                        if self.ra.size % 100 == 0:
                            print(("{:5d} {:10.5f} {:10.6f} {:8.5f} {:20s} ({}/{})").format(self.qnum[-1], self.ra[-1], self.dec[-1], self.zqso[-1], self.fileroot[-1], self.ra.size, len(g)))
                        if (self.ra.size >= maxq):
                            break
            else:
                print("Reading in", samplefile)
                data = ascii.read(basedir+"/"+samplefile, format='no_header')
                self.qnum     = np.array(data['col1'],dtype=np.int32)
                self.ra       = np.array(data['col2'])
                self.dec      = np.array(data['col3'])
                self.zqso     = np.array(data['col4'])
                self.fileroot = np.array(data['col5'], dtype='U30')
                self.fitsfile = data['col6']

        else: # This is only after self.mkasciispec has been run once
            print("Reading in catalog from ", self.basedir+'/'+samplefile)
            data = ascii.read(self.basedir+'/'+samplefile, format='no_header')
            self.qnum     = np.array(data['col1'],dtype=np.int32)
            self.fileroot = np.array(data['col2'], dtype='U30')
            self.ra       = np.array(data['col3'])
            self.dec      = np.array(data['col4'])
            self.zqso     = np.array(data['col5'])
            self.snr      = np.array(data['col6'])

        # Find potential duplicates
        if finddups:
            print("Looking for duplicates")
            self.qnum = np.arange(self.ra.size,dtype=np.int32)
            newq      = np.copy(self.qnum)
            sdsssupercat = SkyCoord(ra=self.ra*u.degree,dec=self.dec*u.degree)
            for q in tqdm(range(self.ra.size)):
                if newq[q] == self.qnum[q]: # Try this to speed things up.. don't need to repeat things
                    s          = SkyCoord(ra=self.ra[q]*u.degree,dec=self.dec[q]*u.degree)
                    ss         = s.separation(sdsssupercat).arcsecond
                    ssdx       = np.extract(ss < 1.0, np.arange(self.qnum.size, dtype=np.int32))
#                    if ssdx.size > 1:
#                        print("Duplicates for ",q," found in ",ssdx)
                    newq[ssdx] = np.min(self.qnum[ssdx])
            self.qnum      = np.copy(newq)
            with open(samplefile+"2","w") as f:
                for q in range(self.ra.size):
                    f.write(("{:5d} {:10.5f} {:10.6f} {:8.5f} {:20s} {:s}\n").format(self.qnum[q], self.ra[q], self.dec[q], self.zqso[q], self.fileroot[q], self.fitsfile[q]))
            print("Sample has ",np.unique(self.qnum).size," unique objects")



    ###############################################################################################

    def mkasciispec(self, samplefile, makeplot=False): # This will write spectra to ASCII files and create a new samplefile
        print("Combining spectra and generating new sample file called ", samplefile)
        self.snr = np.zeros(self.qnum.size)
        uqnum    = np.unique(self.qnum) # Find all of the unique objects (everything else is duplicate obs)
        sp       = 1
        t0       = time.time()
        with open(samplefile, "w") as f:
            for q in tqdm(range(uqnum.size)):
                ishouldwrite = 1
                ssdx = np.extract(self.qnum == uqnum[q], np.arange(self.qnum.size,dtype=np.int32)) # all the observations of the same object
                rootname = self.fitsfile[ssdx[0]][self.fitsfile[ssdx[0]].rfind("/")+1:self.fitsfile[ssdx[0]].rfind(".f")]

                zqso = self.zqso[ssdx[0]]
                totwave,totflux,totivar = self.combinespec(zqso,ssdx,verbose=False)

                datfile = self.basedir+'/ASCII/'+rootname+".dat"
                data = QTable(data=(totwave,totflux,totivar),names=("wave","flux","ivar"))
                data.write(datfile, format="ascii.ecsv", overwrite=True)
                self.snr[ssdx[0]] = np.average(np.multiply(totflux, np.sqrt(totivar)))
                self.fileroot[ssdx[0]] = np.array(rootname, dtype='U30')
                f.write(("{:10d} {:20s} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n").format(self.qnum[ssdx[0]], self.fileroot[ssdx[0]], self.ra[ssdx[0]], self.dec[ssdx[0]], self.zqso[ssdx[0]], self.snr[ssdx[0]]))


    ###############################################################################################

    def readspec(self,dwave,snrlim,R,clobberfits=False,clobbercomp=False):
        fitsfile = self.basedir+f"/dr18_spec_dw{dwave}_SNR{snrlim}_R{R}.fits"
        if os.path.exists(fitsfile) and clobberfits == False:
            print(f"Reading in {fitsfile}")
            t = Table.read(fitsfile, format='fits')
            self.wavebins = np.array(t['wavelength'])
            self.nwave = self.wavebins.size
            self.dwave = (self.wavebins[-1]-self.wavebins[0])/self.nwave
            self.fluxmat = np.transpose(np.array(t['flux']))

            compfile = self.basedir+f"/dr18_spec_dw{dwave}_SNR{snrlim}_R{R}_composite.fits"
            if os.path.exists(compfile) and clobbercomp == False:
                print(f"Reading in {fitsfile}")
                t = Table.read(compfile, format='fits')
                self.aveflux = np.array(t['flux'])
                self.stdflux = np.array(t['std'])
            else:
                ndx  = np.int32(np.extract(self.wavebins > 1700, range(self.wavebins.size)))
                bwdx = np.int32(np.extract(self.wavebins < 1200, range(self.wavebins.size)))
                nq      = np.ones(self.fluxmat.shape)
                for i in tqdm(range(self.fluxmat.shape[0])):
                    asdx = np.extract(self.fluxmat[i,:] == self.fluxmat[i,0], range(self.wavebins.size))
                    if asdx.size > 0:
                        self.fluxmat[i,asdx] = -999
                        nq[i,asdx] = 0

                    nzdx = np.extract(self.fluxmat[i,:] > 0, range(self.wavebins.size))

                    self.fluxmat[i,nzdx] /= np.average(self.fluxmat[i,ndx])

                    if (i % 100 == 0) or (self.fluxmat.shape[0] - i < 10):
                        nq1d = np.sum(nq, axis=0)
                        avdx = np.extract(np.sum(nq, axis=0) > 0, range(self.fluxmat.shape[1]))
                        self.aveflux = np.zeros(self.wavebins.size)
                        self.stdflux = np.zeros(self.wavebins.size)
                        
                        self.aveflux[avdx] = np.average(self.fluxmat[:,avdx],
                                                        weights=nq[:,avdx], axis=0)
                        self.stdflux[avdx] = np.sqrt(np.average(np.square(self.fluxmat[:,avdx] - np.broadcast_to(self.aveflux[avdx],
                                                                                                                 (self.fluxmat.shape[0],avdx.size)
                                                                                                                 )
                                                                          ),
                                                                weights=nq[:,avdx], axis=0))
                        plt.clf()
                        plt.fill_between(self.wavebins,
                                         self.aveflux+self.stdflux,
                                         np.where(self.aveflux-self.stdflux > 0,
                                                  self.aveflux-self.stdflux,
                                                  np.zeros(self.wavebins.size)),
                                         color="#00274c", label="sigma")
                        plt.step(self.wavebins,self.aveflux, color="#ffcb05", label="average")
                        (ylo,yhi) = plt.ylim()
                        plt.step(self.wavebins,0.95*yhi*nq1d/np.max(nq1d), color="#AA0000")
                        plt.title(f"{i} / {self.fluxmat[:,0].size}")
                        plt.legend(ncols=6)
                        plt.ylim(ylo,yhi)
                        plt.pause(0.0001)

                print(f"Writing to {compfile}")
                t = Table(data=[self.wavebins,self.aveflux,self.stdflux,nq1d], names=['wavelength','flux','std','nq1d'])
                t.write(compfile, format='fits', overwrite=True)

        else:
            print("Determining number of objects and wavelength range...")
            nobj2 = 0
            wlo = 1.0e+10
            sdx = np.extract(self.snr > snrlim, range(self.snr.size))
            t0 = time.time()
            for i in tqdm(range(sdx.size)):
                rootname = self.fileroot[sdx[i]]
                datfile = self.basedir+'/ASCII/' + rootname + '.dat'
                if os.path.exists(datfile):
                    nobj2 += 1
                    data = QTable.read(datfile, format="ascii.ecsv")
                    if wlo > np.min(data['wave']):
                        wlo = np.squeeze(np.min(np.concatenate(([wlo], data['wave']))))
                        whi = np.squeeze(calcwave(-calcvel(wlo,np.array([1238.821])), np.array([1550.771])))

            whi = np.squeeze(calcwave(-calcvel(wlo,np.array([1238.821])), np.array([1550.771])))
            self.nwave = int(np.squeeze((whi-wlo)/dwave))

            print("Allocating memory for ",nobj2," objects and ",self.nwave," wavelength bins with SNR > ",snrlim)
            self.wavebins = np.squeeze(np.linspace(wlo,whi,num=self.nwave))
            self.dwave    = (whi-wlo)/self.nwave
            self.fluxmat  = np.zeros((nobj2,self.nwave))

            nobj = 0
            it = 0
            t0 = time.time()
            for i in tqdm(sdx):
                rootname = self.fileroot[i]
                datfile = self.basedir+'/ASCII/' + rootname + '.dat'
                if (os.path.exists(datfile)): # and (not(os.path.exists(linfile))):
                    myspec = spectrum(datfile,self.qnum[i],R)
                    if np.average(myspec.snr) > snrlim:
                        gdx = np.squeeze(np.extract(self.wavebins > np.min(myspec.wavelength), range(self.nwave)))
                        self.fluxmat[nobj,gdx] = np.squeeze(np.interp(np.squeeze(self.wavebins[gdx]),
                                                           np.squeeze(myspec.wavelength),
                                                           np.squeeze(np.where(myspec.flux <=0, 1/np.sqrt(myspec.ivar), myspec.flux))))
                        nobj += 1
                        if nobj/nobj2 > 0.05 * it:
                            ix = np.extract(sdx == i, range(sdx.size))[0]
#                            print(f"Read in {100*nobj/nobj2:.2f} % of objects... ({time.time()-t0:.2f} s elapsed, etd. {(time.time()-t0)*(sdx.size-ix-1)/(ix+1):.2f} s remaining)")
                            it += 1

            print('Quick stats on flux matrix -- Average = ',np.average(self.fluxmat),' Max = ',np.max(self.fluxmat))

            print(f"Writing to {fitsfile}")
            t = Table(data=[self.wavebins], names=['wavelength'])
            t.add_column(Column(name="flux",data=np.transpose(self.fluxmat)))
            t.write(fitsfile, format='fits', overwrite=clobberfits)

    ###############################################################################################
    # These routines are for the Non-negative Matric Factorization intended to continuum-normalize the spectra
    def nmf(self,ncomp,sigclip,done=30,makeplot=True):
        wdx = np.extract(self.wavebins < calcwave(np.array([-5000]),np.array([1215.6701]))[0], range(self.wavebins.size))
        fdx = np.extract(np.sum(self.fluxmat[:,wdx], axis=1) > 0, range(self.fluxmat.shape[0]))
        nobj = fdx.size

        ldx = np.extract(self.wavebins > 1700, range(self.wavebins.size)) # Select
        print(fdx.size,ldx.size,self.fluxmat.shape)
        normfluxes = np.average(self.fluxmat[fdx,ldx[0]:], axis=1)
        nmffluxmat = self.fluxmat[fdx,:] / np.transpose(np.broadcast_to(normfluxes, (self.wavebins.size,nobj)))
        nmffluxmat = np.where(nmffluxmat < 0,
                              np.broadcast_to(self.aveflux,(nobj, self.fluxmat.shape[1])),
                              nmffluxmat)

        self.objfilt = np.ones(nobj, dtype=np.int8)
        self.W = np.zeros((nobj,ncomp))
        print("Starting NMF with",nobj," objects")
        nit = 900
        for donedx in tqdm(range(done)):
            self.nmfmod = NMF(n_components = ncomp, init='random', random_state=0, max_iter=nit)
            ppdx = np.extract(self.objfilt == 1, range(nobj))
            self.W[ppdx,:] = self.nmfmod.fit_transform(nmffluxmat[ppdx,:])
            if (self.nmfmod.n_iter_ == nit):
                nit *= 2
                print("NMF: Increasing iterations to ",nit)
            self.H = self.nmfmod.components_
            if np.any(self.objfilt == 0):
                nndx = np.extract(self.objfilt == 0, range(nobj))
                self.W[nndx,:] = self.nmfmod.transform(nmffluxmat[nndx,:])
            # Consider the distribution of W along each axis - are there any outside (sigclip)-sigma?
            if makeplot:
                self.pltcomp(ncomp//2,2)
                plt.subplot(ncomp//2,2,1)
                plt.title(f"{ppdx.size} objects")
                plt.pause(1)
            
            newobjfilt = np.ones(nobj, dtype=np.int8)
            meanW = np.average(self.W[ppdx,:], axis=1)
            stdW  = np.std(    self.W[ppdx,:], axis=1)
            for j in range(ncomp):
                newobjfilt[np.extract(np.fabs(self.W[:,j] - meanW[j]) > sigclip * stdW[j], range(nobj))] = 0
            self.objfilt = np.copy(newobjfilt)


    def pltcomp(self,nrow,ncol):
        plt.clf()
        ldx = np.extract(self.wavebins < 1215, range(self.wavebins.size))
        hdx = np.extract(self.wavebins > 1215, range(self.wavebins.size))
        for j in range(self.H[:,0].size):
            y = self.H[j,:] * np.average(self.W[:,j])
            ylo = 0.95*np.min(np.extract((self.wavebins > 1300) & (y > 0), y))
            yhi = 1.05*np.max(np.extract((self.wavebins > 1300) & (y > 0), y))
            plt.subplot(nrow,ncol,j+1)
            plt.fill_between(self.wavebins,
                             self.H[j,:] * np.max(self.W[:,j]),
                             self.H[j,:] * np.min(self.W[:,j]),
                             alpha=0.5)
            plt.step(self.wavebins, self.H[j,:] * np.average(self.W[:,j]), "r", label=f"W{j} * H{j}  {np.average(self.H[j,ldx])/np.average(self.H[j,hdx]):.3f}")
            plt.ylim(ylo,yhi)
#            if ylo <= 0: plt.ylim(bottom=np.min(np.extract(self.wavebins > 1300, self.H[j,:] * np.average(self.W[:,j]))))
            plt.yscale("log")
            plt.legend()
        plt.pause(0.1)

    def calcspec(self,W):
        return np.dot(np.transpose(self.H), W)

    # 12/10/2024 - changing transformspec to input wavelength, flux, and ivar arrays instead of instance of spectrum class
    def transformspec(self,wavelength,flux,ivar):
#        flux = np.where(flux < 0.0, 1./np.sqrt(ivar), flux)
        nwave    = self.wavebins.size

        fdx = np.extract((wavelength > self.wavebins[0]) & (wavelength < self.wavebins[-1]) & (flux > 0.0), range(wavelength.size))


#        istart   = 0
#        iend     = -1
#        if (self.wavebins[istart] > wavelength[0]):
#            istart = np.min(np.extract(wavelength > self.wavebins[0],  range(wavelength.size)))
#        if (self.wavebins[iend] < wavelength[-1]):
#            iend = np.max(np.extract(wavelength < self.wavebins[-1], range(wavelength.size)))

#        intflux = np.interp(self.wavebins, wavelength[istart:iend], flux[istart:iend])
        intflux = np.interp(self.wavebins, wavelength[fdx], flux[fdx])
        if np.any(intflux < 0.0):
            print("Negative flux in interpolated spectrum:")
            ndx = np.extract(intflux < 0, range(intflux.size))
            for n in range(ndx.size):
                print(self.wavebins[ndx[n]], intflux[ndx[n]])

        return np.squeeze(self.nmfmod.transform(intflux.reshape(1,-1)))
##    def transformspec(self,qrefspec):
##        flux = np.where(qrefspec.flux < 0.0, 1./np.sqrt(qrefspec.ivar), qrefspec.flux)
##        nwave    = self.wavebins.size
##        istart   = 0
##        iend     = -1
##        if (self.wavebins[istart] > qrefspec.wavelength[0]):
##            istart = np.min(np.extract(qrefspec.wavelength > self.wavebins[0],  range(qrefspec.wavelength.size)))
##        if (self.wavebins[iend] < qrefspec.wavelength[-1]):
##            iend = np.max(np.extract(qrefspec.wavelength < self.wavebins[-1], range(qrefspec.wavelength.size)))
##
##        intflux = np.interp(self.wavebins, qrefspec.wavelength[istart:iend], flux[istart:iend])
##        if np.any(intflux < 0.0):
##            print("Negative flux in interpolated spectrum:")
##            ndx = np.extract(intflux < 0, range(intflux.size))
##            for n in range(ndx.size):
##                print(self.wavebins[ndx[n]], intflux[ndx[n]])
##
##        return np.squeeze(self.nmfmod.transform(intflux.reshape(1,-1)))

    def writecomp(self,compfile):
        with open(compfile, "w") as f:
            for w in range(self.wavebins.size):
                dumstr = ('{}').format(self.wavebins[w])
                for c in range(self.H[:,w].size):
                    dumstr += (' {}').format(self.H[c,w])
                dumstr += '\n'
                f.write(dumstr)

    def readcomp(self,compfile,ncomp):
        everything = np.fromfile(compfile, count=-1, sep=' ')
#        print("sdsscat.readcomp: everything.size = ", everything.size)
        nw = everything.size//(ncomp+1)

        everywhere = np.transpose(everything.reshape(nw,ncomp+1))
        self.wavebins = everywhere[0,:]
        self.H        = everywhere[1:,:]
        self.ncomp    = ncomp

        self.nmfmod = NMF(n_components = ncomp, init='random', random_state=0, max_iter=900)
        self.nmfmod.components_ = self.H

        uniqarray, udx = np.unique(self.qnum, return_index=True)
        self.W = np.zeros((self.qnum.size, ncomp))

    ###############################################################################################
    # clustering.children_ has the amalgamation schedule
    # clustering.distances_ has the linkage distances for each merger
    def clustering(self):
        print("Starting clustering analysis...")
        nobj = self.fluxmat[:,0].size
        plt.clf()
        self.clustering = AG(distance_threshold=0,n_clusters=None).fit(self.fluxmat)
        mdist = np.max(np.sqrt(np.sum(self.fluxmat*self.fluxmat, axis=1)))

        meanspec = np.average(self.fluxmat, axis=0)
        stdspec  = np.std(    self.fluxmat, axis=0)

        print("Generating random control sample...")
        rng           = np.random.default_rng()
        rrr           = rng.normal(size=self.fluxmat.size).reshape(self.fluxmat.shape)
        randflux      = np.broadcast_to(meanspec, self.fluxmat.shape) + rrr * np.broadcast_to(stdspec, self.fluxmat.shape)
        self.randclus = AG(distance_threshold=0,n_clusters=None).fit(randflux)
        rdist         = np.sort(randflux.flatten())
        rcdf          = np.arange(randflux.size)/randflux.size

        print("Determining membership for first significant clusters...")
        refdist = self.randclus.distances_[0]
        refdist = np.max(np.extract(self.clustering.distances_ < self.randclus.distances_, self.clustering.distances_))

        plottype = 1

        nx = 5
        ny = 3
        yp = 0
        ym = np.max(self.fluxmat)+np.max(self.clustering.distances_)
        for qq in range(nx*ny):
            refdist = self.clustering.distances_[-(qq+1)]
            cdx = np.extract(self.clustering.distances_ < refdist , np.arange(nobj, dtype=np.int16))
            self.membership = np.arange(nobj, dtype=np.int16)
            for i in range(cdx.size):
                for j in range(2):
                    c1dx = np.extract(self.membership == self.clustering.children_[cdx[i],j], range(nobj))
                    self.membership[c1dx] = nobj + i
            mdx = np.unique(self.membership)
            print(f"Found {mdx.size} unique clusters")

            if plottype == 1:
                print("Plotting...")
                comp = Table(data=[self.wavebins],names=["Wavelength"])
                compflux = np.zeros((self.wavebins.size,mdx.size))
                plt.subplot(nx,ny,qq+1)
                for i in range(mdx.size):
                    ndx = np.extract(self.membership == mdx[i], range(nobj))
                    if ndx.size > 10:
                        compflux[:,i] = np.average(self.fluxmat[ndx,:], axis=0)
                        rrr = np.average(self.fluxmat[ndx,:], axis=0)
                        plt.step(self.wavebins, rrr, label=f"{mdx[i]} {ndx.size}")
                        ym1 = np.min(np.extract(self.wavebins > 1150, rrr))
                        if ym1 < ym: ym = ym1
                plt.legend(fontsize='xx-small', ncols=2)
                plt.xlim(1150,None)
                plt.ylim(bottom = 0.95*ym)
                plt.grid(True)
                plt.pause(0.001)
                comp.add_column(Column(name="flux",data=compflux))
                comp.write(self.basedir+f"/clustering_components_{qq}.fits", format="fits", overwrite=True)
                plt.pause(0.001)

        if plottype == 0:
            plt.subplot(2,1,2)
            self.plot_dendrogram(self.clustering, truncate_mode="level", p=nobj)
            x1,x2 = plt.xlim()
            plt.plot(np.array([x1,x2]),np.array([refdist,refdist]),"c:")
            plt.ylim([0.9*self.clustering.distances_[0],1.1*self.clustering.distances_[-1]])
            plt.yscale("log")
            plt.xlabel("Object Number")
            plt.ylabel("Linkage Distance")

            plt.subplot(2,1,1)
            plt.step(np.arange(self.clustering.distances_.size), self.clustering.distances_, label="SDSS")
            plt.step(np.arange(self.randclus.distances_.size), self.randclus.distances_, label="random")
            plt.legend(fontsize='xx-small')
            plt.yscale("log")
            plt.ylabel("Linkage Distance")
            plt.xlabel("Iteration")
            plt.pause(0.001)

        t = Table([self.clustering.distances_,self.clustering.children_,self.randclus.distances_], names=["Distance","Merger","RandDist"])
        t.write(self.basedir+"/clustering.fits",format="fits",overwrite=True)

        
    def plot_dendrogram(self, model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    ###############################################################################################
    # Starting here, we have routines for cross-correlating radio data
    #
    ###############################################################################################
    # This catalog comprises the Faint Images of the Radio Sky at Twenty cm
    #(FIRST) Survey. The FIRST survey began in 1993, and covers the north
    # and south Galactic caps. The present 14Dec17 version is derived from the
    # 1993 through 2011 observations. The catalog covers a total of about 
    # 10,575 square degrees of sky (8,444 square degrees in the north Galactic
    # cap and 2,131 square degrees in the south Galactic cap)
    def readfirst(self):
        with fits.open('D:/ganguly/SDSSAAL/FIRST/first_14dec17.fits') as first:
            return first[1].header['NAXIS2'],first[1].data['RA'],first[1].data['DEC'],first[1].data['FPEAK'],first[1].data['FINT']

    ###############################################################################################
    # This catalog comprises the Very Large Array Sky Survey (VLASS)
    # The VLASS survey began in xxxx, and covers xxxx. The present xxxx version is derived from the
    # xxxx through xxxx observations. The catalog covers a total of about 
    # xxxx square degrees of sky 
    def readvlass(self):

        vlass = QTable.read('D:/ganguly/SDSSAAL/dr18/VLASS/vlass.dat', format="ascii.ecsv")
        return len(vlass),vlass['RA'],vlass['DEC'],vlass['Peak_flux'],vlass['Total_flux']
#        vlass = ascii.read('D:/ganguly/VLASS/CIRADA_VLASS1QL_table1_components_v2.csv', header_start=0, data_start=1, delimiter=",", format="csv", guess=False)
#        return np.array(vlass['Component_name']).size,np.array(vlass['RA']),np.array(vlass['DEC']),np.array(vlass['Peak_flux']),np.array(vlass['Total_flux'])

    ###############################################################################################
    # This catalog comprises the LOFAR Two-metre Sky Survey (LoTSS)
    def readlotss(self):
        with fits.open('D:/ganguly/SDSSAAL/dr18/LoTSS/LoTSS_DR2_v110_masked.srl.fits') as lotss:
            return lotss[1].header['NAXIS2'],lotss[1].data['RA'],lotss[1].data['DEC'],lotss[1].data['Peak_flux'],lotss[1].data['Total_flux']
    ###############################################################################################
    # This catalog comprises the Westerbork Northern Sky Survey (WENSS)
    def readwenss(self):
        wenss = ascii.read('D:/ganguly/SDSSAAL/WENSS/allwenss.dat', format="no_header")
        rah  = np.array(wenss['col2'])
        ram  = np.array(wenss['col3'])
        ras  = np.array(wenss['col4'])
        decd = np.array(wenss['col5'])
        decm = np.array(wenss['col6'])
        decs = np.array(wenss['col7'])

        ra = 15 * (rah + (ram + ras/60)/60)
        dec = (np.fabs(decd) + (decm + decs/60)/60) * np.sign(decd)
        
        return (np.array(wenss['col1'])).size,ra,dec,np.array(wenss['col9']),np.array(wenss['col10'])

    ###############################################################################################
    # This catalog comprises the Sydney University Molonglo Sky Survey (SUMSS)
    def readsumss(self):
        sumss = ascii.read('D:/ganguly/SDSSAAL/SUMSS/update.txt', format="no_header")
        rah  = np.array(sumss['col1'])
        ram  = np.array(sumss['col2'])
        ras  = np.array(sumss['col3'])
        decd = np.array(sumss['col4'])
        decm = np.array(sumss['col5'])
        decs = np.array(sumss['col6'])

        ra = 15 * (rah + (ram + ras/60)/60)
        dec = (np.fabs(decd) + (decm + decs/60)/60) * np.sign(decd)
        return rah.size,ra,dec,np.array(sumss['col9']),np.array(sumss['col11'])

    ###############################################################################################
    # This catalog comprises the  Rapid ASKAP Continuum Survey (RACS)
    def readracs(self):

        racs1 = ascii.read('D:/ganguly/SDSSAAL/dr18/RACS/AS110_Derived_Catalogue_racs_dr1_sources_galacticregion_v2021_08_v02_5726.csv', format='csv')
        racs2 = ascii.read('D:/ganguly/SDSSAAL/dr18/RACS/AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv', format='csv')

        return len(racs1)+len(racs2),np.append(racs1['ra'].value,racs2['ra'].value),np.append(racs1['dec'].value,racs2['dec'].value),np.append(racs1['peak_flux'].value,racs2['peak_flux'].value),np.append(racs1['total_flux_source'].value,racs2['total_flux_source'].value)

    ###############################################################################################
    # This calculates the TPCF between SDSS and all of the radio surveys (not just FIRST....)
    def firsttpcf(self,cntfile,snrlim):
        cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

        vlass = True
        sumss = False #   DERP - SUMSS does cover the SDSS footprint!
        racs  = True

        t1 = time.time()
        nfirst,firstra,firstdec,firstfpeak,firstfint = self.readfirst()
        print(("Read in {:7d} {} objects in {} seconds").format(nfirst,'FIRST', time.time()-t1))
        t1 = time.time()
        nlotss,lotssra,lotssdec,lotssfpeak,lotssfint = self.readlotss()
        print(("Read in {:7d} {} objects in {} seconds").format(nlotss,'LoTSS', time.time()-t1))
        t1 = time.time()
        nwenss,wenssra,wenssdec,wenssfpeak,wenssfint = self.readwenss()
        print(("Read in {:7d} {} objects in {} seconds").format(nwenss,'WENSS', time.time()-t1))
        if vlass:
            t1 = time.time()
            nvlass,vlassra,vlassdec,vlassfpeak,vlassfint = self.readvlass()
            print(("Read in {:7d} {} objects in {} seconds").format(nvlass,'VLASS', time.time()-t1))
        if sumss:
            t1 = time.time()
            nsumss,sumssra,sumssdec,sumssfpeak,sumssfint = self.readsumss()
            print(("Read in {:7d} {} objects in {} seconds").format(nsumss,'SUMSS', time.time()-t1))
        if racs:
            t1 = time.time()
            nracs,racsra,racsdec,racsfpeak,racsfint = self.readracs()
            print(("Read in {:7d} {} objects in {} seconds").format(nracs,'RACS', time.time()-t1))

        firstcat = SkyCoord(ra=firstra*u.degree,dec=firstdec*u.degree)
        lotsscat = SkyCoord(ra=lotssra*u.degree,dec=lotssdec*u.degree)
        wensscat = SkyCoord(ra=wenssra*u.degree,dec=wenssdec*u.degree)
        if vlass: vlasscat = SkyCoord(ra=vlassra*u.degree,dec=vlassdec*u.degree)
        if sumss: sumsscat = SkyCoord(ra=sumssra*u.degree,dec=sumssdec*u.degree)
        if racs:  racscat  = SkyCoord(ra=racsra*u.degree,dec=racsdec*u.degree)

        firstra  = np.deg2rad(firstra)
        lotssra  = np.deg2rad(lotssra)
        wenssra  = np.deg2rad(wenssra)
        if vlass: vlassra  = np.deg2rad(vlassra)
        if sumss: sumssra  = np.deg2rad(sumssra)
        if racs:  racsra   = np.deg2rad(racsra)
        firstdec = np.deg2rad(firstdec)
        lotssdec = np.deg2rad(lotssdec)
        wenssdec = np.deg2rad(wenssdec)
        if vlass: vlassdec = np.deg2rad(vlassdec)
        if sumss: sumssdec = np.deg2rad(sumssdec)
        if racs:  racsdec  = np.deg2rad(racsdec)

        rng = np.random.default_rng(12345)

        n     = 1000
        lgdlo = 1.0
        lgdhi = 6.0
        self.dbin  = np.logspace(lgdlo,lgdhi,num=n)
        dbdx  = np.min(np.extract(self.dbin > 100000, range(n)))
        dbdx2 = np.extract((self.dbin > 10000) & (self.dbin < 500000), range(n))
        print(("Setting normalization range as {} to {}").format(self.dbin[dbdx2[0]],self.dbin[dbdx2[-1]]))
        self.ddbin = np.copy(self.dbin)
        self.ddbin[1:-1] = 0.5 * (self.dbin[2:] - self.dbin[:-2])
        self.ddbin[0] = self.ddbin[1]
        self.ddbin[-1] = self.ddbin[-2]
        self.firstncts  = np.zeros(n)
        self.lotssncts  = np.zeros(n)
        self.wenssncts  = np.zeros(n)
        if vlass: self.vlassncts  = np.zeros(n)
        if sumss: self.sumssncts  = np.zeros(n)
        if racs:  self.racsncts   = np.zeros(n)
        self.firsttpcfrcts  = np.zeros(n)
        refpix = (np.extract(self.dbin > 1.0e+4, range(n)))[0]
        udum, udx = np.unique(np.extract(self.snr > snrlim, self.qnum), return_index=True)
        udum, udx = np.unique(self.qnum, return_index=True)
        print('Searching through ',udx.size,' objects')
        dcorrbin = 0
        minsumss = 90 * u.degree
        n0 = 1.08e+4
        a0 = 1.22e+5
        chisq = 27263810808
        oldchisq = 10*chisq
        for q in range(udx.size):
            da    = cosmo.angular_diameter_distance(self.zqso[udx[q]]).to(u.kpc)
            theta = np.max(np.rad2deg(np.arctan2(self.dbin,da.value)))
            c1    = SkyCoord(ra=self.ra[udx[q]]*u.degree,dec=self.dec[udx[q]]*u.degree)

            plt.clf()

            c1first = c1.separation(firstcat) #.to(u.arcsec).value
            if np.min(c1first.value) < theta:
                dfirst = np.tan((np.extract(c1first.value < theta, c1first)).to(u.rad)) * da.value
                dfe    = np.extract(dfirst < np.max(self.dbin), dfirst)
                ncts,dum = np.histogram(dfe, bins=self.dbin)
                self.firstncts[:-1] += ncts
                rp = self.dbin/a0
                drp = self.ddbin/a0
                modelfunc = 2.0 * np.pi * n0 * rp * drp / np.power(1.0 + rp*rp, 1.5)
                chisq = np.sum(np.square(self.firstncts[dbdx2] - modelfunc[dbdx2]))
                if (np.sum(self.firstncts) > 10):
                    tdx = np.extract(self.firstncts > 0, range(self.dbin.size))
                    nit = 1000
                    while nit > 0:
                        oldchisq = chisq
                        olda = a0
                        oldn = n0

                        a0 *= 0.9 + 0.2 * rng.uniform()
                        n0 *= 0.9 + 0.2 * rng.uniform()
                        if a0 > 1.0e+9:
                            print(("Resetting because a0 = {}, n0 = {}").format(a0,n0))
                            n0 = 1.08e+4
                            a0 = 1.22e+5
                            modelfunc = 2.0 * np.pi * n0 * rp * drp / np.power(1.0 + rp*rp, 1.5)
                            chisq = np.sum(np.square(self.firstncts[dbdx2] - modelfunc[dbdx2]))
                            oldchisq = 10*chisq

                        rp = self.dbin/a0
                        drp = self.ddbin/a0
                        modelfunc = 2.0 * np.pi * n0 * rp * drp / np.power(1.0 + rp*rp, 1.5)
                        chisq = np.sum(np.square(self.firstncts[dbdx2] - modelfunc[dbdx2]))

                        if chisq < oldchisq:
                            nit = 1000
                        else:
                            nit -= 1
                            a0 = olda
                            n0 = oldn
                            chisq = oldchisq
                    plt.step(self.dbin[tdx],self.firstncts[tdx],"b")
                    plt.plot(self.dbin[tdx],modelfunc[tdx],"k", label=f"model - a0 = {a0:.3g} n0={n0:.3g} X^2={chisq}")


            c1lotss = c1.separation(lotsscat) #.to(u.arcsec).value
            if np.min(c1lotss.value) < theta:
                dlotss = np.tan((np.extract(c1lotss.value < theta, c1lotss)).to(u.rad)) * da.value
                dle   = np.extract(dlotss < np.max(self.dbin), dlotss)
                ncts,dum = np.histogram(dle, bins=self.dbin)
                self.lotssncts[:-1] += ncts
            if (np.sum(self.firstncts[dbdx2]) > 10) and (np.sum(self.lotssncts[dbdx2]) > 10):
                tdx = np.extract(self.lotssncts > 0, range(self.dbin.size))
#                plt.step(self.dbin[tdx],self.lotssncts[tdx] * self.firstncts[dbdx] / self.lotssncts[dbdx],"g")
                plt.step(self.dbin[tdx],self.lotssncts[tdx] * np.sum(self.firstncts[dbdx2]) / np.sum(self.lotssncts[dbdx2]),"g")

            c1wenss = c1.separation(wensscat) #.to(u.arcsec).value
            if np.min(c1wenss.value) < theta:
                dwenss = np.tan((np.extract(c1wenss.value < theta, c1wenss)).to(u.rad)) * da.value
                dle   = np.extract(dwenss < np.max(self.dbin), dwenss)
                ncts,dum = np.histogram(dle, bins=self.dbin)
                self.wenssncts[:-1] += ncts
            if (np.sum(self.firstncts[dbdx2]) > 10) and (np.sum(self.wenssncts[dbdx2]) > 10):
                tdx = np.extract(self.wenssncts > 0, range(self.dbin.size))
#                plt.step(self.dbin[tdx],self.wenssncts[tdx] * self.firstncts[dbdx] / self.wenssncts[dbdx],"c")
                plt.step(self.dbin[tdx],self.wenssncts[tdx] * np.sum(self.firstncts[dbdx2]) / np.sum(self.wenssncts[dbdx2]),"c")

            if vlass:
                c1vlass = c1.separation(vlasscat) #.to(u.arcsec).value
                if np.min(c1vlass.value) < theta:
                    dvlass = np.tan((np.extract(c1vlass.value < theta, c1vlass)).to(u.rad)) * da.value
                    dle   = np.extract(dvlass < np.max(self.dbin), dvlass)
                    ncts,dum = np.histogram(dle, bins=self.dbin)
                    self.vlassncts[:-1] += ncts
                if (np.sum(self.firstncts[dbdx2]) > 10) and (np.sum(self.vlassncts[dbdx2]) > 10):
                    tdx = np.extract(self.vlassncts > 0, range(self.dbin.size))
    #                plt.step(self.dbin[tdx],self.vlassncts[tdx] * self.firstncts[dbdx] / self.vlassncts[dbdx],"m")
                    plt.step(self.dbin[tdx],self.vlassncts[tdx] * np.sum(self.firstncts[dbdx2]) / np.sum(self.vlassncts[dbdx2]),"m")

            if sumss:
                c1sumss = c1.separation(sumsscat) #.to(u.arcsec).value
                if np.min(c1sumss.value) < theta:
                    dsumss = np.tan((np.extract(c1sumss.value < theta, c1sumss)).to(u.rad)) * da.value
                    dle   = np.extract(dsumss < np.max(self.dbin), dsumss)
                    ncts,dum = np.histogram(dle, bins=self.dbin)
                    self.sumssncts[:-1] += ncts
                if (np.sum(self.firstncts) > 10) and (np.sum(self.sumssncts) > 10):
                    tdx = np.extract(self.sumssncts > 0, range(self.dbin.size))
                    plt.step(self.dbin[tdx],self.sumssncts[tdx] * np.sum(self.firstncts) / np.sum(self.sumssncts),"m")

            if racs:
                c1racs = c1.separation(racscat) #.to(u.arcsec).value
                if np.min(c1racs.value) < theta:
                    dracs = np.tan((np.extract(c1racs.value < theta, c1racs)).to(u.rad)) * da.value
                    dle   = np.extract(dracs < np.max(self.dbin), dracs)
                    ncts,dum = np.histogram(dle, bins=self.dbin)
                    self.racsncts[:-1] += ncts
                if (np.sum(self.firstncts) > 10) and (np.sum(self.racsncts) > 10):
                    tdx = np.extract(self.racsncts > 0, range(self.dbin.size))
                    plt.step(self.dbin[tdx],self.racsncts[tdx] * np.sum(self.firstncts) / np.sum(self.racsncts),"r")

            plt.legend()
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Projected distance (kpc)")
            titstr = ("qnum {} blue: first {} green: lotss {} cyan: wenss {}").format(self.qnum[udx[q]], np.sum(self.firstncts), np.sum(self.lotssncts), np.sum(self.wenssncts))
            if vlass:
                titstr = titstr + (" magenta: vlass {}").format(np.sum(self.vlassncts))
            if racs:
                titstr = titstr + (" red: racs {}").format(np.sum(self.racsncts))
            plt.title(titstr)
            plt.pause(0.0001)


            if q % 100 == 0:
                with open(cntfile, 'w') as f:
                    for i in range(n):
                        fstr = ("{} {} {} {}").format(self.dbin[i],self.firstncts[i],self.lotssncts[i],self.wenssncts[i])
                        if vlass:
                            fstr = fstr+(" {}").format(self.vlassncts[i])
                        if racs:
                            fstr = fstr+(" {}").format(self.racsncts[i])
                        fstr = fstr+"\n"
                        f.write(fstr)


        with open(cntfile, 'w') as f:
            for i in range(n):
                fstr = ("{} {} {} {}").format(self.dbin[i],self.firstncts[i],self.lotssncts[i],self.wenssncts[i])
                if vlass:
                    fstr = fstr+(" {}").format(self.vlassncts[i])
                if racs:
                    fstr = fstr+(" {}").format(self.racsncts[i])
                fstr = fstr+"\n"
                f.write(fstr)


    ###############################################################################################
    def matchradio(self,dcorr):
        cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

        t1 = time.time()
        nfirst,firstra,firstdec,firstfpeak,firstfint = self.readfirst()
        print(("Read in {:7d} {} objects in {} seconds").format(nfirst,'FIRST', time.time()-t1))
        t1 = time.time()
        nlotss,lotssra,lotssdec,lotssfpeak,lotssfint = self.readlotss()
        print(("Read in {:7d} {} objects in {} seconds").format(nlotss,'LoTSS', time.time()-t1))
        t1 = time.time()
        nvlass,vlassra,vlassdec,vlassfpeak,vlassfint = self.readvlass()
        print(("Read in {:7d} {} objects in {} seconds").format(nvlass,'VLASS', time.time()-t1))
        t1 = time.time()
        nracs,racsra,racsdec,racsfpeak,racsfint = self.readracs()
        print(("Read in {:7d} {} objects in {} seconds").format(nracs,'RACS', time.time()-t1))

        firstcat = SkyCoord(ra=firstra*u.degree,dec=firstdec*u.degree)
        lotsscat = SkyCoord(ra=lotssra*u.degree,dec=lotssdec*u.degree)
        vlasscat = SkyCoord(ra=vlassra*u.degree,dec=vlassdec*u.degree)
        racscat  = SkyCoord(ra=racsra*u.degree,dec=racsdec*u.degree)

        udum, udx = np.unique(self.qnum, return_index=True)
        ftotfirst = np.zeros(self.qnum.size) -99
        ftotlotss = np.zeros(self.qnum.size) -99
        ftotvlass = np.zeros(self.qnum.size) -99
        ftotracs  = np.zeros(self.qnum.size) -99
        rcorfirst = np.zeros(self.qnum.size) -99
        rcorlotss = np.zeros(self.qnum.size) -99
        rcorvlass = np.zeros(self.qnum.size) -99
        rcorracs  = np.zeros(self.qnum.size) -99
        nv = 0
        nf = 0
        nr = 0
        nl = 0
        print('Searching through ',udx.size,' objects')
        with open('radio_dr18_match.dat','w') as f:
            for q in udx:
                da    = cosmo.angular_diameter_distance(self.zqso[q]).to(u.kpc)
                theta = np.rad2deg((np.arctan2(dcorr,da.value))) * u.deg
                c1    = SkyCoord(ra=self.ra[q]*u.degree,dec=self.dec[q]*u.degree)

                c1first = c1.separation(firstcat)
                c1lotss = c1.separation(lotsscat)
                c1vlass = c1.separation(vlasscat)
                c1racs  = c1.separation(racscat)

                fdx = np.extract(c1first < theta, range(c1first.size))
                ldx = np.extract(c1lotss < theta, range(c1lotss.size))
                vdx = np.extract(c1vlass < theta, range(c1vlass.size))
                rdx = np.extract(c1racs  < theta, range(c1racs.size))

                if np.sum(np.concatenate((fdx,ldx,vdx,rdx))) > 0: # We have a detection...
                    if vdx.size > 0:
                        vdxmin = np.extract(c1vlass[vdx] == np.min(c1vlass[vdx]), vdx)
                        ftotvlass[q] = np.sum(vlassfint[vdx])
                        if ftotvlass[q] > 0:
                            rcorvlass[q] = vlassfpeak[vdxmin]/ftotvlass[q]
                        nv += 1
                    elif np.min(c1vlass) < (3 * u.arcmin).to(u.deg):
                        ftotvlass[q] = 0.12
                    if fdx.size > 0:
                        fdxmin = np.extract(c1first[fdx] == np.min(c1first[fdx]), fdx)
                        ftotfirst[q] = np.sum(firstfint[fdx])
                        rcorfirst[q] = firstfpeak[fdxmin]/ftotfirst[q]
                        nf += 1
                    elif np.min(c1first) < (3 * u.arcmin).to(u.deg):
                        ftotfirst[q] = 0.15
                    if rdx.size > 0:
                        rdxmin = np.extract(c1racs[rdx] == np.min(c1racs[rdx]), rdx)
                        ftotracs[q] = np.sum(racsfint[rdx])
                        rcorracs[q] = racsfpeak[rdxmin]/ftotracs[q]
                        nr += 1
                    elif np.min(c1racs) < (3 * u.arcmin).to(u.deg):
                        ftotracs[q] = 0.4
                    if ldx.size > 0:
                        ldxmin = np.extract(c1lotss[ldx] == np.min(c1lotss[ldx]), ldx)
                        ftotlotss[q] = np.sum(lotssfint[ldx])
                        rcorlotss[q] = lotssfpeak[ldxmin]/ftotlotss[q]
                        nl += 1
                    elif np.min(c1lotss) < (3 * u.arcmin).to(u.deg):
                        ftotlotss[q] = 0.1

                    prtstr = f"{q:6d} {self.zqso[q]:7.5f}"
                    prtstr = prtstr + (" {:3d} {:6.2f} {:7.3f}").format(vdx.size,ftotvlass[q],rcorvlass[q])
                    prtstr = prtstr + (" {:3d} {:6.2f} {:7.3f}").format(fdx.size,ftotfirst[q],rcorfirst[q])
                    prtstr = prtstr + (" {:3d} {:6.2f} {:7.3f}").format(rdx.size,ftotracs[q], rcorracs[q])
                    prtstr = prtstr + (" {:3d} {:6.2f} {:7.3f}").format(ldx.size,ftotlotss[q],rcorlotss[q])
                    print(time.asctime()," ",prtstr)
                    f.write(prtstr+'\n')

                    pdx = np.extract((ftotfirst > 0) & (ftotlotss > 0) & (ftotvlass > 0) & (ftotracs > 0), range(ftotfirst.size))
                    if pdx.size > 0:
                        plt.clf()
                        dpdx = np.extract((ftotfirst[pdx] > 0.15)  & (ftotlotss[pdx] > 0.1)  & (ftotvlass[pdx] > 0.12)  & (ftotracs[pdx] > 0.4),  pdx)
                        if dpdx.size > 0: plt.plot(np.log(ftotfirst[dpdx]/ftotvlass[dpdx]),np.log(ftotracs[dpdx]/ftotlotss[dpdx]),"bo", markersize=20)
                        dpdx = np.extract((ftotfirst[pdx] > 0)     & (ftotlotss[pdx] > 0)    & (ftotvlass[pdx] > 0)     & (ftotracs[pdx] == 0.4), pdx)
                        if dpdx.size > 0: plt.plot(np.log(ftotfirst[dpdx]/ftotvlass[dpdx]),np.log(ftotracs[dpdx]/ftotlotss[dpdx]),"ro", markersize=16)
                        dpdx = np.extract((ftotfirst[pdx] > 0)     & (ftotlotss[pdx] > 0)    & (ftotvlass[pdx] == 0.12) & (ftotracs[pdx] > 0),    pdx)
                        if dpdx.size > 0: plt.plot(np.log(ftotfirst[dpdx]/ftotvlass[dpdx]),np.log(ftotracs[dpdx]/ftotlotss[dpdx]),"go", markersize=12)
                        dpdx = np.extract((ftotfirst[pdx] > 0)     & (ftotlotss[pdx] == 0.1) & (ftotvlass[pdx] > 0)     & (ftotracs[pdx] > 0),    pdx)
                        if dpdx.size > 0: plt.plot(np.log(ftotfirst[dpdx]/ftotvlass[dpdx]),np.log(ftotracs[dpdx]/ftotlotss[dpdx]),"mo", markersize=8)
                        dpdx = np.extract((ftotfirst[pdx] == 0.15) & (ftotlotss[pdx] > 0)    & (ftotvlass[pdx] > 0)     & (ftotracs[pdx] > 0),    pdx)
                        if dpdx.size > 0: plt.plot(np.log(ftotfirst[dpdx]/ftotvlass[dpdx]),np.log(ftotracs[dpdx]/ftotlotss[dpdx]),"yo", markersize=4)
                        plt.xlabel("F(1.4GHz)/F(3GHz)")
                        plt.ylabel("F(0.14GHz)/F(0.89GHz)")
                        plt.xscale("log")
                        plt.yscale("log")

                        plt.title(("qnum {} vlass {} first {} racs-lo {} lotss {}").format(q,nv,nf,nr,nl))
                        plt.pause(0.0001)

    ###############################################################################################
    def firstf1(self, x, c):
        return x*x*c
    def firstf2(self, x, c):
        return c[0] / (1 + np.power(x/c[1], c[2]))

    def firstfit(self,cntfile):
        data = ascii.read(cntfile, format='no_header')
        dbin = data['col1']
        ncts = data['col2']
        rcts = data['col3']
        i = 0
        dbin2 = np.array([])
        ncts2 = np.array([])
        rcts2 = np.array([])
        while (i < dbin.size):
            ilo = i
            while ((np.sum(rcts[ilo:i]) < 50) or (np.sum(ncts[ilo:i]) < 50)) and ((i < dbin.size)):
                i += 1
            dbin2 = np.append(dbin2, sp.stats.gmean(dbin[ilo:i]))
            ncts2 = np.append(ncts2, np.sum(ncts[ilo:i]))
            rcts2 = np.append(rcts2, np.sum(rcts[ilo:i]))

        rcts2 *= np.sum(ncts2)/np.sum(rcts2)

        xdx = np.extract(dbin2 < 2.0e+3, range(dbin2.size))
        c = np.array([4.04842808e+02,  5.42456219e+01, -1.60225223e+00])
        f1 = np.cumsum(rcts2)

        diff = (np.cumsum(ncts2) - f1) / np.sqrt(np.cumsum(ncts2) + f1)
        plt.clf()
#        plt.subplot(211)
#        plt.step(dbin2[xdx],diff[xdx])
        plt.step(dbin2[xdx],ncts2[xdx]/rcts2[xdx] - 1, "b")
        plt.plot(dbin2[xdx],np.zeros(xdx.size), "r--")
        plt.xscale("log")
        plt.ylim([-0.5,2])
#        plt.subplot(212)
#        plt.step(dbin2[xdx],np.cumsum(ncts2)[xdx])
#        plt.step(dbin2[xdx],f1[xdx])
#        plt.yscale("log")

        print(np.extract(diff[xdx] == np.min(diff[xdx]), dbin2[xdx]), np.min(diff[xdx]))

        f2 = self.firstf2(dbin2, c)
        chi = np.log(np.cumsum(ncts2)) - np.log(f1 + f2)
        chisq = np.sum(chi*chi)

##        scale = 0.1
##        while scale > 1.0e-7:
##            done = int(10000 * (-np.log(scale)))
##            while done > 0:
##                oldc = np.copy(c)
##                oldchisq = chisq
##                c = np.multiply(oldc, (1-scale/2) + scale*np.random.rand(c.size))
##                f2 = self.firstf2(dbin2, c)
##                chi = np.log(np.cumsum(ncts2)) - np.log(f1 + f2)
##                chisq = np.sum(chi*chi)
##
##                if oldchisq < chisq:
##                    c= np.copy(oldc)
##                    chisq = oldchisq
##                    done -= 1
##                else:
##                    plt.clf()
##                    plt.step(dbin2[xdx],np.cumsum(ncts2)[xdx],"r", linewidth=3.0)
##                    plt.ylim([np.min(np.cumsum(ncts2)[xdx]),np.max(np.cumsum(ncts2)[xdx])])
##                    plt.step(dbin2[xdx],f1[xdx]+f2[xdx],"m--", linewidth=3.0)
##                    plt.step(dbin2[xdx],f1[xdx],"y--")
##                    plt.step(dbin2[xdx],f2[xdx],"g--")
##
##                    plt.xscale("log")
##                    plt.yscale("log")
##                    dumstr = str(done)+" "+str(scale)
##                    for d in range(c.size):
##                        dumstr += " "+str(c[d])
##                    dumstr += " "+str(chisq)
##                    plt.title(dumstr)
##                    plt.pause(0.0001)
##                    done = int(10000 * (-np.log(scale)))
##            print(c,chisq,scale)
##            scale /= 10
##
##        f2 = self.firstf2(dbin2, c)
##        plt.clf()
##        plt.step(dbin2[xdx],np.cumsum(ncts2)[xdx],"r", linewidth=3.0)
##        plt.ylim([np.min(np.cumsum(ncts2)[xdx]),np.max(np.cumsum(ncts2)[xdx])])
##        plt.step(dbin2[xdx],f1[xdx]+f2[xdx],"m--", linewidth=3.0)
##        plt.step(dbin2[xdx],f1[xdx],"y--")
##        plt.step(dbin2[xdx],f2[xdx],"g--")
##        plt.xscale("log")
##        plt.yscale("log")
##        dumstr = str(done)+" "+str(scale)
##        for d in range(c.size):
##            dumstr += " "+str(c[d])
##        dumstr += " "+str(chisq)
##        plt.title(dumstr)
##        plt.pause(0.0001)

        return (chisq,c)

    ###############################################################################################
    def firstsrcsingle(self, dcorr):
        cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)

        nfirst,firstra,firstdec,fpeak,fint = self.readfirst()
        print('Read in ',nfirst,' FIRST objects')

        firstcat  = SkyCoord(ra=firstra*u.degree,dec=firstdec*u.degree)
        firstra  = np.deg2rad(firstra)
        firstdec = np.deg2rad(firstdec)

        udum, udx = np.unique(self.qnum, return_index=True)

#       sdsscoords = SkyCoord(ra=self.ra[udx]*u.degree,dec = self.dec[udx]*u.degree)
#       da    = cosmo.angular_diameter_distance(self.zqso[udx]).to(u.kpc)
#       theta = np.rad2deg(np.arctan2(dcorr,da.value))
#       idxc, idxfirstcat, d2d, d3d = sdsscoords.search_around_sky(firstcat, theta*u.deg)


        print('Searching through ',udx.size,' objects')
        dfsingle = np.array([])
        oldchisq = 999999999999
        c0 = 0.181
        c2 = 7.898e-7
        r0 = 2.069
        alpha = -1.868
        for q in range(udx.size):
            da    = cosmo.angular_diameter_distance(self.zqso[udx[q]]).to(u.kpc)
            c1    = SkyCoord(ra=self.ra[udx[q]]*u.degree,dec=self.dec[udx[q]]*u.degree)

            c1first   = c1.separation(firstcat) #.to(u.arcsec).value
            ddx   = np.extract(c1first.value < 90, range(c1first.size))
            df    = np.tan((c1first[ddx]).to(u.rad)) * da.value
            if np.min(df) <= dcorr:
                edx = np.extract(df < dcorr, range(ddx.size))
                if edx.size == 1: # This is a single source, keep the df value
                    dfsingle = np.append(dfsingle, df[edx])
                    print('Object ',udum[q],' matched at d = ',dfsingle[-1],' kpc (total matches: ',dfsingle.size,'/',q+1,')')
                    if dfsingle.size > 1500:
                        dfsort = np.sort(dfsingle)
                        cts    = (1+np.arange(dfsort.size))/(1+dfsort.size)
                        n1 = c0 / (1 + np.power(dfsort/r0, alpha))
                        n2 = c2 * np.square(dfsort)
                        n  = n1 + n2

#                        cts = np.where(cts > 0, cts, 1.0e-30)
                        n1  = np.where(n1  > 0, n1,  1.0e-30)
                        n2  = np.where(n2  > 0, n2,  1.0e-30)
                        n   = np.where(n   > 0, n,   1.0e-30)
                        chisq = np.sum(np.square(np.log10(np.extract(cts > 0, cts))-np.log10(np.extract(cts > 0,n))))

                        niterate = 3000
                        plt.clf()
                        plt.step(np.extract(cts > 1.0e-10, dfsort), np.extract(cts > 1.0e-10, cts),  "r")
                        plt.plot(np.extract(n1  > 1.0e-10, dfsort), np.extract(n1  > 1.0e-10, n1),   "c--")
                        plt.plot(np.extract(n2  > 1.0e-10, dfsort), np.extract(n2  > 1.0e-10, n2),   "m--")
                        plt.plot(np.extract(n   > 1.0e-10, dfsort), np.extract(n   > 1.0e-10, n),    "b")
                        if np.min(cts > 1):
                            plt.xlim([np.min(np.extract(cts > 1, dfsort)) , dfsort[-1]])
                        else:
                            plt.xlim([np.min(np.extract(cts > 0, dfsort)) , dfsort[-1]])
                        plt.ylim([0.75*np.min(np.extract(cts>1.0e-10, cts)),1.0])
                        plt.yscale("log")
                        plt.xscale("log")
                        plt.title(str(niterate)+" "+str(chisq)+" "+str(chisq-oldchisq)+" "+str(c0)+" "+str(c2)+" "+str(r0)+" "+str(alpha))
                        plt.xlabel("Projected distance (kpc)")
                        plt.pause(0.0001)

                        while niterate > 0:
                            oldchisq = chisq
                            oldc0    = c0
                            oldc2    = c2
                            oldr0    = r0
                            oldalpha = alpha

                            c0    *= 0.95 + 0.1*(np.random.rand(1))[0]
                            c2    *= 0.95 + 0.1*(np.random.rand(1))[0]
                            r0    *= 0.95 + 0.1*(np.random.rand(1))[0]
                            alpha *= 0.95 + 0.1*(np.random.rand(1))[0]
                            n1 = c0 / (1 + np.power(dfsort/r0, alpha))
                            n2 = c2 * np.square(dfsort)
                            n  = n1 + n2
                            n1  = np.where(n1  > 0, n1,  1.0e-30)
                            n2  = np.where(n2  > 0, n2,  1.0e-30)
                            n   = np.where(n   > 0, n,   1.0e-30)
                            chisq = np.sum(np.square(np.log10(np.extract(cts > 0, cts))-np.log10(np.extract(cts > 0,n))))
#                            chisq = np.sum(np.square(np.log10(cts)-np.log10(n)))
                            if chisq < oldchisq:
                                plt.clf()
                                plt.step(np.extract(cts > 1.0e-10, dfsort), np.extract(cts > 1.0e-10, cts),  "r")
                                plt.plot(np.extract(n1  > 1.0e-10, dfsort), np.extract(n1  > 1.0e-10, n1),   "c--")
                                plt.plot(np.extract(n2  > 1.0e-10, dfsort), np.extract(n2  > 1.0e-10, n2),   "m--")
                                plt.plot(np.extract(n   > 1.0e-10, dfsort), np.extract(n   > 1.0e-10, n),    "b")
                                if np.min(cts > 1):
                                    plt.xlim([np.min(np.extract(cts > 1, dfsort)) , dfsort[-1]])
                                else:
                                    plt.xlim([np.min(np.extract(cts > 0, dfsort)) , dfsort[-1]])
                                plt.ylim([0.75*np.min(np.extract(cts>1.0e-10, cts)),1.0])
                                plt.yscale("log")
                                plt.xscale("log")
                                plt.title(str(niterate)+" "+str(chisq)+" "+str(chisq-oldchisq)+" "+str(c0)+" "+str(c2)+" "+str(r0)+" "+str(alpha))
                                plt.xlabel("Projected distance (kpc)")
                                plt.pause(0.0001)
                                niterate = 3000
                            else:
                                niterate -= 1
                                chisq = oldchisq
                                c0    = oldc0
                                c2    = oldc2
                                r0    = oldr0
                                alpha = oldalpha
        return chisq,c0,c2,r0,alpha



#                if np.min(np.append(dfirst,dlotss)) < self.dbin[dcorrbin]:
#                    print('Object ',udum[q],' has ',(np.extract(dfirst < self.dbin[dcorrbin], dfirst)).size,' matches in FIRST within ',self.dbin[dcorrbin],' kpc')
#                    print('Object ',udum[q],' has ',(np.extract(dlotss < self.dbin[dcorrbin], dlotss)).size,' matches in LOTSS within ',self.dbin[dcorrbin],' kpc')


#                c1r   = c1.separation(randcat) #.to(u.arcsec).value
#                dr    = np.tan((np.extract(c1r.value < theta, c1r)).to(u.rad)) * da.value
#                dre   = np.extract(dr < np.max(self.dbin), dr)
#                rcts,dum = np.histogram(dre, bins=self.dbin)
#                self.firsttpcfrcts[:-1] += rcts

#                tpcf = np.multiply(self.firsttpcfncts / np.sum(self.firsttpcfncts), np.sum(self.firsttpcfrcts) / np.where(self.firsttpcfrcts > 0, self.firsttpcfrcts, 1.0e-30)) - np.ones(n)

#                tdx = np.extract(self.firsttpcfrcts > 0, range(tpcf.size))
#                dcorrbin = np.min(np.extract((tpcf[tdx] > 0) & (self.firsttpcfrcts[tdx] > 0), range(tdx.size)))
#                while np.max(tpcf[tdx[dcorrbin]:tdx[dcorrbin+2]]) > 0:
#                    dcorrbin += 1
#                dcorrbin = tdx[dcorrbin]

#                plt.step(self.dbin[tdx], tpcf[tdx], "r")
#                plt.plot(self.dbin[tdx], np.zeros(tdx.size), "b--")
#                if (np.sum(self.firstncts) > 10):
#                    plt.clf()
#                    tdx = np.extract(self.firstncts > 0, range(self.dbin.size))
#                    plt.step(self.dbin[tdx],self.firstncts[tdx],"b")
#                    if (np.sum(self.lotssncts) > 10):
#                        tdx = np.extract(self.lotssncts > 0, range(self.dbin.size))
#                        plt.step(self.dbin[tdx],self.lotssncts[tdx] * np.sum(self.firstncts) / np.sum(self.lotssncts),"g")
#                    if (np.sum(self.wenssncts) > 10):
#                        tdx = np.extract(self.wenssncts > 0, range(self.dbin.size))
#                        plt.step(self.dbin[tdx],self.wenssncts[tdx] * np.sum(self.firstncts) / np.sum(self.wenssncts),"c")
#                    if (np.sum(self.sumssncts) > 10):
#                        tdx = np.extract(self.sumssncts > 0, range(self.dbin.size))
#                        plt.step(self.dbin[tdx],self.sumssncts[tdx] * np.sum(self.firstncts) / np.sum(self.sumssncts),"c")
#                    plt.xscale("log")
#                    plt.yscale("log")
#                    plt.title(("qnum {} first {:d} lotss {:d} wenss {:d} sumss {:d}").format(self.qnum[udx[q]], np.sum(self.firstncts), np.sum(self.lotssncts), np.sum(self.wenssncts), np.sum(self.sumssncts)))
#                    plt.xlabel("Projected distance (kpc)")
#                    plt.pause(0.0001)









###############################################################################################
class clustcompfit:
    def __init__(self):
        data = ascii.read('D:/ganguly/SDSSAAL/dr18/cluster_components.ecsv', format="ascii.ecsv", guess=False)
        self.wave = np.array(data['wave'])
        self.comp = np.array([data['c0'],data['c1'],data['c2'],data['c3'],data['c4']])

    def __call__(self,inwave,coeff):
        spec = np.zeros(self.wave.size)
        for i in range(5): spec += coeff[i] * self.comp[:,i]
        specin = np.zeros(inwave.size)
        sdx = np.extract((inwave >= self.wave[0]) & (inwave <= self.wave[-1]), range(inwave.size))
        specin[sdx] = np.interp(inwave[sdx],self.wave,spec)
        return specin
    
"""