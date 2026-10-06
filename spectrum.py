import os
import numpy             as np
#import numdifftools      as nd
#import matplotlib.pyplot as plt
#import random
#import time

#from astropy.convolution   import convolve
#from astropy.modeling      import models
#from astropy.io            import ascii
from astropy.io             import fits
#from astropy.table         import QTable
#from scipy.interpolate     import CubicSpline, make_interp_spline, PchipInterpolator, Akima1DInterpolator, UnivariateSpline
#from scipy.optimize        import minimize
#from scipy.optimize        import SR1
#from scipy.special         import erf

#from abslinesys            import abslinesys
#from doppler               import calcvel
#from doppler               import calcvelzz
#from doppler               import calcwave
#from atomic                import atomic
#from fitmodel              import fitmodel
#from chisqfunc             import chisqfunc

###############################################################################################
class spectrum:
    ###############################################################################################
    def __init__(self, 
                 mypars,
                 fitsfiles, 
                 qnum,
                 zqso,
                 ntabs = 0
                 ):

        self.mypars = mypars
        self.qnum = qnum

        gonogo = True
        for fitsfile in fitsfiles:
            if not os.path.exists(fitsfile):
                gonogo = False

        if gonogo:
            totwave,totflux,totivar = self.combine_spec(zqso,
                                                        fitsfiles,
                                                        ntabs = ntabs+1
                                                        )


    ###############################################################################################
    def combine_spec(self,
                     zqso,
                     fitsfiles,
                     verbose=True,
                     ntabs = 0
                     ):
        with fits.open(fitsfiles[0]) as hdul:
            totflux = hdul[1].data['flux']
            totwave = 10.0**hdul[1].data['loglam'] / (1.0+zqso)
            totivar = hdul[1].data['ivar']
        # Scale the spectrum by the average flux in the 1700-1738 A range
        norm0 = np.average(np.extract((totwave > 1700) & (totwave < 1738.44), totflux))
        totflux = totflux / norm0
        totivar = totivar * norm0 * norm0
        # Filter out wavelength bins where the recorded ivar is zero
        nzdx      = np.extract(totivar > 0, range(totivar.size))
        totflux   = totflux[nzdx]
        totwave   = totwave[nzdx]
        totjvar   = totivar[nzdx]
        totivar   = np.multiply(totivar[nzdx], 1/np.square(totflux))
        totlgflux = np.multiply(totjvar, np.log10(np.where(totflux <= 1/np.sqrt(totjvar), 1/np.sqrt(totjvar), totflux))) # Start of weighted sum
        nq        = np.ones(nzdx.size, dtype=np.int32)

        if verbose: print("\t"*ntabs + f'Beginning with {totwave.size} bins from {totwave[0]} to {totwave[-1]}')

        if fitsfiles.size > 1: # This is not the only one
            for j in np.arange(1,fitsfiles.size,dtype=np.int32):
                with fits.open(fitsfiles[j]) as hdul:
                    flux = hdul[1].data['flux']
                    wave = 10.0**hdul[1].data['loglam'] / (1.0+zqso)
                    ivar = hdul[1].data['ivar']
                # Scale the spectrum by the average flux in the 1700-1738 A range
                norm = np.average(np.extract((wave > 1700) & (wave < 1738.44), flux))
                flux = flux / norm
                ivar = ivar * norm * norm
                # Filter out wavelength bins where the recorded ivar is zero
                nzdx   = np.extract(ivar > 0, range(ivar.size))
                flux   = flux[nzdx]
                wave   = wave[nzdx]
                ivar   = ivar[nzdx]

                logf   = np.log10(np.where(flux <= 1/np.sqrt(ivar), 1/np.sqrt(ivar), flux))

                # Prepend wavelengths smaller than totwave
                pdx = np.array([])
                wlo = np.max(wave[0])
                if (np.min(totwave) > np.min(wave)):
                    pdx       = np.extract(wave < np.min(totwave), np.arange(wave.size, dtype=np.int32))
                    badness   = np.std(flux[pdx])/np.sqrt(np.average(ivar[pdx]))
                    if badness < 1.0e+5:
                        wlo = np.max(wave[pdx])
                        if verbose: print("\t"*ntabs + f'Prepending {pdx.size} bins from {wave[pdx[0]]} to {wave[pdx[-1]]} ({badness})')
                        totwave   = np.concatenate((wave[pdx],                                      totwave))
                        totlgflux = np.concatenate((np.multiply(ivar[pdx], logf[pdx]),            totlgflux))
                        totjvar   = np.concatenate((ivar[pdx],                                      totjvar)) # This is for summing the weights
                        totivar   = np.concatenate((np.multiply(ivar[pdx], 1/np.square(flux[pdx])), totivar)) # This is for propogating errors
                        nq        = np.concatenate((np.ones(pdx.size,dtype=np.int32),                    nq))

                # Append wavelengths larger than totwave
                adx = np.array([])
                whi = np.min(wave[-1])
                if (np.max(totwave) < np.max(wave)):
                    adx       = np.extract(wave > np.max(totwave), np.arange(wave.size, dtype=np.int32))
                    badness   = np.std(flux[adx])/np.sqrt(np.average(ivar[adx]))
                    if badness < 1.0e+5:
                        whi = np.min(wave[adx])
                        if verbose: print("\t"*ntabs + f'Appending {adx.size} bins from {wave[adx[0]]} to {wave[adx[-1]]} ({badness})')
                        totwave   = np.concatenate((  totwave,                                      wave[adx]))
                        totlgflux = np.concatenate((totlgflux,              np.multiply(ivar[adx], logf[adx])))
                        totjvar   = np.concatenate((  totjvar,                                      ivar[adx]))
                        totivar   = np.concatenate((  totivar, np.multiply(ivar[adx], 1/np.square(flux[adx]))))
                        nq        = np.concatenate((       nq,               np.ones(adx.size,dtype=np.int32)))

                # Interpolate and add in the overlapping wavelength regions
                # Problem here that we're using an already-expanded version of totwave?
                ndx     = np.extract((totwave > wlo) & (totwave < whi), range(totwave.size))
                ndx2    = np.extract((wave > np.min(totwave[ndx])) & (wave < np.max(totwave[ndx])), range(wave.size))
                badness = np.std(flux[ndx2])/np.sqrt(np.average(ivar[ndx2]))
                if badness < 1.0e+5:
                    if verbose: print("\t"*ntabs + f'Interpolating and combining {ndx.size} bins from {wlo} to {whi} ({badness})')
                    spf = np.interp(totwave[ndx], wave, np.where(flux <= 1/np.sqrt(ivar), 1/np.sqrt(ivar), flux))
                    spi = np.interp(totwave[ndx], wave, ivar)

                    totlgflux[ndx] += np.multiply(spi, np.log10(spf)) # Note: This is a geometric mean since we averaging in log-space
                    totjvar[ndx]   += spi
                    totivar[ndx]   += np.multiply(spi, 1/np.square(spf))
                    nq[ndx]        += 1

                # If there are missing stretches of wavelength not covered by the previous spectrum, fill them in with the current spectrum
                dw       = np.zeros(totwave.size)
                dw[1:-1] = 0.5*(totwave[2:] - totwave[:-2])
                dw[0]    = dw[1]
                dw[-1]   = dw[-2]
                missdx   = np.extract(dw > wave[-1]-wave[-2], range(totwave.size))
                mdx1     = -1
                for m in range(missdx.size):
                    if m > mdx1:
                        mdx1 = m
                        if mdx1 < missdx.size-1:
                            while (missdx[mdx1+1]-missdx[mdx1] == 1) and (mdx1 < missdx.size-1):
                                mdx1 += 1
                                if mdx1 >= missdx.size-2:
                                    break
                        wdx       = np.extract((wave > totwave[missdx[m]]) & (wave < totwave[missdx[mdx1]]), range(wave.size))
                        if wdx.size > 2:
                            scale = -1
                            done = 0
                            i = 0
                            while (done == 0):
                                i = i+1
                                mstart = np.max([missdx[m]-i,    0])
                                mend   = np.min([missdx[mdx1]+i, totlgflux.size-1])
                                wstart = np.min([wdx[-1]-i,      0])
                                wend   = np.min([wdx[0]+i,       flux.size-1])
                                subf1  = np.where(flux[wdx[0]:wend]    < 0, 1/np.sqrt(ivar[wdx[0]:wend]),    flux[wdx[0]:wend])
                                subf2  = np.where(flux[wstart:wdx[-1]] < 0, 1/np.sqrt(ivar[wstart:wdx[-1]]), flux[wstart:wdx[-1]])
                                oldscale = scale
                                scalelo = np.average(np.power(10.0, np.multiply(totlgflux[mstart:missdx[m]],  1 / totjvar[mstart:missdx[m]])))/np.average(subf1)
                                scalehi = np.average(np.power(10.0, np.multiply(totlgflux[missdx[mdx1]:mend], 1 / totjvar[missdx[mdx1]:mend])))/np.average(subf2)
                                scale   = np.average([scalelo, scalehi])
                                if np.isnan(scale):
                                    scale = scalelo
                                    if np.isnan(scale):
                                        scale = scalehi
                                    if np.isnan(scale):
                                        scale = 1
                                if np.isnan(scale):
                                    print("\t"*ntabs + "Encountered NaN!")
                                    print("\t"*ntabs + "scalelo = ",scalelo," scalehi = ",scalehi)
                                    print("\t"*ntabs + "totlgflux[mstart:missdx[m]] = ",  totlgflux[mstart:missdx[m]])
                                    print("\t"*ntabs + "totlgflux[missdx[mdx1]:mend] = ", totlgflux[missdx[mdx1]:mend])
                                    print("\t"*ntabs + "subf1 = ",subf1)
                                    print("\t"*ntabs + "subf2 = ",subf2)
                                    input("Paused - hit ctrl-C to stop")
                                if verbose: print("                 ",i,mstart,missdx[m],mend,wstart,wdx[0],wdx[-1],wend,scale,np.fabs((scale-oldscale)/oldscale))
                                if (np.fabs((scale-oldscale)/oldscale) < 0.01) and (scale > 0):
                                    done = 1
                            badness = np.std(flux[wdx])/np.sqrt(np.average(ivar[wdx]))
                            if badness < 1.0e+5:
                                if verbose: print("\t"*ntabs + f'Filling in {wdx.size} missing values from {wave[wdx[0]]} to {wave[wdx[-1]]} with scale = {scale}  ({badness})')
                                subf      = np.where(flux[wdx] > 1/np.sqrt(ivar[wdx]), flux[wdx], 1/np.sqrt(ivar[wdx])) * scale
                                subiv     = ivar[wdx] / (scale * scale)
                                totwave   = np.append(totwave,   wave[wdx])
                                totlgflux = np.append(totlgflux, np.multiply(subiv, np.log10(subf)))
                                totjvar   = np.append(totjvar,   subiv)
                                totivar   = np.append(totivar,   np.multiply(subiv,  1/np.square(subf)))
                                nq        = np.append(nq,        np.ones(wdx.size,dtype=np.int32))
                sdx       = np.argsort(totwave)
                totwave   = totwave[sdx]
                totlgflux = totlgflux[sdx]
                totjvar   = totjvar[sdx]
                totivar   = totivar[sdx]
                nq        = nq[sdx]

        avetotlgflux = np.multiply(totlgflux, 1/totjvar)
        totflux = np.power(10.0, avetotlgflux)

        if verbose: print("\t"*ntabs + f'Returning {totwave.size} bins from {totwave[0]} to {totwave[-1]} ({totflux.size} {totivar.size})')

        return totwave,totflux,totivar

"""
        data = QTable.read(datfile, format="ascii.ecsv")
        self.wavelength = np.array(data['wave'])
        self.flux       = np.array(data['flux'])
        self.ivar       = np.array(data['ivar'])

        fnorm = np.average(np.extract((self.wavelength > 1700) & (self.wavelength < 1738.44), self.flux))
        self.flux /= fnorm
        self.ivar *= fnorm*fnorm

        self.nv  = self.ivar.size
        self.snr = np.multiply(self.flux, np.sqrt(self.ivar))

        self.dwave              = np.empty(self.nv)
        self.dwave[1:self.nv-1] = 0.5*np.subtract(self.wavelength[2:self.nv],self.wavelength[0:self.nv-2])
        self.dwave[0]           = self.dwave[1]
        self.dwave[self.nv-1]   = self.dwave[self.nv-2]

        self.dv = 299792.458 * self.dwave/self.wavelength

        myatoms      = atomic(self.wavelength[0],self.wavelength[-1])
        if (myatoms.getspecies(7,5)).size > 0:
            self.vrange  = -np.min(calcvel(self.wavelength,myatoms.wave[(myatoms.getspecies(7,5))[0]])                       )
        else:
            self.vrange = (calcvel(self.wavelength[0],np.array([1238.821])))[0]

        self.fpx         = np.array([])
        self.waveall     = np.array([])
        self.legfitrange = np.array([])
        if (myatoms.getspecies(6,4)).size > 0:
            self.fpx     = np.extract(calcvel(self.wavelength,myatoms.wave[(myatoms.getspecies(6,4))[0]]) < 5000, range(self.nv))
            self.waveall = self.wavelength[self.fpx]
            self.legfitrange = (self.waveall - np.mean(self.waveall))/(self.waveall[-1]-self.waveall[0])

        self.ew     = np.zeros(self.nv)
        self.sew    = np.zeros(self.nv) + np.max(self.flux)
        self.bigew  = np.zeros(self.nv)
        self.bigsew = np.zeros(self.nv) + np.max(self.flux)


        # Set things up for the line spread function
        dwres  = np.average(self.waveall)/(2.35*R)
        self.dvres = 299792.458/(2.35*R)

        if self.waveall.size > 3:
            v   = np.squeeze(calcvel(self.waveall, np.mean(self.waveall)))
            dv  = np.zeros(v.size)
            dv[1:-1] = 0.5*(v[2:] - v[:-2])
            dv[0] = dv[1]
            dv[-1] = dv[-2]
            ldx = np.extract((v>-20*self.dvres) & (v<20*self.dvres), range(v.size))
            n = 1
            while ldx.size <= 2:
                n += 1
                ldx = np.extract((v>-20*self.dvres-n) & (v<20*self.dvres+n), range(v.size))
            if (ldx.size % 2 == 0) and (ldx.size > 2):
                if ldx[-1]+1 < v.size:
                    ldx      = np.append(ldx, ldx[ldx.size-1]+1)
                else:
                    ldx      = ldx[:-2]
            modw         = calcwave(v[ldx], np.mean(self.waveall)) - np.mean(self.waveall)
#            self.lsf   = np.squeeze(np.exp(-0.5*np.square(modw/dwres)))
            self.lsf   = np.squeeze(np.exp(-0.5*np.square(v[ldx]/self.dvres)))
            lsfintegral = np.sum(self.lsf)
            self.lsf /= lsfintegral

            tdx = np.extract(self.lsf > 0.5*np.max(self.lsf), ldx)
        else:
            self.lsf   = np.array([])

        self.msigma     = 0
        self.msigmawave = 0
        self.fittime    = 0

    ###############################################################################################
    def aalsearch(self,sdss,box1,box2):

        vdx,nnflux,nnferr = self.normspec(sdss,box1,box2)

        self.ew[vdx]     = np.multiply(1.0 - nnflux, self.dwave[vdx])
        self.sew[vdx]    = np.multiply(      nnferr, self.dwave[vdx])
        self.bigew[vdx]  = convolve(self.ew[vdx],  np.ones(5)/5, boundary='extend') # vdx.size
        self.bigsew[vdx] = convolve(self.sew[vdx], np.ones(5)/5, boundary='extend') # vdx.size
        zdx              = np.extract(self.bigew < 0, range(self.bigew.size))
        if zdx.size > 0:
            self.bigew[zdx] = 0.0
        nsigspec         = np.multiply(self.bigew, 1/self.bigsew)

        velocity = np.extract(np.fabs(self.nvbvel) < 5000.0, self.nvbvel)

        # Stack the joint detection probablilities of the N V doublet, the C IV doublet, and H I Ly-a
#        nsig  = np.multiply(erf(np.extract(np.fabs(self.nvbvel) < 5000.0, nsigspec)), erf(np.interp(velocity, self.nvrvel,  nsigspec)))
#        nsig += np.multiply(erf(np.interp(velocity, self.civbvel, nsigspec)),         erf(np.interp(velocity, self.civrvel, nsigspec)))
#        nsig +=             erf(np.interp(velocity, self.hivel,   nsigspec))
        nsig  = np.multiply(np.extract(np.fabs(self.nvbvel) < 5000.0, 2*nsigspec), np.interp(velocity, self.nvrvel,  nsigspec))
        nsig += np.multiply(np.interp(velocity, self.civbvel, 2*nsigspec),         np.interp(velocity, self.civrvel, nsigspec))
        nsig +=             np.interp(velocity, self.hivel,   3*nsigspec)

        print("spectrum.aalsearch: nsig.shape = ",nsig.shape," velocity.shape = ",velocity.shape)

        vabs = np.array([])
        nsigcopy = np.copy(nsig)
        while np.max(nsigcopy) > 5000:
#        while np.max(nsigcopy) > 500000:
            vab = np.min(np.extract(nsigcopy == np.max(nsigcopy), velocity))
            print(vab, np.max(nsigcopy))

            # Fit a Gaussian to the nsigcopy array at the location of vab
            nit = 1000
            s    = 50
            g = models.Gaussian1D(amplitude=np.max(nsigcopy),mean=vab,stddev=s)
            chisq=np.sum(np.square(nsigcopy-g(velocity)))
            while nit > 0:
                olds     = s
                oldvab   = vab
                oldchisq = chisq

                vab *= random.uniform(0.99,1.01)
                s   *= random.uniform(0.95,1.05)
                g     = models.Gaussian1D(amplitude=np.max(nsigcopy),mean=vab,stddev=s)
                chisq = np.sum(np.square(nsigcopy-g(velocity)))

                if chisq-oldchisq > -0.1:
                    nit -= 1
                    s    = olds
                    vab  = oldvab
                    chisq = oldchisq
                    g     = models.Gaussian1D(amplitude=np.max(nsigcopy),mean=vab,stddev=s)

                else:
                    plt.clf()
                    plt.subplot(311)
                    plt.step(self.hivel[vdx],   nnflux,           'g')
                    plt.step(self.civbvel[vdx], nnflux+0.5,       'b')
                    plt.step(self.civrvel[vdx], nnflux+1.0,       'r')
                    plt.step(self.nvbvel[vdx],  nnflux+1.5,       'c')
                    plt.step(self.nvrvel[vdx],  nnflux+2.0,       'm')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.0, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.5, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 2.0, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 2.5, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 3.0, 'k')
                    for i in range(vabs.size):
                        plt.plot([vabs[i],vabs[i]],[0.0,4.0])
                    plt.xlim([-5000,5000])
                    plt.ylim([-0.1,3.5])

                    plt.subplot(312)
                    plt.step(self.hivel[vdx],   self.bigew[vdx],           'g')
                    plt.step(self.civbvel[vdx], self.bigew[vdx]+0.5,       'b')
                    plt.step(self.civrvel[vdx], self.bigew[vdx]+1.0,       'r')
                    plt.step(self.nvbvel[vdx],  self.bigew[vdx]+1.5,       'c')
                    plt.step(self.nvrvel[vdx],  self.bigew[vdx]+2.0,       'm')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 0.5, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.0, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.5, 'k')
                    plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 2.0, 'k')
                    for i in range(vabs.size):
                        plt.plot([vabs[i],vabs[i]],[0.0,4.0])
                    plt.xlim([-5000,5000])
                    plt.ylim([-0.1,2.5])

                    plt.subplot(313)
                    plt.step(velocity,    nsigcopy, 'b')
                    plt.step(velocity, g(velocity), 'r')
#                    plt.step(velocity,   nbot, 'r')
#                    plt.step(velocity, 3*nbot, 'g')
                    plt.xlim([-5000,5000])
                    plt.ylim([-0.1*np.max(nsigcopy),1.1*np.max(nsigcopy)])

                    plt.pause(0.001)

#                    print(vab, np.max(nsigcopy),s,nit,chisq)
                    nit = 1000

            print(vab, np.max(nsigcopy),s,nit,chisq)
            vabs = np.append(vabs, vab)
            nsigcopy -= g(velocity)

        return vabs

    ###############################################################################################
    def calcbigew(self,ew,sew):

#        bigew = np.convolve(ew,self.lsf,mode='same')
#        bigsew = np.sqrt(np.convolve(sew*sew, self.lsf,mode='same'))

        bigew  = np.zeros(ew.size)
        bigsew = np.zeros(ew.size)
        for i in range(ew.size):
            istart = np.int32(np.max([0,i - self.lsf.size//2]))
            iend   = np.int32(np.min([i + self.lsf.size//2,ew.size]))

            jstart = self.lsf.size//2 - (i-istart)
            jend   = self.lsf.size//2 - (i-iend)

            bigew[i]  =         np.sum(           ew[istart:iend] * self.lsf[jstart:jend] )  / np.sum(self.lsf[jstart:jend])
            bigsew[i] = np.sqrt(np.sum(np.square(sew[istart:iend] * self.lsf[jstart:jend]))) / np.sum(self.lsf[jstart:jend])

        return bigew,bigsew

    ###############################################################################################
    def calcew(self,mymodel):
        fit = self.mkmodel(mymodel)

        self.ew     = np.zeros(self.wavelength.size)
        self.sew    = np.zeros(self.wavelength.size)
        self.bigew  = np.zeros(self.wavelength.size)
        self.bigsew = np.zeros(self.wavelength.size) + 1.0e-20

        if self.vdx.size == 0:
            self.nflux           = np.ones(self.wavelength.size)
            self.nferr           = np.ones(self.wavelength.size)
            self.nflux[self.fpx] = np.multiply(self.flux[self.fpx],1.0/fit)
            self.nferr[self.fpx] = 1.0/np.multiply(np.sqrt(self.ivar[self.fpx]),fit)
            self.ew[self.fpx]    = np.multiply(1.0 - self.nflux[self.fpx],self.dwave[self.fpx])
            self.sew[self.fpx]   = np.multiply(self.nferr[self.fpx],self.dwave[self.fpx])
        else:
            self.ew[self.fpx]  = np.multiply(1.0 - np.multiply(self.nflux[self.fpx], 1/fit),self.dwave[self.fpx])
            self.sew[self.fpx] = np.multiply(np.multiply(self.nferr[self.fpx], 1/fit),self.dwave[self.fpx])

        self.bigew[self.fpx]  = convolve(np.squeeze(self.ew[ self.fpx]), self.lsf, boundary='extend')
        self.bigsew[self.fpx] = convolve(np.squeeze(self.sew[self.fpx]), self.lsf, boundary='extend')

        return np.multiply(self.bigew, 1.0/self.bigsew)

    ###############################################################################################
    def dosearch(self,sdss,q,box1,box2,verbose,myatoms,basedir,makeplot=False):
        if verbose: print("Starting AAL search on object ", sdss.qnum[q])
        vdx,nnflux,nnferr,ew,sew,bigew,bigsew = self.normspec(sdss,box1,box2,makeplot=False)

        nsigma = 3

        self.ew[vdx]     = np.copy(ew)
        self.sew[vdx]    = np.copy(sew)
        self.bigew[vdx]  = np.copy(bigew)
        self.bigsew[vdx] = np.copy(bigsew)

        bigew  = np.where(self.bigew < 0, np.zeros(self.bigew.size), self.bigew)
        bigsew = np.copy(self.bigsew)

        self.flux        = np.ones(self.wavelength.size)
        self.flux[vdx]   = nnflux
        self.ivar        = np.ones(self.wavelength.size)
        self.ivar[vdx]   = 1/np.square(nnferr)

        nvdx   = myatoms.getspecies(7,  5)
        civdx  = myatoms.getspecies(6,  4)
        hidx   = myatoms.getspecies(1,  1)
        mgiidx = myatoms.getspecies(12, 2)
        hipdx  = np.extract(myatoms.ip >  54.416, range(myatoms.ip.size))
        hipdx  = np.append(hidx,hipdx)
        lopdx  = np.extract(myatoms.ip <= 54.416, range(myatoms.ip.size))

        vel  = calcvel(self.wavelength,myatoms.wave[ nvdx])
        vel2 = calcvel(self.wavelength,myatoms.wave[ hidx])
        vel3 = calcvel(self.wavelength,myatoms.wave[civdx])
        vel4 = calcvel(self.wavelength,myatoms.wave[mgiidx])
        # Indices where the NV 1238 velocity is larger than the most redward H I velocity and the N V 1242 is less than 3100 km/s. (Why the latter?)
        veldx = np.extract((vel[:,0] > vel2[0,0]) & (vel[:,1] < 3100), range(self.wavelength.size))

        # Redshift space, stack is lox-weight arithmetic mean of bigew
        zhi          = (1.0 + sdss.zqso[q]) * (1.0 + 5000/299792.458) - 1
        dz           =  50/299792.458 # 50 km/s    # np.average(self.dwave/self.wavelength)
        print("Searching redshift range z = 0 - ",zhi," in ",int(zhi/dz)," increments of dz = ",dz," (dv = ",299792.458*dz," km/s)")
        z            = np.linspace(0,zhi,num=int(zhi/dz))

        # Compute the "Big EW Probabilities" for the low-ionization (beplo) and the high ionization (bephi) species. H I counts as high.
        # These are not actual probabilities as there is no normalization. It is just a LOX-weighted velocity stack of the Big EW spectrum
        vlo = -5000
        possiblewaves = np.extract((nnflux/nnferr > nsigma) & (self.wavelength[vdx] > np.min(calcwave(np.array([vlo]),myatoms.wave[hidx]))), self.wavelength[vdx])
        wlo = np.min(possiblewaves)
        wdx = np.extract(self.wavelength >= wlo, range(self.wavelength.size))

        bephi        = np.zeros(z.size)
        sigbephi     = np.zeros(z.size)
        totloxhi     = np.zeros(z.size)
        for i in hipdx:
            wobsqsorf      = (1+z)*myatoms.wave[i]/(1+sdss.zqso[q])
            zdx            = np.extract((wobsqsorf > wlo) & (wobsqsorf < self.wavelength[-1]), range(z.size))
            bephi[zdx]    += np.interp(  wobsqsorf[zdx], self.wavelength,           self.bigew)   * myatoms.lox[i]
            sigbephi[zdx] += np.interp(  wobsqsorf[zdx], self.wavelength, np.square(self.bigsew)) * myatoms.lox[i] * myatoms.lox[i]
            totloxhi[zdx] += np.broadcast_to(myatoms.lox[i], (zdx.size))
        tdx = np.extract(totloxhi > 0, range(totloxhi.size))
        bephi[tdx]    = bephi[tdx]/totloxhi[tdx]
        sigbephi[tdx] = np.sqrt(sigbephi[tdx])/totloxhi[tdx]

        beplo        = np.zeros(z.size)
        sigbeplo     = np.zeros(z.size)
        totloxlo     = np.zeros(z.size)
        for i in lopdx:
            wobsqsorf      = (1+z)*myatoms.wave[i]/(1+sdss.zqso[q])
            zdx            = np.extract((wobsqsorf > wlo) & (wobsqsorf < self.wavelength[-1]), range(z.size))
            beplo[zdx]    += np.interp(  wobsqsorf[zdx], self.wavelength,           self.bigew)   * myatoms.lox[i]
            sigbeplo[zdx] += np.interp(  wobsqsorf[zdx], self.wavelength, np.square(self.bigsew)) * myatoms.lox[i] * myatoms.lox[i]
            totloxlo[zdx] += np.broadcast_to(myatoms.lox[i], (zdx.size))
        tdx = np.extract(totloxlo > 0, range(totloxlo.size))
        beplo[tdx]    = beplo[tdx]/totloxlo[tdx]
        sigbeplo[tdx] = np.sqrt(sigbeplo[tdx])/totloxlo[tdx]

        alsys = self.findsys(z,bephi,sigbephi,wdx,nsigma,sdss.zqso[q],myatoms)
        alsys.mergelist(self.findsys(z,beplo,sigbeplo,wdx,nsigma,sdss.zqso[q],myatoms))



        if alsys.nabs > 0:
            alsys.filtersys()

            with open(basedir+'/Vabs/' + sdss.fileroot[q] + '.vabs',"w") as f:
                print("         vabs           vlo           vhi    bepm    beps    nsig nvbew nvbsew nvrew nvrsew  hiew hisew civbew civbsew civrew civrsew mgiibew mgiibsew mgiirew mgiirsew")
                for i in range(alsys.nabs):
                    dumstr  = f"{alsys.vabs[i]:13.3f} {alsys.vlo[i]:13.3f} {alsys.vhi[i]:13.3f} {alsys.bepm[i]:7.5f} {alsys.beps[i]:7.5f} {alsys.nsig[i]:7.3f}"
##                    dumstr += f" {alsys.nvbew[i]:5.0f} {alsys.nvbsew[i]:6.0f} {alsys.nvrew[i]:5.0f} {alsys.nvrsew[i]:6.0f}"
##                    dumstr += f" {alsys.hiew[i]:5.0f} {alsys.hisew[i]:5.0f}"
##                    dumstr += f" {alsys.civbew[i]:6.0f} {alsys.civbsew[i]:7.0f} {alsys.civrew[i]:6.0f} {alsys.civrsew[i]:7.0f}"
##                    dumstr += f" {alsys.mgiibew[i]:7.0f} {alsys.mgiibsew[i]:8.0f} {alsys.mgiirew[i]:7.0f} {alsys.mgiirsew[i]:8.0f}"
                    dumstr += f" {alsys.ew[i,alsys.nvdx[0]]:5.0f} {alsys.sew[i,alsys.nvdx[0]]:6.0f} {alsys.ew[i,alsys.nvdx[1]]:5.0f} {alsys.sew[i,alsys.nvdx[1]]:6.0f}"
                    dumstr += f" {alsys.ew[i,alsys.hidx[0]]:5.0f} {alsys.sew[i,alsys.hidx[0]]:6.0f}"
                    dumstr += f" {alsys.ew[i,alsys.civdx[0]]:5.0f} {alsys.sew[i,alsys.civdx[0]]:6.0f} {alsys.ew[i,alsys.civdx[1]]:5.0f} {alsys.sew[i,alsys.civdx[1]]:6.0f}"
                    dumstr += f" {alsys.ew[i,alsys.mgiidx[0]]:5.0f} {alsys.sew[i,alsys.mgiidx[0]]:6.0f} {alsys.ew[i,alsys.mgiidx[1]]:5.0f} {alsys.sew[i,alsys.mgiidx[1]]:6.0f}"
                    print(dumstr)
                    f.write(dumstr+"\n")

        model = fitmodel(np.array([]),np.array([]),np.array([]),np.array([]),alsys.nabs,alsys.vabs,np.array([]),np.array([]),np.array([]),myatoms)
        sf    = np.sort(self.flux[veldx])
        bbb = np.sort(self.bigew)

        if makeplot:
            lstr = np.array(['c--','m--','y--','r--'])
            vzz = calcvelzz(z,sdss.zqso[q])
            plt.clf()
            # Low ionization systems
            plt.subplot(311)
            vlim = np.array([np.min(vel4)-3000,np.min([3000,np.max(vel4)])])
            plt.fill_between(vzz, -nsigma*sigbeplo, nsigma*sigbeplo, color='c')
            plt.fill_between(vzz,        -sigbeplo,        sigbeplo, color='y')
            for i in lopdx:
                plt.step(calcvel(self.wavelength[wdx],myatoms.wave[i]),self.bigew[wdx])
            plt.step(        vzz,           beplo,                  'm',   linewidth=3.0)
            for i in range(alsys.nabs):
                plt.plot([alsys.vabs[i],alsys.vabs[i]],[bbb[0],bbb[-1]],lstr[i % lstr.size], linewidth=3.0)
            plt.plot(vlim,[0,0],"k")
            plt.xlim(vlim)
            plt.ylim([np.max(np.extract((vzz > vlim[0]) & (vzz < vlim[-1]), beplo)),
                      np.min(np.extract((vzz > vlim[0]) & (vzz < vlim[-1]), beplo))])
            if alsys.nabs > 0:
                plt.title(("qnum {}: Max at v = {:.2f} with bep = {}").format(sdss.qnum[q],alsys.vabs[0],alsys.bepm[0]))
            else:
                plt.title(("qnum {}").format(sdss.qnum[q]))

            # High ionization systems
            plt.subplot(312)
            vlim = np.array([np.min([vel,vel3])-3000,3000])
            plt.fill_between(vzz, -nsigma*sigbephi, nsigma*sigbephi, color='c')
            plt.fill_between(vzz,        -sigbephi,        sigbephi, color='y')
            for i in hipdx:
                plt.step(calcvel(self.wavelength[wdx],myatoms.wave[i]),self.bigew[wdx])
            plt.step(vzz,bephi, 'm', linewidth=3.0)
            for i in range(alsys.nabs):
                plt.plot([alsys.vabs[i],alsys.vabs[i]],[bbb[0],bbb[-1]],lstr[i % lstr.size], linewidth=3.0)
            plt.plot(vlim,[0,0],"k")
            plt.xlim(vlim)
            plt.ylim([np.max(np.extract((vzz > vlim[0]) & (vzz < vlim[-1]), bephi)),
                      np.min(np.extract((vzz > vlim[0]) & (vzz < vlim[-1]), bephi))])

            plt.subplot(313)
            wdx = np.extract(self.bigew > 0, self.wavelength)
            wlim = np.array([wdx[0],wdx[-1]])
            plt.step(self.wavelength,self.flux, 'b')
            self.markspec(fitmodel(np.array([]),np.array([]),np.array([]),np.array([]),alsys.nabs,alsys.vabs,np.array([]),np.array([]),np.array([]),myatoms),0.8*sf[-1],wlim[0],wlim[1])
            plt.plot(wlim,[1,1],"k")
            plt.xlim(wlim)
            plt.ylim([0.9*sf[0],1.2*sf[-1]])

            plt.pause(0.1)

##        return (np.extract(alsys.civbew > 3*alsys.civbsew, alsys.civbew)).size, (np.extract(alsys.nvbew > 3*alsys.nvbsew, alsys.nvbew)).size, (np.extract((alsys.nvbew > 3*alsys.nvbsew) & (alsys.civbew > 3*alsys.civbsew), alsys.nvbew)).size
        return (np.extract(alsys.ew[:,alsys.civdx[0]] > 3*alsys.sew[:,alsys.civdx[0]], alsys.ew[:,alsys.civdx[0]])).size, (np.extract(alsys.ew[:,alsys.nvdx[0]] > 3*alsys.sew[:,alsys.nvdx[0]], alsys.ew[:,alsys.nvdx[0]])).size, (np.extract((alsys.ew[:,alsys.nvdx[0]] > 3*alsys.sew[:,alsys.nvdx[0]]) & (alsys.ew[:,alsys.civdx[0]] > 3*alsys.sew[:,alsys.civdx[0]]), alsys.ew[:,alsys.nvdx[0]])).size

    ###############################################################################################
    def findsys(self,z, bep,sigbep,wdx,nsigma,zqso,myatoms):

        nvdx   = myatoms.getspecies( 7, 5)
        civdx  = myatoms.getspecies( 6, 4)
        hidx   = myatoms.getspecies( 1, 1)
        mgiidx = myatoms.getspecies(12, 2)

##        vel  = calcvel(self.wavelength[wdx],myatoms.wave[  nvdx])
##        vel2 = calcvel(self.wavelength[wdx],myatoms.wave[  hidx])
##        vel3 = calcvel(self.wavelength[wdx],myatoms.wave[ civdx])
##        vel4 = calcvel(self.wavelength[wdx],myatoms.wave[mgiidx])
        vel = np.empty([wdx.size,myatoms.species.size])
        for i in range(myatoms.species.size):
             vel[:,i] = np.squeeze(calcvel(self.wavelength[wdx],myatoms.wave[i]))

        print("self.wavelength.shape = ",self.wavelength.shape)
        print("self.ew.shape = ",self.ew.shape)
        print("wdx.shape = ", wdx.shape)

##        alsys = abslinesys()
        alsys = abslinesys(myatoms)
        bepc     = np.copy(bep)
        done = 0
        while done == 0:
            nn = np.where(np.isfinite(bepc/sigbep), bepc/sigbep, 0)
            idx = np.squeeze(np.extract(nn == np.max(nn), range(bepc.size)))
            if idx.size > 1:
                idx = idx[0]
            if (nn[idx] < nsigma) or (bepc[idx] == 0):
                done = 1
            else:
                ilo = np.copy(idx)
                while (bepc[ilo] > nsigma * sigbep[ilo]) and (ilo > 0):
                    ilo -= 1
                ihi = np.copy(idx)
                while (bepc[ihi] > nsigma * sigbep[ihi]) and (ihi < bepc.size-1):
                    ihi += 1
                idx = np.squeeze(np.extract(bepc[ilo:ihi] == np.max(bepc[ilo:ihi]), np.arange(ilo,ihi,dtype=np.int32)))

                if bepc[idx] > nsigma*sigbep[idx]:
                    vlo      = calcvelzz(z[ilo],zqso)
                    vhi      = calcvelzz(z[ihi],zqso)
                    alsys.addsystem(           calcvelzz(z[idx],zqso),
                                    vlo      = vlo,
                                    vhi      = vhi,
                                    nsig     = bepc[idx]/sigbep[idx],
                                    bepm     = bepc[idx],
                                    beps     = sigbep[idx],
                                    ew       = 1000 * np.sum(np.where((vel >= vlo) & (vel <= vhi),
                                                                      np.transpose(np.broadcast_to(self.ew[wdx],
                                                                                                   np.transpose(vel).shape)),
                                                                      np.zeros(vel.shape)),
                                                             axis=0),
                                    sew      = 1000 * np.sqrt(np.sum(np.where((vel >= vlo) & (vel <= vhi),
                                                                              np.transpose(np.broadcast_to(np.square(self.sew[wdx]),
                                                                                                           np.transpose(vel).shape)),
                                                                              np.zeros(vel.shape)),
                                                                     axis=0)))
##                                    nvbew    = 1000 * np.sum(np.extract((vel[ :,0] >= vlo) & (vel[ :,0] <= vhi), self.ew)),
##                                    nvrew    = 1000 * np.sum(np.extract((vel[ :,1] >= vlo) & (vel[ :,1] <= vhi), self.ew)),
##                                    hiew     = 1000 * np.sum(np.extract((vel2[:,0] >= vlo) & (vel2[:,0] <= vhi), self.ew)),
##                                    civbew   = 1000 * np.sum(np.extract((vel3[:,0] >= vlo) & (vel3[:,0] <= vhi), self.ew)),
##                                    civrew   = 1000 * np.sum(np.extract((vel3[:,1] >= vlo) & (vel3[:,1] <= vhi), self.ew)),
##                                    mgiibew  = 1000 * np.sum(np.extract((vel4[:,0] >= vlo) & (vel4[:,0] <= vhi), self.ew)),
##                                    mgiirew  = 1000 * np.sum(np.extract((vel4[:,1] >= vlo) & (vel4[:,1] <= vhi), self.ew)),
##                                    nvbsew   = 1000 * np.sqrt(np.sum(np.extract((vel[ :,0] >= vlo) & (vel[ :,0] <= vhi), np.square(self.sew)))),
##                                    nvrsew   = 1000 * np.sqrt(np.sum(np.extract((vel[ :,1] >= vlo) & (vel[ :,1] <= vhi), np.square(self.sew)))),
##                                    hisew    = 1000 * np.sqrt(np.sum(np.extract((vel2[:,0] >= vlo) & (vel2[:,0] <= vhi), np.square(self.sew)))),
##                                    civbsew  = 1000 * np.sqrt(np.sum(np.extract((vel3[:,0] >= vlo) & (vel3[:,0] <= vhi), np.square(self.sew)))),
##                                    civrsew  = 1000 * np.sqrt(np.sum(np.extract((vel3[:,1] >= vlo) & (vel3[:,1] <= vhi), np.square(self.sew)))),
##                                    mgiibsew = 1000 * np.sqrt(np.sum(np.extract((vel4[:,0] >= vlo) & (vel4[:,0] <= vhi), np.square(self.sew)))),
##                                    mgiirsew = 1000 * np.sqrt(np.sum(np.extract((vel4[:,1] >= vlo) & (vel4[:,1] <= vhi), np.square(self.sew)))))

                bepc[ilo:ihi+1] = 0

        return alsys

    ###############################################################################################
    def fitspec(self,sdss,box1,box2,absonly,myatoms):

        self.fittime = time.time()
        vdx,nnflux,nnferr,ew,sew,bigew,bigsew = self.normspec(sdss,box1,box2)

        self.nflux = np.ones(self.wavelength.size)
        self.nferr = np.ones(self.wavelength.size)

        self.nflux[vdx] = np.copy(nnflux)
        self.nferr[vdx] = np.copy(nnferr)
        self.vdx        = np.copy(vdx)

        if absonly == False:
            W = sdss.transformspec(self)
            nmfflux = sdss.calcspec(W)

            # Determine the initial Legendre order for the self.fpx region
            # First smooth the specrtum and then fit the smoothed spectrum for the initial guess
            smfac   = int(np.round(nsmth * (2803.5315 - 2796.3543)/np.average(self.dwave[self.fpx]), decimals=0))
            if smfac % 2 == 0:
                smfac += 1
            smflux           = convolve(self.flux[self.fpx], np.ones(smfac)/smfac, boundary='extend')
            degree           = 3
            r1               = np.interp(self.wavelength[self.fpx], sdss.wavebins, nmfflux)
            r                = np.polynomial.legendre.legfit(self.legfitrange, smflux-rnh1, degree, w=np.sqrt(self.ivar[self.fpx]), full=True)
            r2               = np.polynomial.legendre.legval(self.legfitrange, rnh[0])
            chisqnu          = np.sum(np.multiply(np.square(r1 + r2 - smflux), self.ivar[self.fpx])) / self.fpx.size
            nsigma           = self.calcew(fitmodel(sdss.wavebins, sdss.H, W, np.array([]), 0, np.array([]), np.array([]), np.array([]), np.array([]), myatoms))
            hisign           = np.extract(nsigma[self.fpx] < 0.0, np.arange(self.fpx.size))
            oldn             = 9999999.9
            while (np.mean(np.fabs(nsigma[hisign])) > np.min([oldn,0.6])) and (np.fabs(oldn-np.mean(np.fabs(nsigma[hisign]))) > 0.1):
                oldn    = np.mean(np.fabs(nsigma[hisign]))
                degree += 1
                r     = np.polynomial.legendre.legfit(self.legfitrange, smflux-r1, degree, w=np.sqrt(self.ivar[self.fpx]), full=True)
                r2    = np.polynomial.legendre.legval(self.legfitrange, r[0])
                chisqnu = np.sum(np.multiply(np.square(r1 + r2 - smflux), self.ivar[self.fpx])) / self.fpx.size
                plt.clf()
                plt.subplot(211)
                plt.step(self.nvbvel[self.fpx], self.flux[self.fpx], 'k')
                plt.step(self.nvbvel[self.fpx], smflux,              'b')
                plt.step(self.nvbvel[self.fpx], r1,                  'r')
                plt.step(self.nvbvel[self.fpx], r2,                  'g')
                plt.step(self.nvbvel[self.fpx], r1+r2,               'm')
                plt.subplot(212)
                plt.step(self.nvbvel[self.fpx], smflux-r1,           'b')
                plt.step(self.nvbvel[self.fpx], r2,                  'g')
                plt.pause(0.1)
                nsigma = self.calcew(fitmodel(sdss.wavebins, sdss.H, W, r[0], 0, np.array([]), np.array([]), np.array([]), np.array([]), myatoms))
                hisign = np.extract(nsigma[self.fpx] < 0.0, np.arange(self.fpx.size))
                print(degree, np.mean(np.fabs(nsigma[hisign])), chisqnu, r[0])
            print(degree, np.mean(np.fabs(nsigma[hisign])), chisqnu, r[0])


        # Make the initial model with the NMF projections and superposed Legendre polynomials
        if absonly == False:
            mymodel    = fitmodel(sdss.wavebins, sdss.H, W, r[0], 0, np.array([]), np.array([]), np.array([]), np.array([]), myatoms)
        else:
            mymodel    = fitmodel(np.array([]), np.array([]), np.array([]), np.array([]), 0, np.array([]), np.array([]), np.array([]), np.array([]), myatoms)
        x          = mymodel.packmod()
        if absonly == False:
            xsq        = chisqfunc(x, sdss.wavebins, sdss.H, mymodel, self) # Initialzed the Chi^2 function class
        else:
            xsq        = chisqfunc(x, np.array([]), np.array([]), mymodel, self) # Initialzed the Chi^2 function class
        chisq      = xsq(x)
        abswavefilter = np.ones(self.fpx.size) # Initial array to keep track of where we have placed absorbers
        done    = 0
        while done == 0:
            fit       = self.mkmodel(mymodel)

            oldchisq  = chisq
            npar      = mymodel.w.size + mymodel.legorder + mymodel.vabs.size + mymodel.bhis.size + (np.extract(mymodel.bnts > 0, np.arange(mymodel.bnts.size))).size + (np.extract(mymodel.tabs > 0, np.arange(mymodel.tabs.size))).size
            print('\n\n----------------------------------------------')
            print('Initial model:')
            mymodel.printmod()
            print("\n")
            self.pltspec(fit, mymodel)

            x        = mymodel.packmod()
            if mymodel.absonly == False:
                xsq  = chisqfunc(x, sdss.wavebins, sdss.H, mymodel, self)
                xsqh = chisqfunc(x, sdss.wavebins, sdss.H, mymodel, self, 10000)
            else:
                xsq  = chisqfunc(x, np.array([]), np.array([]), mymodel, self)
                xsqh = chisqfunc(x, np.array([]), np.array([]), mymodel, self, 10000)
            xsqjac   = xsq.jacob
            hfunc    = SR1()
            if (mymodel.absonly == False) or (mymodel.nabs > 0):
                res = minimize(xsq, x, method='trust-constr', jac=xsqjac, hess=hfunc, bounds = mymodel.bind(x,np.max(self.flux[self.fpx]),self.waveall[0],self.waveall[-1]),
                               tol=1.0e-3, options={'gtol': 1.0e-3, 'verbose': 1, 'initial_tr_radius': 10})

                mymodel.unpackmod(res.x)
                mymodel.printmod()

                chisq = res.fun
                if mymodel.nabs > 0:
                    print('----------------------------------------------')
                    print('Computing Hessian to get uncertainties...\n')
                    t0   = time.time()
                    x2   = np.where(np.fabs(res.x) < 1.0e-3, np.ones(res.x.size), res.x)
                    x2[-mymodel.atoms.nion*mymodel.nabs:] = np.where(x2[-mymodel.atoms.nion*mymodel.nabs:] < 1.0e-3,
                                                                       np.ones(mymodel.atoms.nion*mymodel.nabs), x2[-mymodel.atoms.nion*mymodel.nabs:])
                    xdx  = np.extract(x2 != 1, range(x2.size)) # Flag where there are good values
                    x2dx = np.extract(x2 == 1, range(x2.size)) # Flag where there are fixed values
                    H    = hfunc.get_matrix()
                    for xx in x2dx:
                        H[xx,:] = 0
                        H[:,xx] = 0
                    print('Hessian took ',time.time()-t0,' seconds to compute...\n')
                    # Before inverting, we want to mask out the rows/columns of fixed values
                    Hcopy    = H[  ~np.all(H   == 0, axis=1)]
                    Hct      = np.transpose(Hcopy)
                    Hct2     = Hct[~np.all(Hct == 0, axis=1)]
                    Hcopy2   = np.transpose(Hct2)
                    C       = np.linalg.inv(Hcopy2)
                    sp      = np.zeros(res.x.size)
                    sp[xdx] = np.sqrt(np.fabs(np.diag(C)))

                    mymodel.unpackerr(sp)

                print('----------------------------------------------\n\n')
                print('Minimum Chisq model:')
                print(('{:d} tol = {:.3f} chisq = {:.3f} (npar = {:d} chisq/dof = {:.5f})').format(res.nit, 1.0e-3, chisq,npar, chisq/(self.fpx.size-npar),')'))
                mymodel.printmod(unc=True)

                # Clean up on bad absorbers
                (sl, sb) = mymodel.bind(res.x,np.max(self.flux[self.fpx]),self.waveall[0],self.waveall[-1]).residual(res.x)
                sldx = np.extract(sl < 0, range(sl.size))
                sbdx = np.extract(sb < 0, range(sb.size))
                b  = np.sqrt(np.square(mymodel.bhis) + np.square(mymodel.bnts))
                sigmabob = np.sqrt(np.square(mymodel.bhis * mymodel.sigbhis) + np.square(mymodel.bnts * mymodel.sigbnts)) / np.square(b)
                badness = mymodel.sigvabs/np.fabs(mymodel.vabs) + sigmabob + np.min(mymodel.sigtabs/(mymodel.tabs+1.0e-10), axis=0)
                rmabs = np.where(badness > 1.5, np.ones(mymodel.nabs), np.zeros(mymodel.nabs))
                for i in range(mymodel.nabs):
                    for j in range(mymodel.atoms.nion):
                        if 1.0-np.exp(-mymodel.tabs[j,i]) < np.min(1.0/self.snr[self.fpx]): # Buried in the noise...
                            mymodel.tabs[j,i] = 0.0
                        gsdx = mymodel.atoms.getspecies(mymodel.atoms.anum[mymodel.atoms.idx[j]], mymodel.atoms.ion[mymodel.atoms.idx[j]])
                        obswave = calcwave(mymodel.vabs[i],mymodel.atoms.wave[gsdx])
                        if np.any((obswave > self.waveall[0]) & (obswave < self.waveall[-1])) == False:
                            mymodel.tabs[j,i] = 0.0
                    if np.sum(mymodel.tabs[:,i]) < 0.1: #If optical depths are too low (or negative) then get rid of them
                        rmabs[i] = 1

                if np.sum(rmabs) > 0:
                    rdx = np.extract(rmabs == 1, range(mymodel.nabs))
                    print("Removing absorbers: ")
                    for r in rdx:
                        print(("{:d} with badness {:.3f} = {:.3f} + {:.3f} + {:.3f}").format(r,badness[r], mymodel.sigvabs[r]/np.fabs(mymodel.vabs[r]), sigmabob[r], np.min(mymodel.sigtabs[:,r]/(mymodel.tabs[:,r]+1.0e-10))))
                    mymodel.removeabs(rmabs)

                # Clean up on bad continuum parameters
                if mymodel.absonly == False:
                    mymodel.a = np.trim_zeros(mymodel.a, trim='b')

                fit = self.mkmodel(mymodel)
                self.pltspec(fit,mymodel)

            # Are we done? Or do we have some absorption left to fit?
            nsigma = self.calcew(mymodel)
            nsigma = nsigma[self.fpx] # nsigma has the same size as self.fpx
            losig  = np.extract((nsigma > 0.0) & (abswavefilter > 0), np.arange(nsigma.size)) # Absorption
            if mymodel.absonly == False:
                hisig = np.extract(nsigma < 0.0, np.arange(nsigma.size)) # Continuum
                refpix = (np.extract(nsigma == np.max(nsigma), np.arange(nsigma.size)))[0]
            else:
                refpix = (np.extract(nsigma[losig] == np.max(nsigma[losig]), losig))[0]
##            lopix  = np.copy(refpix)-1
##            while self.bigew[lopix] > self.bigew[lopix+1] and lopix > 0:
##                lopix -= 1
##            hipix  = np.copy(refpix)+1
##            while self.bigew[hipix] > self.bigew[hipix-1] and hipix < self.bigew.size:
##                hipix += 1
##            sigv = np.squeeze(calcvel(self.wavelength[hipix],self.wavelength[lopix])/2.35)
##            if sigv < 0:
##                sigv = 20
            sigv = self.dvres
            done = 1 # Assume we're done unless proven otherwise
            if losig.size > 0:
                print("Absorption side: ", round(np.max(nsigma[losig]),4))
            else:
                print("Absorption side: ")
            if mymodel.absonly == False:
                if hisig.size > 0:
                    print("Continuum side: ", round(np.mean(np.fabs(nsigma[hisig])),4))
                else:
                    print("Continuum side: ")
            if nsigma[refpix] > 2.4:
                if (np.max(nsigma[losig]) > 0.4769362762044698733814183536431305598089697490594706447038): # and (np.max(nsigma[losig]) >= np.min([np.mean(np.fabs(nsigma[hisign])),np.mean(np.fabs(nsigma[hisigc]))])): #erf^-1(0.5)
                    done = 0
                    print(("Adding absorption... ({:.3f} @ {:.3f} abswavefilter[{:d}] = {:d})").format(nsigma[refpix],self.wavelength[self.fpx[refpix]],refpix,int(abswavefilter[refpix])))
                    self.msigma = nsigma[refpix]
                    self.msigmawave = self.wavelength[self.fpx[refpix]]
                    abswavefilter[refpix] = 0
                    # Need a lox-based likelyhood of the identity of this absorption feature...
                    # What is the velocity of the absorber assuming the identify of each myatoms.wave
                    velarray = np.squeeze(np.transpose(calcvel(self.wavelength[self.fpx[refpix]], mymodel.atoms.wave))) # This should be a (mymodel.atoms.wave.size,) shape
                    velarray = np.extract(velarray < 5000, velarray)
                    # For each of those velocities, at what wavelength do all of the other transitions appear?
                    wavearray = calcwave(velarray, mymodel.atoms.wave) # This should be a (velarray.size,mymodel.atoms.wave.size) shape
                    probarray = np.zeros(velarray.size)
                    for v in range(velarray.size):
                        gdx = np.extract((wavearray[v,] > self.waveall[0]) & (wavearray[v,] < self.waveall[-1]), range(mymodel.atoms.wave.size))
                        if gdx.size > 0:
                            gdsig = np.interp(wavearray[v,gdx], self.waveall, nsigma)
                            probarray[v] = np.sum(mymodel.atoms.lox[gdx] * gdsig) # Total guess as to whether this is a good metric...
                    newvelocity = np.squeeze(np.extract(probarray == np.max(probarray), velarray)) # Assuming that probarray actually worked...
                    newwaves    = np.squeeze(calcwave(newvelocity,mymodel.atoms.wave))
                    newtabs     = np.zeros(mymodel.atoms.nion)
                    newtabslim  = np.zeros(mymodel.atoms.nion)
                    for n in range(mymodel.atoms.nion):
                        gsdx       = mymodel.atoms.getspecies(mymodel.atoms.anum[mymodel.atoms.idx[n]], mymodel.atoms.ion[mymodel.atoms.idx[n]])
                        gscdx      = np.extract((newwaves[gsdx] > self.waveall[0]) & (newwaves[gsdx] < self.waveall[-1]), gsdx)

                        if gscdx.size > 0:
                            gsldx         = np.extract(mymodel.atoms.lox[gscdx] == np.max(mymodel.atoms.lox[gscdx]), gscdx)
                            newtabs[n]    = -np.log(np.interp(newwaves[gsldx[0]], self.waveall,   self.nflux[self.fpx]))
                            newtabslim[n] = -np.log(np.interp(newwaves[gsldx[0]], self.waveall, 1-self.nferr[self.fpx]))
                    newtabs = np.where(newtabs < newtabslim, np.zeros(mymodel.atoms.nion), newtabs)
                    newbnt = 0.0
                    if (np.extract(newtabs > 0, newtabs)).size > 1:
                        newbnt = 5.0
                    mymodel.addabs(newvelocity, sigv, newbnt, newtabs)

            if mymodel.absonly == False:
                if (np.mean(np.fabs(nsigma[hisig])) > 0.4769362762044698733814183536431305598089697490594706447038) and (mymodel.legorder == mymodel.a.size): #erf^-1(0.5)
                    print("Increasing Legendre order")
                    mymodel.legorder += 5
                    mymodel.a = np.append(mymodel.a, np.zeros(5))
                    done = 0
                else:
                    mymodel.legorder = mymodel.a.size

        return mymodel

    ###############################################################################################
    def getvlims(self,vab,varray):
        ldx = np.min(np.extract(varray > vab, range(varray.size)))
        while self.ew[ldx] > 0:
            ldx -= 1
        hdx = np.max(np.extract(varray < vab, range(varray.size)))
        while self.ew[hdx] > 0:
            hdx += 1

        return ldx,hdx

    ###############################################################################################
    def linelist(self,vabs,sdss,box1,box2):
        vdx,nnflux,nnferr = self.normspec(sdss,box1,box2)

        self.ew[vdx]     = np.multiply(1.0 - nnflux, self.dwave[vdx])
        self.sew[vdx]    = np.multiply(      nnferr, self.dwave[vdx])

        tnv  = np.ones(vabs.size)
        tciv = np.ones(vabs.size)
        thi  = np.ones(vabs.size)
        kT   = 1.0e+4 * np.ones(vabs.size)
        bnt  = 10 * np.ones(vabs.size)
        modtau = np.zeros(vdx.size)
        for i in range(vabs.size):
            modtau += self.tauNV(self.wavelength[vdx],tnv[i],np.sqrt(kT[i]/14+bnt[i]*bnt[i]),vabs[i]) + self.tauHI(self.wavelength[vdx],thi[i],np.sqrt(kT[i]+bnt[i]*bnt[i]),vabs[i]) + self.tauCIV(self.wavelength[vdx],tciv[i],np.sqrt(kT[i]/12+bnt[i]*bnt[i]),vabs[i])
        modflux = np.exp(-modtau)
        chisq   = np.sum(np.square(np.multiply(nnflux-modflux, 1/nnferr)))
        nit  = 10000
        nprt = 1
        scale = 0.01
        while nit > 0:
            oldtnv   = np.copy(tnv)
            oldtciv  = np.copy(tciv)
            oldthi   = np.copy(thi)
            oldbnt   = np.copy(bnt)
            oldkT    = np.copy(kT)
            oldvabs  = np.copy(vabs)
            oldchisq = chisq

            vabs = np.multiply(vabs, np.random.uniform(low=1-scale/100,high=1+scale/100,size=vabs.size))
            tnv  = np.multiply(tnv,  np.random.uniform(low=1-scale,high=1+scale,size=vabs.size))
            tciv = np.multiply(tciv, np.random.uniform(low=1-scale,high=1+scale,size=vabs.size))
            thi  = np.multiply(thi,  np.random.uniform(low=1-scale,high=1+scale,size=vabs.size))
            kT   = np.multiply(kT,   np.random.uniform(low=1-scale,high=1+scale,size=vabs.size))
            bnt  = np.multiply(bnt,  np.random.uniform(low=1-scale,high=1+scale,size=vabs.size))

            modtau = np.zeros(vdx.size)
            for i in range(vabs.size):
                modtau += self.tauNV(self.wavelength[vdx],tnv[i],np.sqrt(kT[i]/14+bnt[i]*bnt[i]),vabs[i]) + self.tauHI(self.wavelength[vdx],thi[i],np.sqrt(kT[i]+bnt[i]*bnt[i]),vabs[i]) + self.tauCIV(self.wavelength[vdx],tciv[i],np.sqrt(kT[i]/12+bnt[i]*bnt[i]),vabs[i])
            modflux = np.exp(-modtau)
            chisq   = np.sum(np.square(np.multiply(nnflux-modflux, 1/nnferr)))

            if chisq > oldchisq:
                nit -= 1
                tnv   = np.copy(oldtnv)
                tciv  = np.copy(oldtciv)
                thi   = np.copy(oldthi)
                kT    = np.copy(oldkT)
                bnt   = np.copy(oldbnt)
                vabs  = np.copy(oldvabs)
                chisq = oldchisq
                if (nit % 100 == 0):
                    scale = np.max([np.square(scale),0.1])
#                    print(("{:10.2f} {} {}").format(chisq,nit,scale))
            else:
                plt.clf()
                plt.step(self.hivel[vdx],   nnflux,           'g')
                plt.step(self.civbvel[vdx], nnflux+0.5,       'b')
                plt.step(self.civrvel[vdx], nnflux+1.0,       'r')
                plt.step(self.nvbvel[vdx],  nnflux+1.5,       'c')
                plt.step(self.nvrvel[vdx],  nnflux+2.0,       'm')

                plt.plot(self.hivel[vdx],   modflux,          'y')
                plt.plot(self.civbvel[vdx], modflux+0.5,      'y')
                plt.plot(self.civrvel[vdx], modflux+1.0,      'y')
                plt.plot(self.nvbvel[vdx],  modflux+1.5,      'y')
                plt.plot(self.nvrvel[vdx],  modflux+2.0,      'y')

                plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.0, 'k')
                plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 1.5, 'k')
                plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 2.0, 'k')
                plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 2.5, 'k')
                plt.plot(self.nvbvel[vdx],  nnflux*0.0 + 3.0, 'k')
                for i in range(vabs.size):
                    plt.plot([vabs[i],vabs[i]],[0.0,4.0])
                plt.xlim([-5000,5000])
                plt.ylim([-0.1,3.5])
                plt.pause(0.0001)
                nprt += 1
                if nprt % 1 ==0:
                    print(("{:10.2f} {} {}").format(chisq,nit,scale))
                    for i in range(vabs.size):
                        print(("       {:10.2f} {:10.6f} {:10.6f} {:10.6f} {:10.0f} {:10.2f}").format(vabs[i],thi[i],tciv[i],tnv[i],kT[i],bnt[i]))
                nit = 10000
                scale = np.max([np.square(scale),0.1])

#        mymodel = fitmodel(sdss.wavebins, sdss.H, W, W, [0], [0], vabs.size, vabs, np.sqrt(kT), bnt, tnv, thi, tciv, np.zeros(vabs.size))



        for i in range(vabs.size):
            b       = np.sqrt(kT[i]/14 + bnt[i] * bnt[i])
            nvbdvel = calcvel(np.linspace(1220,1250,num=600), 1238.821) / b
            nvbew   = np.sum((1 - np.exp(- tnv[i] * (np.exp(-np.square(nvbdvel)))))*0.05)

            b        = np.sqrt(kT[i]/12 + bnt[i] * bnt[i])
            civbdvel = calcvel(np.linspace(1530,1560,num=600), 1548.195) / b
            civbew   = np.sum((1 - np.exp(- tciv[i] * (np.exp(-np.square(civbdvel)))))*0.05)

            b      = np.sqrt(kT[i] + bnt[i] * bnt[i])
            hidvel = calcvel(np.linspace(1200,1230,num=600), 1215.670) / b
            hiew   = np.sum((1 - np.exp(- tnv[i] * (np.exp(-np.square(hidvel)))))*0.05)

            print(("{:10.2f} {:10.6f} {:10.6f} {:10.6}").format(vabs[i],nvbew,civbew,hiew))


    ###############################################################################################
    def markspec(self,model,mflux,wlo,whi):
        obswave = calcwave(model.vabs,model.atoms.wave)
        for i in range(model.nabs):
            idx = np.extract((obswave[i,] > wlo) & (obswave[i,] < whi), range(model.atoms.wave.size))
            if idx.size > 0:
                for j in idx:
                    lstr = 'ok'
                    if (model.atoms.anum[j] == 6): # C  - red
                        lstr = 'or'
                    elif(model.atoms.anum[j] == 7): # N - blue
                        lstr = 'ob'
                    elif(model.atoms.anum[j] == 12): # Mg - magenta
                        lstr = 'om'
                    elif(model.atoms.anum[j] == 13): # Al - cyan
                        lstr = 'oc'
                    elif(model.atoms.anum[j] == 14): # Si - green
                        lstr = 'og'
                    elif(model.atoms.anum[j] == 16): # S - yellow
                        lstr = 'oy'
                    plt.plot([obswave[i,j]], [mflux * (1.05 + i * 0.25/model.nabs)], lstr)
#                    if (model.atoms.anum[j] == 6)   and (model.atoms.ion[j] == 4): # C IV doublet - red
#                        lstr = 'or'
#                    elif(model.atoms.anum[j] == 7)  and (model.atoms.ion[j] == 5): # N V doublet - blue
#                        lstr = 'ob'
#                    elif(model.atoms.anum[j] == 12) and (model.atoms.ion[j] == 2): # Mg II doublet - magenta
#                        lstr = 'om'
#                    elif(model.atoms.anum[j] == 13) and (model.atoms.ion[j] == 3): # Al III doublet - cyan
#                        lstr = 'oc'
#                    elif(model.atoms.anum[j] == 14) and (model.atoms.ion[j] == 4): # Si IV doublet - green
#                        lstr = 'og'
#                    plt.plot([obswave[i,j],      obswave[i,j]],           [mflux * (1.05 + (i-1)   * 0.25/model.nabs), mflux * (1.05 +  i      * 0.25/model.nabs)], lstr)
#                plt.plot([np.min(obswave[i,idx]),np.max(obswave[i,idx])], [mflux * (1.05 + (i-0.5) * 0.25/model.nabs), mflux * (1.05 + (i-0.5) * 0.25/model.nabs)], 'k--')
                lstr = np.array(['c--','m--','y--','r--'])
                plt.plot([np.min(obswave[i,idx]),np.max(obswave[i,idx])], [mflux * (1.05 + i * 0.25/model.nabs), mflux * (1.05 + i * 0.25/model.nabs)], lstr[i % lstr.size])

    ###############################################################################################
    def mkmodel(self,model,oversamp=1):
        waveall     = np.interp(np.arange(oversamp*self.waveall.size)/oversamp,     np.arange(self.waveall.size),     self.waveall)
        legfitrange = np.interp(np.arange(oversamp*self.legfitrange.size)/oversamp, np.arange(self.legfitrange.size), self.legfitrange)
        lsf         = np.interp(np.arange(oversamp*self.lsf.size)/oversamp,         np.arange(self.lsf.size),         self.lsf)
        if lsf.size % 2 == 0:
            lsf = lsf[:-1]
        if model.absonly: # Just make a normalized spectrum
            continuum = np.ones(waveall.size)
        else: # Start with the unabsorbed spectrum as well
            continuum = np.interp(waveall, model.Hwaves, np.dot(np.transpose(model.H), model.w)) if np.max(model.w) > 1.0e-10 else np.zeros(waveall.size)
            ###############################################################################################
            if model.a.size > 0:
                continuum += np.polynomial.legendre.legval(legfitrange, model.a)
        ###############################################################################################
        if model.nabs > 0:
            optdepth = model.taubb(waveall)

            if model.absonly == False:
                hrfit  = np.multiply(continuum, np.exp(-optdepth))
            else:
                hrfit  = np.exp(-optdepth)

            fit = convolve(np.squeeze(hrfit), lsf, boundary='extend')
        else:
            fit = continuum
        ###############################################################################################

        return fit

    ###############################################################################################
    def normspec(self,sdss,box1,box2,makeplot=True,mode='nmf'):

        vdx      = np.extract((self.wavelength > sdss.wavebins[0]) & (self.wavelength < sdss.wavebins[-1]), range(self.wavelength.size))
        boxsize1 = int( box1 / np.average(self.dwave[vdx]))
        boxsize2 = int( box2 / np.average(self.dwave[vdx]))
        if boxsize1 % 2 == 0:
            boxsize1 += 1
        if boxsize2 % 2 == 0:
            boxsize2 += 1

        ldx = np.extract((self.wavelength > 1700) & (self.wavelength < sdss.wavebins[-1]), range(self.wavelength.size))
        refflux = np.average(self.flux[ldx])

        plt.clf()
        if mode == 'cluster':
            print("Calculating distances...")
            distances = np.zeros(sdss.fluxmat.shape[0])
            for i in range(sdss.fluxmat[0,:].size):
                fint = np.interp(self.wavelength[vdx], sdss.wavebins, sdss.fluxmat[i,:])
                fint /= np.average(fint[ldx])
                distances = np.sqrt(np.square(self.flux[vdx]/refflux - fint))
            ddx = np.argsort(distances)

            # Look at the combinations of anywhere from 3 to 30 nearest neighbors,
            # make distance-weighted composites, and record the chi-squared values
            chisq = np.array([])
            nc    = np.array([], dtype=np.int32)
            for ncomb in range(3,30):
                modflux = np.zeros(vdx.size)
                for i in range(1,ncomb+1):
                    fint = np.interp(self.wavelength[vdx], sdss.wavebins, sdss.fluxmat[ddx[i],:])
                    fint /= np.average(fint[ldx])
                    modflux += fint / np.square(distances[ddx[i]])
                modflux /= np.sum(1 / np.square(distances[ddx[1:ncomb+1]]))
                chisq = np.append(chisq, np.sum(np.square(self.flux[vdx]/refflux - fint)) / vdx.size)
                nc    = np.append(nc, ncomb)

            # Pick out the composite with the best chi-squared
            cdx = np.int32(np.extract(chisq == np.min(chisq), range(chisq.size))[0])
            modflux = np.zeros(vdx.size)
            for i in range(1,nc[cdx]+1):
                fint = np.interp(self.wavelength[vdx], sdss.wavebins, sdss.fluxmat[ddx[i],:])
                fint /= np.average(fint[ldx])
                modflux += fint / np.square(distances[ddx[i]])
            modflux /= np.sum(1 / np.square(distances[ddx[1:nc[cdx]+1]]))
            lvdx = np.extract(self.wavelength[vdx] > 1700, range(vdx.size))
            modflux /= np.average(modflux[lvdx])
        else: # mode = 'nmf'...
            modflux = sdss.calcspec(sdss.transformspec(self.wavelength,
                                                       self.flux/refflux,
                                                       self.ivar * refflux * refflux))
#            modflux = sdss.calcspec(sdss.transformspec(self.wavelength,
#                                                       convolve(self.flux/refflux, np.ones(boxsize1)/boxsize1,
#                                                                boundary='extend'),
#                                                       self.ivar * refflux * refflux))

        if mode == 'cluster':
            mmodflux = np.copy(modflux)
        else: # mode = 'nmf'...
            mmodflux = np.interp(self.wavelength[vdx], sdss.wavebins, modflux)  # vdx.size
        nflux    = np.multiply(self.flux[vdx]/refflux, 1/(mmodflux+1.0e-7))  # vdx.size
        ferr     = (1/np.sqrt(self.ivar[vdx]))/refflux
        nferr    = np.multiply(ferr, 1/(mmodflux+1.0e-7))


        # 12/8/2024 - Trying a new approach to accounting for absorption instead of the while loop below
        # Calculate the equivalent with spectra using the NMF-normalized flux spectra
        # Do two smoothings of boxsize1 and boxsize2 without rejecting absorption and then one boxcar2
        # smoothing with an abs rejection.
        nnflux = np.copy(nflux)
        nnferr = np.copy(nferr)
        for (bcs,nsig) in [[boxsize1,3000],[boxsize2,3000],[boxsize2,0.5],[boxsize2,0.5]]:
            ew      = (1.0 - nnflux)*self.dwave[vdx] # vdx.size
            sew     = self.dwave[vdx] * nferr #     np.zeros(vdx.size)

            bigew,bigsew = self.calcbigew(ew,sew)
            # Indices where there is no absorption - to be used for interpolating before smoothing
            adx = np.extract(bigew < nsig*bigsew, range(vdx.size))
            pre_smnflux = np.interp(self.wavelength[vdx], self.wavelength[vdx[adx]], nnflux[adx]) # vdx.size

            # Smooth and renormalize
            smnflux = np.zeros(vdx.size)  # vdx.size
            smnflux = convolve(pre_smnflux, np.ones(bcs)/bcs, boundary='extend')  # vdx.size
            nnflux  = np.multiply(nnflux, 1/(smnflux+1.0e-7)) # vdx.size
            nnferr  = np.multiply(nnferr, 1/(smnflux+1.0e-7)) # vdx.size

            if bcs == boxsize1:
                psmnflux1 = np.copy(pre_smnflux)
                smnflux1 = np.copy(smnflux)
                nnflux1  = np.copy(nnflux)
            elif nsig == 3000:
                psmnflux2 = np.copy(pre_smnflux)
                smnflux2 = np.copy(smnflux)
                nnflux2  = np.copy(nnflux)
            else:
                psmnflux3 = np.copy(pre_smnflux)
                smnflux3 = np.copy(smnflux)
                nnflux3  = np.copy(nnflux)
                bigew3   = np.copy(bigew)
                bigsew3  = np.copy(bigsew)

        ew      = (1.0 - nnflux)*self.dwave[vdx] # vdx.size
        sew     = self.dwave[vdx] * nnferr #     np.zeros(vdx.size)
        bigew,bigsew = self.calcbigew(ew,sew)

        if makeplot:
            plt.subplot(5,1,1)
            plt.step(self.wavelength[vdx],self.flux[vdx]/refflux, label='spectrum')
            plt.plot(self.wavelength[vdx],mmodflux, label=mode)
            plt.plot(self.wavelength[vdx],
                     np.multiply(np.multiply(np.multiply(mmodflux,
                                                         smnflux1),
                                             smnflux2),
                                 smnflux3),
                     label=mode+' x smoothing')
            (ylo,yhi) = plt.ylim()
#            print(np.extract(bigew >= 3.0*bigsew, range(vdx.size)).size)
#            for vv in np.extract(bigew >= 3.0*bigsew, range(vdx.size)):
#                if vv > vdx.size//2:
#                    plt.plot([self.wavelength[vdx[vv]],self.wavelength[vdx[vv]]],[ylo,yhi],'k--')
            (xlo,xhi) = plt.xlim()
            plt.legend()

            plt.subplot(5,1,2)
            plt.step(self.wavelength[vdx],np.multiply(self.flux[vdx]/refflux, 1/(mmodflux+1.0e-7)))
            plt.step(self.wavelength[vdx],nflux)
            plt.step(self.wavelength[vdx],smnflux1)
            plt.xlim(xlo,xhi)
            (ylo,yhi) = plt.ylim()

            plt.subplot(5,1,3)
            plt.step(self.wavelength[vdx],nnflux1)
            plt.step(self.wavelength[vdx],smnflux2)
            plt.xlim(xlo,xhi)
            plt.ylim(ylo,yhi)

            plt.subplot(5,1,4)
            plt.step(self.wavelength[vdx],nnflux2, label='og')
            plt.step(self.wavelength[adx],nnflux2[adx], label='og-abs')
            plt.plot(self.wavelength[vdx],smnflux3)

#            plt.fill_between(self.wavelength[vdx[-vdx.size//2:]],
#                             +bigsew3[vdx[-vdx.size//2:]],
#                             -bigsew3[vdx[-vdx.size//2:]],
#                             color='#00274c')
#            plt.step(self.wavelength[vdx[-vdx.size//2:]],
#                     -bigew3[vdx[-vdx.size//2:]], color='#ffcb05')


            plt.xlim(xlo,xhi)
            plt.ylim(ylo,yhi)
            plt.legend()

            plt.subplot(5,1,5)
            plt.step(self.wavelength[vdx],nnflux)
            plt.step(self.wavelength[vdx],np.ones(nnflux.size))
            plt.xlim(xlo,xhi)
            plt.ylim(ylo,yhi)

        plt.pause(0.001)

        updx = np.extract((nnflux > 5) | (nnferr > 5), range(nnflux.size))
        if updx.size > 0:
            nnflux[updx] = 1
            nnferr[updx] = 1

        return vdx,nnflux,nnferr,ew,sew,bigew,bigsew

##        OLD STUFF
##        absmask = np.where(bigew > 1.5*bigsew, np.ones(vdx.size, dtype=np.int8), np.zeros(vdx.size, dtype=np.int8))  # vdx.size
##
##        smnflux = np.zeros(vdx.size)  # vdx.size
##
##        smnflux = convolve(nflux, np.ones(boxsize1)/boxsize1, boundary='extend')  # vdx.size
##        nnflux  = np.multiply(nflux, 1/(smnflux+1.0e-7)) # vdx.size
##        nnferr  = np.multiply(nferr, 1/(smnflux+1.0e-7)) # vdx.size
##
##        smnflux = convolve(nnflux, np.ones(boxsize2)/boxsize2, boundary='extend')  # vdx.size
##        nnflux  = np.multiply(nnflux, 1/(smnflux+1.0e-7)) # vdx.size
##        nnferr  = np.multiply(nnferr, 1/(smnflux+1.0e-7)) # vdx.size
##
##
##
##
##        na = 0
##        oldna = 1
##        fn = np.fabs((oldna-na))/(np.min([na,oldna])+1.0e-7)
##        oldfn = 40
##        nit = 100
##        while (na != oldna) and (fn != oldfn) and (nit > 0):
##            oldna = na
##            oldfn = fn
##            nit -= 1
##            ew      = (1.0 - nnflux)*self.dwave[vdx] # vdx.size
##            sew     = self.dwave[vdx]/np.multiply(np.multiply(mmodflux, smnflux), np.sqrt(self.ivar[vdx])) #     np.zeros(vdx.size)
##
##            bigew,bigsew = self.calcbigew(ew,sew)
##
##            absmask = np.where(bigew > 1.5*bigsew, np.ones(vdx.size, dtype=np.int8), np.zeros(vdx.size, dtype=np.int8))  # vdx.size
##            na = np.sum(absmask, dtype=np.int16)
##            fn = np.fabs((oldna-na))/(np.min([na,oldna])+1.0e-7)
##
##            nflux = np.multiply(self.flux[vdx], 1/(mmodflux+1.0e-7))  # vdx.size
##            nferr = 1/np.multiply(np.sqrt(self.ivar[vdx]), mmodflux+1.0e-7)  # vdx.size
##
##            adx = np.extract(absmask == 1, range(vdx.size))
##            if adx.size > 0:
##                nadx = np.extract(absmask == 0, range(vdx.size))
##                nflux[adx] = np.interp(self.wavelength[vdx[adx]],self.wavelength[nadx],nflux[nadx])
##                nferr[adx] = np.interp(self.wavelength[vdx[adx]],self.wavelength[nadx],nferr[nadx])
##
##            smnflux1 = np.ones(vdx.size)
##            smnflux1 = convolve(nflux, np.ones(boxsize1)/boxsize1, boundary='extend')  # vdx.size
##            nnflux1  = np.multiply(self.flux[vdx],            1/(np.multiply(mmodflux, smnflux1)+1.0e-7)) # vdx.size
##            nnferr1  = 1/np.multiply(np.sqrt(self.ivar[vdx]), np.multiply(mmodflux, smnflux1)+1.0e-7) # vdx.size
##
##            if adx.size > 0:
##                nnflux1[adx] = np.interp(self.wavelength[vdx[adx]],self.wavelength[nadx],nnflux1[nadx])
##                nnferr1[adx] = np.interp(self.wavelength[vdx[adx]],self.wavelength[nadx],nnferr1[nadx])
##            smnflux2 = convolve(nnflux1, np.ones(boxsize2)/boxsize2, boundary='extend')  # vdx.size
##            nnflux  =   np.multiply(self.flux[vdx],         1/(np.multiply(np.multiply(mmodflux, smnflux1),smnflux2)+1.0e-7)) # vdx.size
##            nnferr  = 1/np.multiply(np.sqrt(self.ivar[vdx]),   np.multiply(np.multiply(mmodflux, smnflux1),smnflux2)+1.0e-7) # vdx.size
##
##            self.continuum = np.ones(self.wavelength.size)
##            self.continuum[vdx] = np.multiply(self.flux[vdx], 1/nnflux)
##
##            ew      = (1.0 - nnflux)*self.dwave[vdx] # vdx.size
##            sew     = self.dwave[vdx]/np.multiply(np.multiply(mmodflux, smnflux), np.sqrt(self.ivar[vdx])) #     np.zeros(vdx.size)
##            bigew,bigsew = self.calcbigew(ew,sew)

    ###############################################################################################
    def pltspec(self,fit,model):

        plt.clf()

        #################################################################################################
        plt.subplot(211)
        plt.xlim([self.waveall[0],self.waveall[-1]])
        plt.xlabel('Rest Wavelength [A]')
        plt.ylabel('Flux')
        tstr = "qnum: "+str(self.qnum)+" S/N = "+str(np.round(np.average(np.multiply(self.nflux[self.fpx],1/self.nferr[self.fpx])),1))
        if model.a.size > 0:
            tstr = tstr+" a = "+str(model.a.size)
        plt.title(tstr)

        mflux = 1.0
        if self.vdx.size == 0:
            nmfcont    = np.dot(np.transpose(model.H), model.w)
            nmfspl     = CubicSpline(model.Hwaves,nmfcont)
            ndx        = np.extract((self.waveall > np.min(model.Hwaves)) & (self.waveall < np.max(model.Hwaves)), range(self.waveall.size))
            nmfcont    = nmfspl(self.waveall[ndx])
            legcont    = np.zeros(self.waveall.size)
            if model.a.size > 0:
                legcont = np.polynomial.legendre.legval(self.legfitrange, model.nha)
            spfit = CubicSpline(self.waveall, fit)

            plt.step(self.waveall, self.flux[self.fpx],               'k') # Spectrum
            plt.step(self.waveall, np.sqrt(1.0/self.ivar[self.fpx]),  'k') # Error spectrum
            plt.plot(self.waveall[ndx], nmfcont,                    'm--') # NMF continuum
            plt.step(self.waveall[ndx], nmfcont+legcont[ndx],       'b--') # NMF + Legendre continuum
            plt.plot(self.waveall, spfit(self.waveall),               'g', linewidth=3.0) # Total fit

            sflux = np.sort(self.nflux[self.fpx])
        else:
            spfit = CubicSpline(self.waveall,  fit)
            sflux = np.sort(self.nflux[self.fpx])

            plt.step(self.waveall, self.nflux[self.fpx],     'k')
            plt.step(self.waveall, self.nferr[self.fpx],     'k')
            plt.step(self.waveall, np.ones(self.fpx.size), 'b--')
            plt.plot(self.waveall, spfit(self.waveall),      'g', linewidth=3.0)


        mflux = sflux[int(0.9*sflux.size)]
        self.markspec(model,mflux,self.waveall[0],self.waveall[-1])

        if (self.msigmawave > self.waveall[0]) and (self.msigmawave < self.waveall[-1]):
            plt.plot([self.msigmawave,self.msigmawave],[0,mflux * 1.35],'c--',linewidth=3.0)
        plt.ylim([0,mflux * 1.35])


        #################################################################################################
        # Msigma panel
        if (self.msigmawave > 0):
            plt.subplot(212)
            plt.xlim([self.msigmawave-20,self.msigmawave+20])
            plt.xlabel('Rest Wavelength [A]')
            plt.ylabel('Flux')

            spfit = CubicSpline(self.waveall,  fit)
            sflux = np.sort(self.nflux[self.fpx])
            plt.step(self.waveall, self.nflux[self.fpx],     'k')
            plt.step(self.waveall, self.nferr[self.fpx],     'k')
            plt.step(self.waveall, np.ones(self.fpx.size), 'b--')
            plt.plot(self.waveall, spfit(self.waveall),      'g', linewidth=3.0)
            lstr = np.array(['c--','m--','y--','r--'])
            for i in range(model.nabs):
                plt.plot(self.waveall, self.mkmodel(fitmodel(model.Hwaves, model.H, model.w, model.a, 1, model.vabs[i], model.bhis[i], model.bnts[i], model.tabs[:,i], model.atoms)), lstr[i % lstr.size])
            plt.plot([self.msigmawave,self.msigmawave],[0,mflux * 1.35],'c--',linewidth=3.0)

#                for j in range(model.atoms.nion):
#                    idx = np.extract((model.atoms.anum == model.atoms.anum[model.atoms.idx[j]]) & (model.atoms.ion == model.atoms.ion[model.atoms.idx[j]]), range(model.atoms.anum.size))
#                    gsdx = model.atoms.getspecies(model.atoms.anum[model.atoms.idx[j]],model.atoms.ion[model.atoms.idx[j]])
#                    if gsdx.size > 1:
#                        obswave = np.squeeze(calcwave(model.vabs[i], model.atoms.wave[gsdx]))
#                    else:
#                        obswave = np.array([np.squeeze(calcwave(model.vabs[i], model.atoms.wave[gsdx]))])
#                    for g in range(gsdx.size):
#                        if (obswave[g] > self.msigmawave-20) and (obswave[g] < self.msigmawave+20):
#                            tau0 = model.tabs[j,i] * model.atoms.flam[gsdx[g]] / np.max(model.atoms.flam[gsdx])
#                            b    = np.sqrt(model.bhis[i] * model.bhis[i] * 1.008 / model.atoms.amass[model.atoms.idx[j]] + model.bnts[i] * model.bnts[i])
#                            plt.plot(self.waveall, convolve(np.exp(-np.squeeze(model.atoms.tausingle(self.waveall,model.atoms.wave[gsdx[g]],tau0,b,model.vabs[i]))), self.lsf), lstr[i % lstr.size])



            mflux = sflux[int(0.9*sflux.size)]
            self.markspec(model,mflux,self.msigmawave-20,self.msigmawave+20)
            plt.ylim([0,mflux * 1.35])

        plt.pause(0.01)

    ###############################################################################################
    def pltspeconly(self,qnum,snr,myatoms,vabsfile='blah'):
        if os.path.exists(vabsfile):
            data = ascii.read(vabsfile, format="no_header")
            vabs = np.array(data['col1'])
            model = fitmodel(np.array([]),np.array([]),np.array([]),np.array([]),vabs.size,vabs,np.array([]),np.array([]),np.array([]),myatoms)

        plt.clf()

        sps = np.array([311,312,313], dtype=int)

        for i in range(sps.size):
            wlo = self.wavelength[0] + (i/sps.size) * (self.wavelength[-1]-self.wavelength[0])
            whi = wlo                + (1/sps.size) * (self.wavelength[-1]-self.wavelength[0])
            sf  = np.sort( np.extract((self.wavelength > wlo) & (self.wavelength < whi) & (self.flux > 0), self.flux))
            plt.subplot(sps[i])
            plt.step(self.wavelength, self.flux,              'k')
            if os.path.exists(vabsfile):
                self.markspec(model,np.average(sf),wlo,whi)
            plt.xlim([wlo,whi])
#            print(sps[i], sf[0],sf[-1])
            plt.ylim([sf[0],sf[-1]])
            plt.yscale('log')
            plt.ylabel('Flux')
            if i == 0:
                plt.title("qnum: "+str(qnum)+" S/N = "+str(np.round(snr,1)))
        plt.pause(0.01)

    ###############################################################################################
    def wciv(self,x):
        return calcwave(x,1548.195)

    ###############################################################################################
    def vciv(self,x):
        return calcvel(x,1548.195)






    def something(self,nvcivdx):
        ilo = np.copy(nvcivdx)
        idx = np.extract(self.bigew[ilo] > 0, range(ilo.size))
        while idx.size > 0:
            ilo[idx] -= 1
            idx = np.extract(self.bigew[ilo] > 0, range(ilo.size))

        ihi = np.copy(nvcivdx)
        idx = np.extract(self.bigew[ihi] > 0, range(ihi.size))
        while idx.size > 0:
            ihi[idx] += 1
            idx = np.extract(self.bigew[ihi] > 0, range(ihi.size))

        return ilo,ihi

    ###############################################################################################


#        for vab in vabs:
#            ldx,hdx = self.getvlims(vab,self.nvbvel)
#            vlo = self.nvbvel[ldx]
#            vhi = self.nvbvel[hdx]

#            nvbew  = 1000*np.sum(np.extract(( self.nvbvel >= vlo) & ( self.nvbvel <= vhi), self.ew))
#            nvrew  = 1000*np.sum(np.extract(( self.nvrvel >= vlo) & ( self.nvrvel <= vhi), self.ew))

#            if (nvbew < 0) or (nvrew < 0):
#                ldx,hdx = self.getvlims(vab,self.civbvel)
#                vlo = self.civbvel[ldx]
#                vhi = self.civbvel[hdx]

#                nvbew  = 1000*np.sum(np.extract(( self.nvbvel >= vlo) & ( self.nvbvel <= vhi), self.ew))
#                nvrew  = 1000*np.sum(np.extract(( self.nvrvel >= vlo) & ( self.nvrvel <= vhi), self.ew))

#            civbew = 1000*np.sum(np.extract((self.civbvel >= vlo) & (self.civbvel <= vhi), self.ew))
#            civrew = 1000*np.sum(np.extract((self.civrvel >= vlo) & (self.civrvel <= vhi), self.ew))
#            hiew   = 1000*np.sum(np.extract((  self.hivel >= vlo) & (  self.hivel <= vhi), self.ew))




#       optdepth = np.zeros(waveall.size)
#        for i in range(model.vabs.size):
#            bhi       = np.sqrt(model.bhis[i] * model.bhis[i]                + model.bnts[i] * model.bnts[i])
#            bnv       = np.sqrt(model.bhis[i] * model.bhis[i] * 1.008/14.007 + model.bnts[i] * model.bnts[i])
#            bciv      = np.sqrt(model.bhis[i] * model.bhis[i] * 1.008/12.011 + model.bnts[i] * model.bnts[i])
#            bmgii     = np.sqrt(model.bhis[i] * model.bhis[i] * 1.008/24.305 + model.bnts[i] * model.bnts[i])

##            print("optdepth.shape = ",optdepth.shape)
##            print((np.sum(self.tauNV(waveall,   model.tabs[0, i], bnv,   model.vabs[i]),axis=1)).shape)

#            optdepth += np.sum(self.tauNV(waveall,   model.tabs[0, i], bnv,   model.vabs[i]),axis=1)
#            optdepth += np.sum(self.tauHI(waveall,   model.tabs[1, i], bhi,   model.vabs[i]),axis=1)
#            optdepth += np.sum(self.tauCIV(waveall,  model.tabs[2, i], bciv,  model.vabs[i]),axis=1)
#            optdepth += np.sum(self.tauMgII(waveall, model.tabs[3, i], bmgii, model.vabs[i]),axis=1)

#        return optdepth

#    def tauNV(self,wave,tau,b,v):
#        nvbdvel   = (calcvel(wave, 1238.821)  - v) / b
#        nvrdvel   = (calcvel(wave, 1242.804)  - v) / b
#        return tau * (np.exp(-np.square(nvbdvel)) + 0.5 * np.exp(-np.square(nvrdvel)))

#    def tauHI(self,wave,tau,b,v):
#        hidvel    = (calcvel(wave, 1215.670)  - v) / b
#        return tau *  np.exp(-np.square(hidvel))

#    def tauCIV(self,wave,tau,b,v):
#        civbdvel  = (calcvel(wave, 1548.195)  - v) / b
#        civrdvel  = (calcvel(wave, 1550.771)  - v) / b
#        return tau * (np.exp(-np.square(civbdvel)) + 0.5 * np.exp(-np.square(civrdvel)))

#    def tauMgII(self,wave,tau,b,v):
#        mgiibdvel = (calcvel(wave, 2796.3543) - v) / b
#        mgiirdvel = (calcvel(wave, 2803.5315) - v) / b
#        return tau * (np.exp(-np.square(mgiibdvel)) + 0.5 * np.exp(-np.square(mgiirdvel)))





        # Velocity space, stack is lox-weight geometric mean of bigew
##
##        print("Looking through wavelength range ",self.wavelength[veldx[0]]," - ",self.wavelength[veldx[-1]])
##        print("Looking through velocity   range ",vel2[0,0]," - 3100")
##        bep    = np.zeros(self.wavelength.size)
##        sigbep = np.zeros(self.wavelength.size)
##        bep[veldx] = np.ones(veldx.size)
##        bep[veldx] *= np.interp(vel[veldx,0], vel[ :,0], np.power(bigew, myatoms.lox[nvdx[0]]))
##        bep[veldx] *= np.interp(vel[veldx,0], vel[ :,1], np.power(bigew, myatoms.lox[nvdx[1]]))
##        bep[veldx] *= np.interp(vel[veldx,0], vel2[:,0], np.power(bigew, myatoms.lox[hidx[0]]))
##        sigbep[veldx]  = np.square(np.interp(vel[veldx,0], vel[ :,0], bigsew) * myatoms.lox[nvdx[0]] / np.interp(vel[veldx,0], vel[ :,0], self.bigew) / totlox)
##        sigbep[veldx] += np.square(np.interp(vel[veldx,0], vel[ :,1], bigsew) * myatoms.lox[nvdx[1]] / np.interp(vel[veldx,0], vel[ :,1], self.bigew) / totlox)
##        sigbep[veldx] += np.square(np.interp(vel[veldx,0], vel2[:,0], bigsew) * myatoms.lox[hidx[0]] / np.interp(vel[veldx,0], vel2[:,0], self.bigew) / totlox)
##        sigbep[veldx]  = np.power(bep[veldx], 1/totlox) * np.sqrt(sigbep[veldx])
##
##        bep = np.power(bep, 1/totlox)
"""