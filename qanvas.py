#import os
#import numpy as np
#import matplotlib.pyplot as plt
#import scipy as sp
import sys
#import time

#from scipy.interpolate     import CubicSpline
#from scipy.optimize        import minimize, Bounds, leastsq
#from scipy.optimize        import root_scalar  as rs
#from astropy.convolution   import convolve
#from sklearn.decomposition import NMF
#from astropy               import units        as u
#from astropy.coordinates   import SkyCoord
#from astropy.io            import fits, ascii
#from astropy.table         import QTable,Table
#from tqdm                  import tqdm

#from atomic                import atomic
#from doppler               import calcvel, calcvelzz, calcwave
#from fitmodel              import fitmodel
from sdsscat               import sdsscat
#from spectrum              import spectrum
#from chisqfunc             import chisqfunc
from readpars              import readpars

# timeit to time a function
# tqdm for a progress bar


####################################################################
breakstr = "#" * 50

inputfile = "qanvaspar.txt"

if len(sys.argv) > 1:
    inputfile = sys.argv[1]

print(breakstr)
print(breakstr)
print("Prepare to QANVAS Cosmic Noon!")
print(breakstr)
####################################################################
mypars = readpars(inputfile)
# Parameters and Program Flow


if mypars.verbose:
    print('---------------------------------------')
    print('Running with the following parameters:')
    print('---------------------------------------')
#    print('R = ', R, '(dvres = ',dvres,', dwres = ',dwres,')')
    print('nsamp = ',mypars.nsamp)
#    print(snrlim, ' < S/N < ',snrulim)
#    print("min LOX = ",minlox)
#    print("n-sigma for detection = ",nsigma)
    print('---------------------------------------')
    print('Base directory: ',mypars.basedir)
#    print('Sample file 1: ',mypars.basedir+'/'+samplefile1)
#    print('Sample file 0: ',mypars.basedir+'/'+samplefile0)
#    print('---------------------------------------')
#    print('NMF parameters...')
#    print('ncomp = ',ncomp)
#    print('dwave = ',dwave)
#    print('sigclip = ',sigclip)
#   print('compfile = ',compfile)
#    print('box1 = ',box1)
#    print('box2 = ',box2)
#    print('---------------------------------------')
#   print('Program Flow:')
#    print('donmf      = ', donmf)
#    print('dofit      = ', dofit)
#    print('absonly    = ', absonly)
#    print('dosearch   = ', dosearch)
#    print('specid     = ', specid)
#    print('firsttpcf  = ', firsttpcf)
#    print('docluster  = ', docluster)
#    print('verbose    = ', verbose)
    print('---------------------------------------')


####################################################################
sdss = sdsscat(mypars)


#t0 = time.time()

# Calling sequence:
#if False: #These things have already been done and can be by-passed. Keeping in case we want to rerun the entire analysis for a clean go
    #                                               readog, readfits, finddups
    # Read in the original list of observations, grab coordinates and redshifts from fits headers, and write samplefile0
#    sdss = sdsscat(basedir, samplefile0,     nsamp, readog=True, readfits=True,  finddups=False)
    # Read in samplefile0 and look for duplicates. Write samplefile0+"2" that has duplicates accounted.
#    sdss = sdsscat(basedir, samplefile0,     nsamp, readog=True, readfits=False, finddups=True)
    # Read in samplefile0+"2".
#    sdss = sdsscat(basedir, samplefile0+"2", nsamp, readog=True, readfits=False, finddups=False)
    # Generate combined ASCII spectra
#    sdss.mkasciispec(samplefile1)

#sdss = sdsscat(basedir, samplefile1,      nsamp)

#if verbose:
#    print("Finished reading catalog in "+str(time.time()-t0)+" seconds")

#if False:
#    sdss.readspec(dwave,snrlim,R,clobberfits=True, clobbercomp=True)
#    sdss.readspec(dwave,snrlim,R,clobberfits=False,clobbercomp=True)

# Note: 54002 unique objects

#if donmf:
#    sdss.readspec(dwave,snrlim,R)
#    sdx = np.extract(sdss.snr > snrlim, range(sdss.snr.size))
#    sdss.nmf(ncomp,sigclip,makeplot=True)
#    sdss.writecomp(compfile)
#    Wnew = np.zeros((sdss.qnum.size, ncomp))
#    wdx = np.extract(sdss.wavebins < calcwave(np.array([-5000]),np.array([1215.6701]))[0], range(sdss.wavebins.size))
#    fdx = np.extract(np.sum(sdss.fluxmat[:,wdx], axis=1) > 0, range(sdss.fluxmat.shape[0]))
#    Wnew[fdx,:] = np.copy(sdss.W)
#    sdss.W = np.copy(Wnew)
#    sdss.pltcomp(ncomp//2,2)
#    meanW = np.average(sdss.W[fdx,:], axis=0)
#    stdW  = np.std(    sdss.W[fdx,:], axis=0)
#    for (warray,prtstr) in [(meanW,f"<W> = "),(stdW,f"s_W= ")]:
#        for w in warray: prtstr += f" {w:.5f}"
#        print(prtstr)
#    input("Finished with NMF. Hit any key to end program.")

#if docluster:
#    sdss.readspec(dwave,snrlim,R)
#    sdss.clustering()

#if dofit or dosearch or specid:
#    t1 = time.time()
#    sdss.readcomp(compfile, ncomp)
#    if verbose: print("Finished reading NMF components in "+str(time.time()-t1)+" seconds")
#    uniqarray, udx = np.unique(sdss.qnum, return_index=True)

#    if dosearch:

        # What is the NV1238 velocity corresponding to HI1215 @ -5000 km/s?
#        nvvelhiaal = np.fabs(calcvel(calcwave(np.array([-5000]),
#                                              np.array([1215.6701])),
#                                     np.array([1238.821])))[0]

#        sdss.readspec(dwave,snrlim,R)

#        ncivq = 0
#        nnvq  = 0
#        nncq  = 0
#        nq    = 0
#        vrangeflag = 0
#        fpxflag    = 0
#        snrflag    = 0
#        nvcovflag  = 0
#        with open("aalsearch2.dat", "w") as f:
#            if verbose: print('dosearch - Looping through ',udx.size,' objects')
#            with open(basedir+"/searchlog", "w") as searchlog:
#                for q in np.extract(sdss.qnum[udx] > -1, udx):
#                    datfile = basedir+'/ASCII/' + sdss.fileroot[q] + '.dat'
#                    if (os.path.exists(datfile)) and (np.all(sdss.qnum[:q] != sdss.qnum[q])):
#                        if verbose: print('Reading in ',datfile) #,' and writing ',linfile)
#                        searchlog.write(('Reading in {}\n').format(datfile))
#                        myspec = spectrum(datfile,sdss.qnum[q],R)
#                        myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[q]))
#                        vvv = np.fabs(calcvel(myspec.wavelength,myatoms.wave[myatoms.getspecies(7,5)]))
#                        if (myspec.vrange >= nvvelhiaal) and (myspec.fpx.size > 0) and (sdss.snr[q] > snrlim) and (np.min(vvv) < 200):
#                            vdx,nnflux,nnferr,ew,sew,bigew,bigsew = myspec.normspec(sdss,box1,box2,makeplot=True)

#                            nciva,nnva,nnvciva = myspec.dosearch(sdss,q,box1,box2,verbose,myatoms,basedir,makeplot=True)

#                            nq += 1
#                            if nciva   > 0: ncivq += 1
#                            if nnva    > 0: nnvq  += 1
#                            if nnvciva > 0: nncq  += 1

#                            print(f"{sdss.qnum[q]:5d} {nciva:3d} {nnva:3d} {nnvciva:3d} {nq:5d} {ncivq/nq:5.3f} {nnvq/nq:5.3f} {nncq/nq:5.3f}")
#                            f.write(f"{sdss.qnum[q]:5d} {nciva:3d} {nnva:3d} {nnvciva:3d}\n")
#                            searchlog.write(f"{sdss.qnum[q]:5d} {nciva:3d} {nnva:3d} {nnvciva:3d}\n")
#                            plt.pause(5)
#                            input("Inspect dosearch's work and then hit any key to continue...")

#                        else:
#                            dumstr = ""
#                            printit = False
#                            if (myspec.vrange < nvvelhiaal):
#                                vrangeflag += 1
#                                dumstr += f"            myspec.vrange = {myspec.vrange}"
#                                printit = True
#                            if myspec.fpx.size <= 0:
#                                fpxflag += 1
#                                dumstr += f"            myspec.fpx.size = {myspec.fpx.size}"
#                                printit = True
#                            if (sdss.snr[q] <= snrlim):
#                                dumstr += f"            sdss.snr[{q}] = {sdss.snr[q]}"
#                                snrflag += 1
#                                printit = True
#                            if vvv.size > 0:
#                                if (np.min(vvv) >= 200):
#                                    dumstr += f"            Closest pixel to N V = {np.min(np.fabs(calcvel(myspec.wavelength,myatoms.wave[myatoms.getspecies(7,5)])))}"
#                                    printit = True
#                            else:
#                                dumstr += "            N V not covered"
#                                printit = True
#                                nvcovflag += 1
#                            if printit:
#                                print(dumstr)
#                                print(f"            vrangeflag {vrangeflag} fpxflag {fpxflag} snrflag {snrflag} nvcovflag {nvcovflag}")
#                                searchlog.write(f"            vrangeflag {vrangeflag} fpxflag {fpxflag} snrflag {snrflag} nvcovflag {nvcovflag}\n")

#                    else:
#                        if (os.path.exists(datfile) == False):
#                            print(f"{datfile} does not exist")
#                            searchlog.write(f"{datfile} does not exist\n")
#                        else:
#                            print(f"qnum = {sdss.qnum[q]} repeat offender")
#                            searchlog.write(f"qnum = {sdss.qnum[q]} repeat offender\n")
                            






##                    oldilo = 0
##                    oldihi = 0
##                    for i in range(nvcivdx.size):
##                        if (ilo[i] != oldilo) and (oldihi != ihi[i]):
##                            oldilo = ilo[i]
##                            iloihi = ihi[i]
##                            v      = (np.extract(myspec.bigew[ilo[i]:ihi[i]] == np.max(myspec.bigew[ilo[i]:ihi[i]]), myspec.civbvel[ilo[i]:ihi[i]]))[0]
##
##                            nvw    = calcwave(myspec.civbvel[ilo[i]:ihi[i]],1238.821)
##                            nvwdx  = np.extract((myspec.wavelength > np.min(nvw)) & (myspec.wavelength < np.max(nvw)), range(myspec.wavelength.size))
##                            nvidx  = (np.extract(myspec.bigew[nvwdx] == np.max(myspec.bigew[nvwdx]), nvwdx))[0]
##                            nvilo  = nvidx
##                            while (myspec.bigew[nvilo] > 0) and (nvilo > 0):
##                                nvilo -= 1
##                            nvihi  = nvidx
##                            while myspec.bigew[nvihi] > 0:
##                                nvihi += 1
##
##                            # Is the N V doublet separated? --> is the velocity range larger than the velocity separation?
##                            if calcvel(myspec.wavelength[nvihi],myspec.wavelength[nvilo]) > calcvel(1242.804,1238.821): #Then it is blended
##                                nvblo = nvilo
##                                # Use the upper limit as an upper limit on 1242 and scale back to the end of 1238
##                                nvrhi = nvihi
##                                nvm   = (np.extract(myspec.wavelength > myspec.wavelength[nvihi]*1238.821/1242.804, range(myspec.wavelength.size)))[0]
##                            else: #Not blended
##                                nvblo = nvilo
##                                nvm   = nvihi
##                                nvrhi = (np.extract(myspec.wavelength > myspec.wavelength[nvm]*1242.804/1238.821, range(myspec.wavelength.size)))[0]
##                            plt.subplot(211)
##                            plt.plot([myspec.wavelength[nvblo],myspec.wavelength[nvblo]],[0,(np.extract(myspec.wavelength > myspec.wavelength[nvblo], myspec.flux))[0]],"g--")
##                            plt.plot([myspec.wavelength[nvm],  myspec.wavelength[nvm]],  [0,(np.extract(myspec.wavelength > myspec.wavelength[nvm],   myspec.flux))[0]],"g--")
##                            plt.plot([myspec.wavelength[nvrhi],myspec.wavelength[nvrhi]],[0,(np.extract(myspec.wavelength > myspec.wavelength[nvrhi], myspec.flux))[0]],"g--")
##
##                            plt.subplot(212)
##                            wlo = calcwave(myspec.civbvel[ilo[i]], 1548.195)
##                            whi = calcwave(myspec.civbvel[ihi[i]], 1550.771)
##                            w = (np.extract(myspec.bigew[ilo[i]:ihi[i]] == np.max(myspec.bigew[ilo[i]:ihi[i]]), myspec.wavelength[ilo[i]:ihi[i]]))[0]
##                            wm = w * (1 + 1550.771/1548.195) / 2
##                            plt.plot([wlo,wlo], [0.0, (np.extract(myspec.wavelength > wlo, myspec.flux))[0]], "g--")
##                            plt.plot([whi,whi], [0.0, (np.extract(myspec.wavelength > whi, myspec.flux))[0]], "g--")
##                            plt.plot([wm,  wm], [0.0, (np.extract(myspec.wavelength > wm,  myspec.flux))[0]], "g--")
##
##                            civewb = 1000 * np.sum(np.extract((myspec.wavelength > wlo)                      & (myspec.wavelength < wm),                       myspec.ew))
##                            civewr = 1000 * np.sum(np.extract((myspec.wavelength > wm)                       & (myspec.wavelength < whi),                      myspec.ew))
##                            nvewb  = 1000 * np.sum(np.extract((myspec.wavelength > myspec.wavelength[nvblo]) & (myspec.wavelength < myspec.wavelength[nvm]),   myspec.ew))
##                            nvewr  = 1000 * np.sum(np.extract((myspec.wavelength > myspec.wavelength[nvm])   & (myspec.wavelength < myspec.wavelength[nvrhi]), myspec.ew))
##
##                            civsewb = 1000 * np.sqrt(np.sum(np.extract((myspec.wavelength > wlo)                      & (myspec.wavelength < wm),                       np.square(myspec.sew))))
##                            civsewr = 1000 * np.sqrt(np.sum(np.extract((myspec.wavelength > wm)                       & (myspec.wavelength < whi),                      np.square(myspec.sew))))
##                            nvsewb  = 1000 * np.sqrt(np.sum(np.extract((myspec.wavelength > myspec.wavelength[nvblo]) & (myspec.wavelength < myspec.wavelength[nvm]),   np.square(myspec.sew))))
##                            nvsewr  = 1000 * np.sqrt(np.sum(np.extract((myspec.wavelength > myspec.wavelength[nvm])   & (myspec.wavelength < myspec.wavelength[nvrhi]), np.square(myspec.sew))))
##
##                            print(("{} {:7.0f} {:7.0f} {:7.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}").format(i, myspec.civbvel[nvcivdx[i]],myspec.nvbvel[nvblo],myspec.nvbvel[nvm], civewb, civsewb, civewr, civsewr, nvewb, nvsewb, nvewr, nvsewr))

##                    break

"""
#    if specid: # dosearch should happen first so that an aalsearch2.dat file exists.
#        qstart  = -1
#        data    = ascii.read("aalsearch2.dat", format="no_header")
#        qnum    = np.extract(data['col1'] > qstart, data['col1'])
#       civsys  = np.extract(data['col1'] > qstart, data['col2'])
#        vabs    = np.empty([])
#        vlo     = np.empty([])
#       vhi     = np.empty([])
#        bepm    = np.empty([])
#        beps    = np.empty([])
#        nvbew   = np.empty([])
#        nvbsew  = np.empty([])
#        nvrew   = np.empty([])
#        nvrsew  = np.empty([])
#        hiew    = np.empty([])
#        hisew   = np.empty([])
#        civbew  = np.empty([])
#        civbsew = np.empty([])
#        civrew  = np.empty([])
#        civrsew = np.empty([])
#        mgiibew  = np.empty([])
#        mgiibsew = np.empty([])
#        mgiirew  = np.empty([])
#        mgiirsew = np.empty([])
#        vranges = np.empty([])
#        qnum2   = np.empty([], dtype=np.int32)
#        nq = 0
#        nnq = 0
#        if (True): # Loop through all quasars
#            qcounter = 1000
#            lgratlo = 30
#            lgrathi = -30
#            oldddd = -1
#            for q in udx:
#                qdx = np.extract(qnum == sdss.qnum[q], qnum)
#                if qdx.size > 0:
#                    nq += 1
#                cdx = np.extract(qnum == sdss.qnum[q], civsys)
#                if cdx.size > 0:
#                    if cdx[0] > 0:
#                        datfile  = basedir+'/ASCII/' + sdss.fileroot[q] + '.dat'
#                        vabsfile = basedir+'/Vabs/'  + sdss.fileroot[q] + '.vabs'
#                        if (os.path.exists(datfile)) and (np.all(sdss.qnum[:q] != sdss.qnum[q])) and (False): # Do the absorber search on this object
#                            myspec = spectrum(datfile,sdss.qnum[q],R)
#                            if myspec.vrange > 299792.458:
#                                input("Pause for vrange check")
#                            myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[q]))
#                            vranges = np.append(vranges, -myspec.vrange)
#                            vranges = np.extract(np.fabs(vranges) < 299792.458, vranges) # Sanity check on the array
#                            if myspec.vrange > 10651: # This gives coverage to -5000 km/s for H I
#                                myspec.dosearch(sdss,q,box1,box2,verbose,myatoms,basedir)
#                        else:
#                            if (os.path.exists(datfile)) == False: print("missing datfile -- ",os.path.exists(datfile))

#                        if (os.path.exists(vabsfile)) and (True): # Compile all of the absorbers from all objects
#                            nnq += 1
#                            data    = ascii.read(vabsfile,format="no_header")
#                            vabs    = np.append(vabs,    data['col1'])
#                            vlo     = np.append(vlo,     data['col2'])
#                            vhi     = np.append(vhi,     data['col3'])
#                            bepm    = np.append(bepm,    data['col4'])
#                            beps    = np.append(beps,    data['col5'])
#                            nvbew   = np.append(nvbew,   data['col7'])
#                            nvbsew  = np.append(nvbsew,  data['col8'])
#                            nvrew   = np.append(nvrew,   data['col9'])
#                            nvrsew  = np.append(nvrsew,  data['col10'])
#                            hiew    = np.append(hiew,    data['col11'])
#                            hisew   = np.append(hisew,   data['col12'])
#                            civbew  = np.append(civbew,  data['col13'])
#                            civbsew = np.append(civbsew, data['col14'])
#                            civrew  = np.append(civrew,  data['col15'])
#                            civrsew = np.append(civrsew, data['col16'])
#                            mgiibew  = np.append(mgiibew,  data['col17'])
#                            mgiibsew = np.append(mgiibsew, data['col18'])
#                            mgiirew  = np.append(mgiirew,  data['col19'])
#                            mgiirsew = np.append(mgiirsew, data['col20'])
#                            qnum2   = np.append(qnum2,   np.broadcast_to(sdss.qnum[q],(len(data))))

#                            myspec = spectrum(datfile,sdss.qnum[q],R)
#                            myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[q]))
#                            vranges = np.append(vranges, -myspec.vrange)

#                            cdx = np.extract((civbew > civbsew) & (vabs > -10651) & (nvbew > nvbsew), range(civbew.size))
#                            rat = nvbew[cdx]/civbew[cdx]
#                            subvabs = vabs[cdx]

#                            if (False): # Examine the N V, C IV, and H I velocity-stacked profiles
#                                vref,sigvref = (1091.78998613, 242.20382901)
#                                dvos = np.fabs(subvabs-vref)/sigvref
#                                ddx  = np.extract(dvos < 1, cdx)
#                                if (ddx.size > 3) and (ddx.size > oldddd):
#                                    oldddd = ddx.size
##                                    sdx = np.argsort(vabs[ddx])
#                                    print(ddx.size, qnum2[ddx])
#                                    plt.clf()
#                                    for i in range(ddx.size):
#                                        ddd = ddx[i]
#                                        fileroot = (np.extract(sdss.qnum == qnum2[ddd], sdss.fileroot))[0]
#                                        zqso     = (np.extract(sdss.qnum == qnum2[ddd],     sdss.zqso))[0]
#                                        datfile = basedir+'/ASCII/' + fileroot + '.dat' 
#                                        vabsfile = basedir+'/Vabs/'  + fileroot + '.vabs'
#                                        myspec = spectrum(datfile,qnum2[ddd],R)
#                                        myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+zqso))

#                                        if i >= 9 * (ddx.size//9):
#                                            s = 331 + (i % 9)
#                                            print(i, 9 * (i//9), s, qnum2[ddx[i]])
#                                            plt.subplot(s)

#                                            hidx = myatoms.getspecies(1,1)
#                                            wlim = calcwave(np.array([np.min(vabs[ddx])-3000,np.max(vabs[ddx])+3000]),myatoms.wave[hidx[0]])
#                                            wdx = np.extract((myspec.wavelength > wlim[0]) & (myspec.wavelength < wlim[1]), range(myspec.wavelength.size))
#                                            v = calcvel(myspec.wavelength[wdx],myatoms.wave[hidx[0]])
#                                            v0 = vabs[ddd]
#                                            plt.step(v,myspec.flux[wdx]/np.max(myspec.flux[wdx]), "g")


#                                            civdx = myatoms.getspecies(6,4)
#                                            wlim = calcwave(np.array([np.min(vabs[ddx])-3000,np.max(vabs[ddx])+3000]),myatoms.wave[civdx[0]])
#                                            wdx = np.extract((myspec.wavelength > wlim[0]) & (myspec.wavelength < wlim[1]), range(myspec.wavelength.size))
#                                            vb = calcvel(myspec.wavelength[wdx],myatoms.wave[civdx[0]])
#                                            vr = calcvel(myspec.wavelength[wdx],myatoms.wave[civdx[1]])
#                                            plt.step(vb,myspec.flux[wdx]/np.max(myspec.flux[wdx]) + 1, "b")
#                                            plt.step(vr,myspec.flux[wdx]/np.max(myspec.flux[wdx] )+ 1, "r--")

#                                            nvdx = myatoms.getspecies(7,5)
#                                            wlim = calcwave(np.array([np.min(vabs[ddx])-3000,np.max(vabs[ddx])+3000]),myatoms.wave[nvdx[0]])
#                                            wdx = np.extract((myspec.wavelength > wlim[0]) & (myspec.wavelength < wlim[1]), range(myspec.wavelength.size))
#                                            vb = calcvel(myspec.wavelength[wdx],myatoms.wave[nvdx[0]])
#                                            vr = calcvel(myspec.wavelength[wdx],myatoms.wave[nvdx[1]])
#                                            plt.step(vb,myspec.flux[wdx]/np.max(myspec.flux[wdx]) + 2, "b")
#                                            plt.step(vr,myspec.flux[wdx]/np.max(myspec.flux[wdx]) + 2, "r--")

#                                            (ylo,yhi) = plt.ylim()
#                                            (xlo,xhi) = plt.xlim()
#                                            plt.plot([v0,v0],[ylo,yhi], "r")
#                                            plt.text(xlo + 0.1 * (xhi-xlo),ylo + 0.1 * (yhi-ylo),("qnum {}").format(qnum2[ddx[i]]))

#                                    plt.pause(0.001)

#                            if (cdx.size > 5) and (True): # Analyze the N V 1238/C IV 1548 equivalent width ratio and vabs distribution
#                                rat   = nvbew[cdx]/civbew[cdx]
#                                lgrat = np.log10(rat+1.0e-7)
#                                if np.max(lgrat) > 8:
#                                    input("Paused for examination of lgrat")
#                                vsort = np.sort(vranges)
#                                vsort = np.extract(vsort < 299792.458, vsort)
#                                if np.max(np.fabs(vsort) > 299792.458):
#                                    input("Paused for vsort check")
#                                if np.min(lgrat) < lgratlo:
#                                    lgratlo = np.min(lgrat)
#                                    qqq = np.squeeze(np.extract(lgrat == lgratlo, cdx))
#                                    ldx = -1
#                                    if qqq.size > 1:
#                                        ldx = qqq[0]
#                                    elif qqq.size == 1:
#                                        ldx = qqq
#                                    ldx = np.squeeze(ldx)
#                                if ldx >= 0:
#                                    if np.max(lgrat) > lgrathi:
#                                        lgrathi = np.max(lgrat)
#                                        ldx = np.extract(lgrat == lgrathi, cdx)
#                                    if (True):
#                                        counts, bins = np.histogram(lgrat, bins=30)
#                                        if np.sum(counts) > 50:
                                            # Least squares fit. Starting values found by inspection.
                                            # Best fit: [ 3.81359878e+02, -1.61440791e-01,  2.41220880e-01,  1.59874282e+02,       6.71831581e-01,  4.95647312e-01]
                                            # Root at 0.22411076198716937
#                                            fbins = bins[:-1] + 0.5*(bins[1]-bins[0])
#                                            fit = leastsq(double_gaussian_fit, [np.max(counts),                                   -1.57728785e-01,  2.41639639e-01,
#                                                                                (1.48498268e+02/3.78872182e+02) * np.max(counts),  6.51444240e-01,  5.14383209e-01 ], args=(fbins, counts))
#                                            xlo = 0.5 * (fit[0][1] + fit[0][4])
#                                            xhi = xlo
#                                            while diff_gaussian(xlo, fit[0]) / diff_gaussian(xhi, fit[0]) > 0:
#                                                xlo -= 0.01 * np.min([fit[0][2], fit[0][5]])
#                                                xhi += 0.01 * np.min([fit[0][2], fit[0][5]])
#                                            x = rs(diff_gaussian, args=fit[0], bracket=[xlo,xhi])

#                                        counts2, bins2 = np.histogram(subvabs, bins=30)
#                                        if np.sum(counts2) > 50:
#                                            fbins2 = bins2[:-1] + 0.5*(bins2[1]-bins2[0])
#                                            fit2 = leastsq(dbl_gaussian_cnst_fit, [np.max(counts2), -2.5e+03,  1.0e+03,
#                                                                                   np.max(counts2),  1.5e+03,  1.5e+03,
#                                                                                   counts2[2]], args=(fbins2, counts2))
#                                            #Result: fit2[0] = array([45.74215452, -2271.65388937, 831.85274261, 80.21174863, 1091.78998613, -242.20382901, 18.32894026])

#                                        plt.clf()
#                                        plt.subplot(221)
#                                        plt.plot(vabs[cdx],rat,"ob")
#                                        plt.plot([np.min(vabs[cdx]),   np.max(vabs[cdx])],   [np.median(rat),np.median(rat)],"r--")
#                                        if np.sum(counts) > 50:
#                                            if (fit[0][0] > 0) and (fit[0][1] > np.min(lgrat))  and (fit[0][1] < np.max(lgrat)):
#                                                plt.plot([np.min(vabs[cdx]), np.max(vabs[cdx])], np.power(10.0,[fit[0][1],fit[0][1]]),"m--")
#                                            if fit[0][3] > 0 and (fit[0][4] > np.min(lgrat))  and (fit[0][4] < np.max(lgrat)):
#                                                plt.plot([np.min(vabs[cdx]), np.max(vabs[cdx])], np.power(10.0,[fit[0][4],fit[0][4]]),"m--")
#                                        plt.plot([np.median(vabs[cdx]),np.median(vabs[cdx])],[np.min(rat),   np.max(rat)],   "r--")
#                                        xlo, xhi = plt.xlim()
#                                        plt.yscale("log")
#                                        plt.title(("qnum {} median vabs {} km/s median ratio {} nq {} nnq {}").format(sdss.qnum[q], np.median(vabs[cdx]), np.median(rat), nq, nnq))
#                                        plt.ylabel("W(N V 1238)/W(C IV 1548)")

#                                        plt.subplot(223)
#                                        plt.step(vsort, (1+np.arange(vsort.size))/vsort.size)
#                                        plt.xlim([xlo,xhi])
#                                        if np.min(np.extract(vsort > xlo, (1+np.arange(vsort.size))/vsort.size)) < 1:
#                                            plt.ylim([np.min(np.extract(vsort > xlo, (1+np.arange(vsort.size))/vsort.size)),1])
#                                        plt.xlabel("Relative Velocity (km/s)")
#                                        plt.yscale("log")


#                                        plt.subplot(222)
#                                        plt.stairs(counts, bins)
#                                        plt.xlim([bins[0],bins[-1]])
#                                        plt.ylim([0,1.1*np.max(counts)])
#                                        plt.xlabel("log W(N V 1238)/W(C IV 1548)")

#                                        if np.sum(counts) > 50:
#                                            # Least squares fit. Starting values found by inspection.
#                                            bbins = np.interp(np.arange(10*fbins.size)/10, np.arange(fbins.size), fbins)
#                                            plt.plot(bbins, double_gaussian(bbins, fit[0]),  'r' )
#                                            if x.converged:
#                                                plt.plot([x.root,x.root],[0,np.max(counts)],"m--")
#                                            if fit[0][0] > 0.0:
#                                                plt.plot(bbins, one_gaussian(bbins, fit[0][:3]), 'g--' )
#                                            if fit[0][3] > 0.0:
#                                                plt.plot(bbins, one_gaussian(bbins, fit[0][3:]), 'g--' )
#                                            plt.title(fit[0])
#                                        else:
#                                            plt.title((" {} counts").format(np.sum(counts)))

#                                        plt.subplot(224)
#                                        plt.stairs(counts2, bins2)
#                                        plt.xlabel("Relative Velocity (km/s)")
#                                        plt.xlim([xlo,xhi])
#                                        plt.ylim([0,1.1*np.max(counts2)])
#                                        if np.sum(counts) > 50:
#                                            bbins2 = np.interp(np.arange(10*fbins2.size)/10, np.arange(fbins2.size), fbins2)
#                                            plt.plot(bbins2, dbl_gaussian_cnst(bbins2, fit2[0]),  'r' )
                                            if fit2[0][0] > 0.0:
                                                plt.plot(bbins2, one_gaussian(bbins2, fit2[0][:3]), 'g--' )
                                            if fit2[0][3] > 0.0:
                                                plt.plot(bbins2, one_gaussian(bbins2, fit2[0][3:-1]), 'g--' )

                                        plt.pause(0.1)
                        else:
                            if (os.path.exists(vabsfile)) == False: print("vabsfile exists -- ",(os.path.exists(vabsfile)))
#                else:
#                    print("cdx.size = ",cdx.size)

            if (True):
                with open(basedir+"/Vabs/all.vabs", "w") as f:
                    for i in range(qnum2.size):
                        f.write(("{:6d} {:10.3f} {:10.3f} {:10.3f} {:7.5f} {:7.5f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f} {:5.0f}\n").format(qnum2[i],vabs[i], vlo[i], vhi[i], bepm[i], beps[i], nvbew[i], nvbsew[i], nvrew[i], nvrsew[i], hiew[i], hisew[i], civbew[i], civbsew[i], civrew[i], civrsew[i], mgiibew[i], mgiibsew[i], mgiirew[i], mgiirsew[i]))

        if (False): # Now we have all systems read them in and the equivalent width ratio calculated
            data = ascii.read(basedir+"/Vabs/all.vabs", format="no_header")
            qnum2   = np.array(data['col1'], dtype=np.int32)
            vabs    = np.array(data['col2'])
            vlo     = np.array(data['col3'])
            vhi     = np.array(data['col4'])
            bepm    = np.array(data['col5'])
            beps    = np.array(data['col6'])
            nvbew   = np.array(data['col7'])
            nvbsew  = np.array(data['col8'])
            nvrew   = np.array(data['col9'])
            nvrsew  = np.array(data['col10'])
            hiew    = np.array(data['col11'])
            hisew   = np.array(data['col12'])
            civbew  = np.array(data['col13'])
            civbsew = np.array(data['col14'])
            civrew  = np.array(data['col15'])
            civrsew = np.array(data['col16'])

            # Cull the low-ionization systems for which we didn't even grab hi-ip equivalent widths
            nzdx = np.extract(nvbew+nvrew+civbew+civrew > 0, np.arange(qnum2.size,dtype=np.int32()))
            print("Out of ",qnum2.size," systems, ",nzdx.size," are high-ionization")
            print("Number of quasars with S/N > ",snrlim,": ",(np.extract(sdss.snr > snrlim, sdss.qnum)).size)
            print("Number of quasars with S/N > ",snrlim," with N V/C IV abs: ",(np.unique(qnum2[nzdx])).size)

            rat    = np.zeros(nzdx.size)
            ratlim = np.zeros(nzdx.size)
            rdx = np.extract((nvbew[nzdx] > 0) & (civbew[nzdx] > 0), np.arange(nzdx.size,dtype=np.int32()))
            rat[rdx] = nvbew[nzdx[rdx]] / civbew[nzdx[rdx]]

            rdx = np.extract((nvbew[nzdx] < nvbsew[nzdx]) & (civbew[nzdx] > 0), np.arange(nzdx.size,dtype=np.int32()))
            rat[rdx] = nvbsew[nzdx[rdx]] / civbew[nzdx[rdx]]
            ratlim[rdx] = -1

            rdx = np.extract((nvbew[nzdx] > 0) & (civbew[nzdx] < civbsew[nzdx]), np.arange(nzdx.size,dtype=np.int32()))
            rat[rdx] = nvbsew[nzdx[rdx]] / civbsew[nzdx[rdx]]
            ratlim[rdx] = +1

            plt.clf()
            plt.plot(np.extract(rat > 0.01, vabs[nzdx]),np.extract(rat > 0.01, rat),"ob")
            (xlo,xhi) = plt.xlim()
            plt.plot([xlo,xhi],[1.0,1.0],"r--")
#            (xlo,xhi) = plt.xlim()
#            plt.plot([xlo,xhi],[xlo,xhi],"r--")
#            plt.xscale("log")
            plt.yscale("log")
##            
##            sdx = np.argsort(-rat) # Negative so we get the reverse sort - largest W(NV)/W(C IV) on top
##            for qdx in sdx:
##                if np.max(np.extract(qnum2 == qnum2[qdx], rat)) > 0.22816789553156333:
##                    vdx = np.extract((qnum2 == qnum2[qdx]) & (nvbew > 0.22816789553156333 * civbew), range(qnum2.size))
##                    q = np.extract(sdss.qnum == qnum2[qdx], range(sdss.qnum.size))
##                    q = q[0]
##                    datfile  = basedir+'/ASCII/' + sdss.fileroot[q] + '.dat'
##                    myspec = spectrum(datfile,sdss.qnum[q],R)
##                    myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[q]))
##                    nvvel  = calcvel(myspec.wavelength, myatoms.wave[myatoms.getspecies(7,5)])
##                    civvel = calcvel(myspec.wavelength, myatoms.wave[myatoms.getspecies(6,4)])
##
##                    deltavnv  = np.max(calcvel(myatoms.wave[myatoms.getspecies(7,5)], myatoms.wave[myatoms.getspecies(7,5)]))
##                    deltavciv = np.max(calcvel(myatoms.wave[myatoms.getspecies(6,4)], myatoms.wave[myatoms.getspecies(6,4)]))
##
##                    ndx,nnflux,nnferr,ew,sew,bigew,bigsew = myspec.normspec(sdss,box1,box2)
##                    continuum = myspec.flux[ndx]/nnflux
##
##                    plt.clf()
##                    plt.subplot(221)
##                    nvdx = np.extract((nvvel[:,0] > np.min(vlo[vdx])-2000) & (nvvel[:,0] < np.max(vhi[vdx])+2000), range(nvvel.size))
##                    plt.step(nvvel[nvdx,0],myspec.flux[nvdx])
##                    plt.plot(nvvel[nvdx,0],np.interp(nvvel[nvdx,0],nvvel[ndx,0],continuum))
##                    sf = np.max(myspec.flux[nvdx])
##                    mf = np.min(myspec.flux[nvdx])
##                    plt.plot([vlo[vdx],vhi[vdx]],[mf+0.9*(sf-mf),mf+0.9*(sf-mf)],"b")
##                    plt.plot([vlo[vdx]+deltavnv,vhi[vdx]+deltavnv],[mf+0.8*(sf-mf),mf+0.8*(sf-mf)],"r")
##                    plt.title(("qnum {} W(N V) = {} W(C IV) = {}").format(sdss.qnum[q],nvbew[vdx],civbew[vdx]))
##                    
##                    plt.subplot(223)
##                    civdx = np.extract((civvel[:,0] > np.min(vlo[vdx])-2000) & (civvel[:,0] < np.max(vhi[vdx])+2000), range(civvel.size))
##                    sf = np.max(myspec.flux[civdx])
##                    mf = np.min(myspec.flux[civdx])
##                    plt.step(civvel[civdx,0],myspec.flux[civdx])
##                    plt.plot(civvel[civdx,0],np.interp(civvel[civdx,0],civvel[ndx,0],continuum))
##                    plt.plot([vlo[vdx],vhi[vdx]],[mf+0.9*(sf-mf),mf+0.9*(sf-mf)],"b")
##                    plt.plot([vlo[vdx]+deltavciv,vhi[vdx]+deltavciv],[mf+0.8*(sf-mf),mf+0.8*(sf-mf)],"r")
##
##                    plt.subplot(222)
##                    nvdx = np.extract((nvvel[myspec.fpx,0] > np.min(vlo[vdx])-2000) & (nvvel[myspec.fpx,0] < np.max(vhi[vdx])+2000), range(myspec.fpx.size))
##                    plt.step(nvvel[myspec.fpx[nvdx],0],nnflux[nvdx])
##                    plt.step(nvvel[myspec.fpx[nvdx],0],nnferr[nvdx])
##                    sf = np.max(nnflux[nvdx])
##                    mf = np.min(nnflux[nvdx])
##                    plt.plot(nvvel[myspec.fpx[nvdx],0],np.ones(nvdx.size))
##                    plt.plot([vlo[vdx],vhi[vdx]],[mf+0.9*(sf-mf),mf+0.9*(sf-mf)],"b")
##                    plt.plot([vlo[vdx]+deltavnv,vhi[vdx]+deltavnv],[mf+0.8*(sf-mf),mf+0.8*(sf-mf)],"r")
##                    
##                    plt.subplot(224)
##                    civdx = np.extract((civvel[myspec.fpx,0] > np.min(vlo[vdx])-2000) & (civvel[myspec.fpx,0] < np.max(vhi[vdx])+2000), range(myspec.fpx.size))
##                    plt.step(civvel[myspec.fpx[civdx],0],nnflux[civdx])
##                    plt.step(civvel[myspec.fpx[civdx],0],nnferr[civdx])
##                    sf = np.max(nnflux[civdx])
##                    mf = np.min(nnflux[civdx])
##                    plt.plot(civvel[myspec.fpx[civdx],0],np.ones(civdx.size))
##                    plt.plot([vlo[vdx],vhi[vdx]],[mf+0.9*(sf-mf),mf+0.9*(sf-mf)],"b")
##                    plt.plot([vlo[vdx]+deltavciv,vhi[vdx]+deltavciv],[mf+0.8*(sf-mf),mf+0.8*(sf-mf)],"r")
##
##
##                    plt.pause(0.001)

    if dofit:
        qstart = -1
        data   = ascii.read("../../aalsearch.dat", format="no_header")
        qnum   = np.extract(data['col1'] > qstart, data['col1'])
        civsys = np.extract(data['col1'] > qstart, data['col2'])
        nvsys  = np.extract(data['col1'] > qstart, data['col3'])
        if verbose: print('dofit - Looping through ',udx.size,' objects')
        vranges = np.array([])
        for q in range(udx.size):
            cdx = np.extract(qnum == sdss.qnum[udx[q]], civsys)
            if cdx.size > 0:
                if cdx[0] > 0:
                    datfile = basedir+'/ASCII/' + sdss.fileroot[udx[q]] + '.dat'
                    linfile = basedir+'/Lin/'+sdss.fileroot[udx[q]]+'.lin'
                    if (os.path.exists(datfile)) and (np.all(sdss.qnum[:udx[q]] != sdss.qnum[udx[q]])):
                        if verbose: print('Reading in ',datfile)
                        myspec = spectrum(datfile,sdss.qnum[udx[q]],R)
                        if (myspec.vrange >= 5000) and myspec.fpx.size > 0 and (sdss.snr[udx[q]] > snrlim): # and (sdss.qnum[udx[q]] == 19):
                            vranges = np.append(vranges, myspec.vrange)
                            myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[udx[q]]))
                            if os.path.exists(linfile): # Read in the model from file...
                                if verbose: print("Reading in model from ",linfile)
                                mymodel = fitmodel(np.array([]), np.array([]), np.array([]), np.array([]), 0, np.array([]), np.array([]), np.array([]), np.array([]), myatoms)
                                mymodel.readmod(linfile)
                            else: # Do the fit
                                if verbose: print("Getting NMF projection...")
                                sdss.W[udx[q],:] = sdss.transformspec(myspec)
                                if verbose: print("Starting spectral fit...")
                                mymodel = myspec.fitspec(sdss,box1,box2,absonly,myatoms)
                                mymodel.writemod(linfile)
                            mymodel.getsystems(myspec,sdss.qnum[udx[q]])
                            with open("vranges.dat", "w") as f:
                                for v in vranges:
                                    f.write(str(v)+"\n")


if firsttpcf:
    if False:
        sdss.firsttpcf('firsttpcfcnts2.dat',snrlim)
    if False:
        (chisq, fit) = sdss.firstfit('firsttpcfcnts.dat')
        print(chisq,fit)

    dcorr = 200
    sdss.matchradio(dcorr)
#        chisq,c0,c2,r0,alpha = sdss.firstsrcsingle(1000)

print("Finished all processes in "+str(time.time()-t0)+" seconds")


####################################################################
if False: # This is just for trying stuff on individual objects
    t1 = time.time()
    sdss.readcomp(compfile, ncomp)
    if verbose: print("Finished reading NMF components in "+str(time.time()-t1)+" seconds")
    uniqarray, udx = np.unique(sdss.qnum, return_index=True)
    for q in np.extract(sdss.qnum[udx] > -1, udx):
#        if q > 24167:
        if q > 19:
            datfile = basedir+'/ASCII/' + sdss.fileroot[q] + '.dat'
            if (os.path.exists(datfile)) and (np.all(sdss.qnum[:q] != sdss.qnum[q])):
                if verbose: print('Reading in ',datfile)
                myspec = spectrum(datfile,sdss.qnum[q],R)
                myatoms = atomic(myspec.wavelength[0],myspec.wavelength[-1]*(1+sdss.zqso[q]))
                vvv = np.fabs(calcvel(myspec.wavelength,myatoms.wave[myatoms.getspecies(7,5)]))
                if (myspec.vrange >= 10651) and (myspec.fpx.size > 0) and (sdss.snr[q] > snrlim) and (np.min(vvv) < 200):
                    if verbose: print("Getting NMF projection...")
                    sdss.W[q,:] = sdss.transformspec(myspec)
                    vdx,nnflux,nnferr,ew,sew,bigew,bigsew = myspec.normspec(sdss,box1,box2)
                    nciva,nnva,nnvciva = myspec.dosearch(sdss,q,box1,box2,verbose,myatoms,basedir)

#                    vv = calcvelzz(z,sdss.zqso[q])
#                    vvdx = np.extract((vv > -13000) & (vv < -12500), range(vv.size))
#                    for vdx in vvdx:
#                        print(vv[vdx],bephi[vdx]/sigbephi[vdx])

#                    wlo = 1600
#                    whi = 1640
#                    whi = 1525
#                    plt.subplot(311)
#                    plt.xlim(calcvel(np.array([wlo,whi]), myatoms.wave[(myatoms.getspecies(12,2))[0]]))
#                    plt.ylim([0.1,-0.05])
#                    plt.subplot(312)
#                    plt.xlim(calcvel(np.array([wlo,whi]), myatoms.wave[(myatoms.getspecies(6,4))[0]]))
#                    plt.ylim([0.05,-0.05])
#                    plt.subplot(313)
#                    plt.xlim([wlo,whi])
#                    plt.pause(5)
##                    plt.clf()
##                    plt.subplot(211)
##                    plt.step(myspec.wavelength[vdx],bigew,"b")
##                    (ylo,yhi) = plt.ylim()
##                    plt.ylim([yhi,ylo])
##                    plt.step(myspec.wavelength[vdx], bigsew,   "r")
##                    plt.step(myspec.wavelength[vdx],-bigsew,   "r")
##                    plt.step(myspec.wavelength[vdx], 3*bigsew, "g")
##                    plt.step(myspec.wavelength[vdx],-3*bigsew, "g")
##                    plt.title(("qnum {} {} {}").format(sdss.qnum[q],
##                                                       2*(np.extract((np.fabs(bigew)<bigsew) & (bigew < 0), bigew)).size/bigew.size,
##                                                       2*(np.extract((np.fabs(ew)<sew) & (ew < 0), ew)).size/ew.size                ))
##                    plt.subplot(212)
##                    plt.step(myspec.wavelength[vdx],myspec.flux[vdx])
##                    plt.pause(0.001)
                    print("q = ",q)
                    break




#            break

#            sdsst = sdss.calcspec(sdss.W[q,:])

##            plt.clf()
##
##            if myspec.nhdx.size > 3:
##                plt.subplot(211)
##                plt.step(myspec.wavelength[myspec.nhdx], myspec.flux[myspec.nhdx],              'b')
##                plt.step(myspec.wavelength[myspec.nhdx], np.sqrt(1.0/myspec.ivar[myspec.nhdx]), 'g')
##                wdx = np.extract((sdss.wavebins > myspec.wavelength[myspec.nhdx[0]]) & (sdss.wavebins < myspec.wavelength[myspec.nhdx[-1]]), range(sdss.wavebins.size))
##                plt.plot(sdss.wavebins[wdx],sdsst[wdx],"r")
##
##                plt.ylabel('Flux')
##            plt.title("qnum: "+str(sdss.qnum[q])+" S/N = "+str(np.round(sdss.snr[q],1)))
##
##            if myspec.cdx.size > 3:
##                plt.subplot(212)
##                plt.step(myspec.wavelength[myspec.cdx], myspec.flux[myspec.cdx],              'b')
##                plt.step(myspec.wavelength[myspec.cdx], np.sqrt(1.0/myspec.ivar[myspec.cdx]), 'b')
##                wdx = np.extract((sdss.wavebins > myspec.wavelength[myspec.cdx[0]]) & (sdss.wavebins < myspec.wavelength[myspec.cdx[-1]]), range(sdss.wavebins.size))
##                plt.plot(sdss.wavebins[wdx],sdsst[wdx],"r")
##
##                plt.ylabel('Flux')
##            plt.xlabel('Rest Wavelength [A]')
##            plt.pause(0.1)


#        input("Hit enter when ready")

##    ###############################################################################################
##if inspectbal == 1:
##    f    = open(samplefile,"r")
##    for line in f:
##        x = line.split()
##        if (x[0] != "#"):
##            qnum = int(x[0])
##            snr  = float(x[-1])
##            if qnum > -1:
##                rootname = x[1]
##                datfile = basedir+'ASCII/' + rootname + '.dat'
##                if (os.path.exists(datfile)) and (snr > snrlim) and (snr < snrulim):
##                    myspec = spectrum(datfile)
##                    plt.clf()
##                    plt.subplot(2,1,1)
##                    plt.xlim([np.min(myspec.wavelength[myspec.fpx]),np.max(myspec.wavelength[myspec.fpx])])
##                    ymax = np.max(np.extract((myspec.wavelength > 1200) & (myspec.wavelength < 1300), myspec.flux))
##                    plt.ylim([-0.2*ymax, 1.1*ymax])
##                    plt.step(myspec.wavelength, myspec.flux,              'k')
##                    plt.step(myspec.wavelength, np.sqrt(1.0/myspec.ivar), 'k')
##                    plt.xlabel('Rest Wavelength [A]')
##                    plt.ylabel('Flux')
##                    plt.title("qnum: "+str(qnum)+" S/N = "+str(np.round(snr,1)))
##                    plt.subplot(2,1,2)
##                    spf = np.interp(myspec.nvbvel[myspec.nhdx], myspec.civbvel, myspec.flux)
##                    plt.scatter(myspec.flux[myspec.nhdx],spf, s=0.5)
##                    plt.pause(0.1)
##                    print(qnum, snr, rootname)
##                    input("Hit enter to continue")


##    ###############################################################################################


#ncts         = np.zeros(modvel.size)
#ccts         = np.zeros(modvel.size)
#lgnvfluxave  = np.zeros(modvel.size)
#lgcivfluxave = np.zeros(modvel.size)
#f    = open(samplefile,"r")
#for line in f:
#    x = line.split()
#    if (x[0] != "#") and (float(x[-1]) > snrlim):
#        qnum = int(x[0])
#        snr  = float(x[-1])
#        if qnum > -1:
#            rootname = x[1]
#            datfile = basedir+'ASCII/' + rootname + '.dat'
#            linfile = 'D:/ganguly/SDSSAAL/dr18/Linelists/' + rootname + '.lin'
#            if (os.path.exists(datfile)): # and (not(os.path.exists(linfile))):
#                myspec = spectrum(datfile)
#                print('\n', datfile, myspec.vrange)
#                modvel = np.linspace(-myspec.vrange,myspec.vrange,num=int(2*myspec.vrange)) # dv= 1 km/s
#                lgflux = np.log10(np.where(myspec.flux[myspec.cdx] < 1.0/np.sqrt(myspec.ivar[myspec.cdx]), 1.0/np.sqrt(myspec.ivar[myspec.cdx]), myspec.flux[myspec.cdx]))
#                spl = CubicSpline(myspec.civbvel[myspec.cdx], lgflux)
#                hdx = np.extract((modvel > np.min(myspec.civbvel[myspec.cdx])) & (modvel < np.max(myspec.civbvel[myspec.cdx])), range(modvel.size))
#                lgcivfluxave[hdx] = np.multiply(lgcivfluxave[hdx], ccts[hdx]) + spl(modvel[hdx])
#                ccts[hdx] += 1
#                lgcivfluxave[hdx] = np.multiply(lgcivfluxave[hdx], 1/ccts[hdx])

##                vref = np.min(np.extract(ncts ==     np.max(ncts), modvel))
##                v90  = np.min(np.extract(ncts >  0.9*np.max(ncts), modvel))

#                plt.clf()
#                plt.subplot(211)
#                plt.step(modvel, np.power(10.0, lgnvfluxave))
#                plt.xlim([-30000,30000])
#                plt.title(str(int(np.max(ncts)))+" "+str(qnum)+" "+str(snr))

#                plt.subplot(212)
#                plt.step(modvel, np.power(10.0, lgcivfluxave))
#                plt.xlim([-30000,30000])

#                plt.subplot(313)
#                plt.step(modvel,mcts/np.max(mcts))
#                plt.plot([vref,vref], [0,1])
#                plt.plot([v90,v90], [0,0.9])
#                plt.text(vref, 0.05, str(int(vref)))
#                plt.text(v90,  0.0, str(int(v90)))
#                plt.pause(0.001)
#                cdx    = np.squeeze(np.argwhere((modvel>-200.0) & (modvel<200.0)))
#                if cdx.size % 2 == 0:
#                    cdx = np.append(cdx, cdx[cdx.size-1]+1)
#                modw = 1300.0 * ( (1.0 + (modvel[cdx]/299792.458))/(1.0 - (modvel[cdx]/299792.458)) - 1.0) / 2.0
#                lsf  = np.exp(-0.5*np.square(modw/dwres))
#                tq = time.process_time()
#                niterate = 0
#                mymodel = myspec.fitspec()
#                mymodel.writemod(linfile)
#        break
#f.close()


##    ###############################################################################################
##
##    def change(self,spec):
##        da      = 1.0*(2.0*rng.random(self.nha.size)-1.0)
##        if self.nha[0] + da[0] < 0:
##            da[0] = 0
##            self.nha[0] = np.fabs(self.nha[0])
##        self.nha += da
##
##        da      = 1.0*(2.0*rng.random(self.ca.size)-1.0)
##        if self.ca[0] + da[0] < 0:
##            da[0] = 0
##            self.ca[0] = np.fabs(self.ca[0])
##        self.ca += da
##
##        if self.nabs > 0:
##            self.tabs  = np.fabs(self.tabs)
##            dtabs      = 0.2 * self.tabs * (2.0 * rng.random((4, self.nabs)) - 1.0) # max 20% change
##            dvabs      = 5.0 *             (2.0 * rng.random(self.nabs)      - 1.0) # max 5.0 km/s
##            dkts       = 0.3 * self.kts  * (2.0 * rng.random(self.nabs) - 1.0)      # max 30% change
##            dbnts      = 0.1 * self.bnts * (2.0 * rng.random(self.nabs) - 1.0)      # max 10% change
##            for i in range(self.nabs):
##                if (self.vabs[i] < np.min(spec.hivel)): # H I is not covered
##                    dtabs[1, i] = -self.tabs[1, i]
##                if (self.vabs[i] < np.min(spec.nvbvel)): # N V is not covered
##                    dtabs[0, i] = -self.tabs[0, i]
##                if (self.vabs[i] < np.min(spec.civbvel)): # C IV is not covered
##                    dtabs[2, i] = -self.tabs[2, i]
##                if self.vabs[i] > np.max(spec.mgiibvel[spec.cdx]): # Mg II is not covered
##                    dtabs[3, i] = -self.tabs[3, i]
##            self.tabs += dtabs
##            self.vabs += dvabs
##            self.kts  += dkts
##            self.bnts += dbnts
##



"""