import os
import numpy             as np

from astropy               import units as u
from astropy.table         import Table

class atomic:
    ###############################################################################################

    def __init__(self,basedir,
                 wlo,
                 whi,
                 minlox=8.5,
                 ntabs = 0):
        datfile = "atoms4.dat"
        self.minlox = minlox
        if (os.path.exists(basedir+"/"+datfile)):
            data = Table.read(basedir+"/"+datfile,format='ascii')
            self.species = np.copy(data['col1'])
            self.wave    = np.copy(data['col2']) * u.Angstrom
            self.f       = np.copy(data['col3'])
            self.gamma   = np.copy(data['col4']) / u.s #units????
            self.ip      = np.copy(data['col5']) * u.eV
            self.anum    = np.copy(data['col6'])
            self.ion     = np.copy(data['col7'])
            self.abund   = np.copy(data['col8'])
            self.amass   = np.copy(data['col9']) * u.u

            self.lox = self.abund + np.log10(self.f*self.wave.value + 1.0e-30)

            ldx = np.extract((self.lox > minlox) & (self.wave > wlo) & (self.wave < whi) & (self.ip >= 13.5 * u.eV), range(self.wave.size))

            self.species = self.species[ldx]
            self.wave    = self.wave[ldx]
            self.f       = self.f[ldx]
            self.gamma   = self.gamma[ldx]
            self.ip      = self.ip[ldx]
            self.anum    = self.anum[ldx]
            self.ion     = self.ion[ldx]
            self.abund   = self.abund[ldx]
            self.amass   = self.amass[ldx]
            self.lox     = self.lox[ldx]
            self.lsdx    = np.argsort(self.lox)

            ions,idx = np.unique(100*self.anum + self.ion, return_index=True)
            self.ions = ions
            self.nion = ions.size
            self.idx = idx

            self.specstr = np.copy(self.species)
            self.plusstr = np.copy(self.species)
            for i in range(self.species.size):
                self.specstr[i] = self.atomstr(self.anum[i])+' '+self.ionstr(self.ion[i])
                plusses = ''
                if self.ion[i] == 1:
                    plusses = ''
                else:
                    plusses += '+'
                    if self.ion[i] > 2:
                        plusses += f"{self.ion[i]-1}"
                self.plusstr[i]  = self.atomstr(self.anum[i])+plusses

            self.flam = self.f * self.wave

        else:
            print("\t" * ntabs + basedir+'/'+datfile," does not exist")

        datfile = "photo.dat"
        if (os.path.exists(basedir+"/"+datfile)):
          data = Table.read(basedir+"/"+datfile,format='ascii')
          self.photo_Z     = data['col1']
          self.photo_N     = data['col2']
          self.photo_E_th  = data['col3'] * u.eV
          self.photo_E_max = data['col4'] * u.eV
          self.photo_E_0   = data['col5'] * u.eV
          self.photo_sig_0 = data['col6'] * u.Mbarn
          self.photo_y_a   = data['col7']
          self.photo_p     = data['col8']
          self.photo_y_w   = data['col9']
          self.photo_y_0   = data['col10'] * u.eV
          self.photo_y_1   = data['col11']
        else:
            print("\t" * ntabs + basedir+'/'+datfile," does not exist")

    ###############################################################################################
    def getspecies(self,anum,ion):
        return np.extract((self.anum == anum) & (self.ion == ion), range(self.anum.size))

    ###############################################################################################

    def atomstr(self,anum):
      if anum == 1:
        return 'H'
      elif anum == 2:
        return 'He'
      elif anum == 3:
        return 'Li'
      elif anum == 4:
        return 'Be'
      elif anum == 5:
        return 'B'
      elif anum == 6:
        return 'C'
      elif anum == 7:
        return 'N'
      elif anum == 8:
        return 'O'
      elif anum == 9:
        return 'F'
      elif anum == 10:
        return 'Ne'
      elif anum == 11:
        return 'Na'
      elif anum == 12:
        return 'Mg'
      elif anum == 13:
        return 'Al'
      elif anum == 14:
        return 'Si'
      elif anum == 15:
        return 'P'
      elif anum == 16:
        return 'S'
      elif anum == 17:
        return 'Cl'
      elif anum == 18:
        return 'Ar'
      elif anum == 19:
        return 'K'
      elif anum == 20:
        return 'Ca'
      elif anum == 21:
        return 'Sc'
      elif anum == 22:
        return 'Ti'
      elif anum == 23:
        return 'V'
      elif anum == 24:
        return 'Cr'
      elif anum == 25:
        return 'Mn'
      elif anum == 26:
        return 'Fe'
      elif anum == 27:
        return 'Co'
      elif anum == 28:
        return 'Ni'
      elif anum == 29:
        return 'Cu'
      elif anum == 30:
        return 'Zn'
      else:
        return 'nope'

    def ionstr(self,inum):
        if inum == 1:
            return 'I'
        elif inum == 2:
            return 'II'
        elif inum == 3:
            return 'III'
        elif inum == 4:
            return 'IV'
        elif inum == 5:
            return 'V'
        elif inum == 6:
            return 'VI'
        else:
            return 'nope'

    ###############################################################################################

    def cloudyelem(self,anum):
      if anum == 1:
        return (  "hydrogen", "hydr")
      elif anum == 2:
        return (    "helium", "heli")
      elif anum == 3:
        return (   "lithium", "lith")
      elif anum == 4:
        return ( "beryilium", "bery")
      elif anum == 5:
        return (     "boron", "boro")
      elif anum == 6:
        return (    "carbon", "carb")
      elif anum == 7:
        return (  "nitrogen", "nitr")
      elif anum == 8:
        return (    "oxygen", "oxyg")
      elif anum == 9:
        return (  "fluorine", "fluo")
      elif anum == 10:
        return (      "neon", "neon")
      elif anum == 11:
        return (    "sodium", "sodi")
      elif anum == 12:
        return ( "magnesium", "magn")
      elif anum == 13:
        return (  "aluminum", "alum")
      elif anum == 14:
        return (   "silicon", "sili")
      elif anum == 15:
        return ("phosphorus", "phos")
      elif anum == 16:
        return (   "sulpher", "sulp")
      elif anum == 17:
        return (  "chlorine", "chlo")
      elif anum == 18:
        return (     "argon", "argo")
      elif anum == 19:
        return ( "potassium", "pota")
      elif anum == 20:
        return (   "calcium", "calc")
      elif anum == 21:
        return (  "scandium", "scan")
      elif anum == 22:
        return (  "titanium", "tita")
      elif anum == 23:
        return (  "vanadium", "vana")
      elif anum == 24:
        return (  "chromium", "chro")
      elif anum == 25:
        return ( "manganese", "mang")
      elif anum == 26:
        return (      "iron", "iron")
      elif anum == 27:
        return (    "cobalt", "coba")
      elif anum == 28:
        return (    "nickel", "nick")
      elif anum == 29:
        return (    "copper", "copp")
      elif anum == 30:
        return (      "zinc", "zinc")
      else:
        return (    "nuh-uh", "nope")

###############################################################################################

