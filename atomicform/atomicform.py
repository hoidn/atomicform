from __future__ import with_statement
from __future__ import division
#from __future__ import absolute_import

import numpy as np
import csv
import operator
import collections
from io import StringIO
import re
import pkgutil
from utils import utils
from io import open
from itertools import imap
from itertools import izip
import pandas as pd

PKG_NAME = __name__.split(u'.')[0]

#output of ElementData[#, "ElectronConfiguration"] & /@ Range[100] in 
#mathematica, i.e. electron configurations of the first 100 elements
s = open(utils.resource_path(u"data/configurations.txt", pkg_name = PKG_NAME), u'r').read()
#s = open("../data/configurations.txt", 'r').read()
 
#remove MMA list delimiters
s2 =s.replace(u"{",u"").replace(u"}", u"").replace(u"\r", u"") 
 
#convert from string representation of an int list of depth 1
toIntList = lambda x: list(imap(int, x.split(u","))) 
 
#list representing electron configurations  indexed by z, n, l
eConfigs = list([list(imap(toIntList, x.split(u"\t"))) for x in s2.split(u'\n')])

nOrbitals = [sum(list(imap(len, elt))) for elt in eConfigs]


u"""
compute the approximate atomic form factors of various atoms and ions 
using tabulated values of the fit coefficients
"""
#file with the tabulated coefficients
tableFile = u'data/all_atomic_ff_coeffs.txt'

#junk characters in the data file to get rid of
deleteChars = '\xe2\x80\x83'

hklList = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [3, 0, 0], [3, 1, 0], [3, 1, 1], [2, 2, 2]]

#positions of atoms of the two species in the rock salt structure
positions1 = ((0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5))
positions2 = ((.5, .5, .5), (.5, 0, 0), (0, .5, 0), (0, 0, 0.5))


#bond locations for the unit cell based on the above positions. Note the 
#coordination number is 6, so there are 24 of these per unit cell
bondingLocations = [[0.5, 0.75, 0.5], [0.5, 0.25, 0.0], [0.0, 0.75, 0.0], [0.0, 0.25, 0.5], [0.5, 0.5, 0.75], [0.5, 0.0, 0.25], [0.0, 0.5, 0.25], [0.0, 0.0, 0.75], [0.75, 0.5, 0.5], [0.75, 0.0, 0.0], [0.25, 0.5, 0.0], [0.25, 0.0, 0.5], [0.25, 0.0, 0.0], [0.75, 0.5, 0.0], [0.75, 0.0, 0.5], [0.25, 0.5, 0.5], [0.0, 0.25, 0.0], [0.5, 0.75, 0.0], [0.5, 0.25, 0.5], [0.0, 0.75, 0.5], [0.0, 0.0, 0.25], [0.5, 0.5, 0.25], [0.5, 0.0, 0.75], [0.0, 0.5, 0.75]]

def cleanStr(chars, s):
    u"""delete all occurences of the characters in the string chars from the
       string s
    """
    r = re.compile(chars)
    return re.sub(r, '', s)


#rawStr = cleanStr(deleteChars, utils.resource_path(tableFile))
#rawStr = cleanStr(deleteChars, pkgutil.get_data('atomicform', tableFile))
#rawStr = cleanStr(deleteChars, open(tableFile, 'r').read())


table = pd.read_csv('atomicform/data/all_atomic_ff_coeffs.txt', delimiter = '\t', header=None)
table[0] = table[0].str.strip()
table.index = table[0]
table = table.drop(0, 1)

#rawTable = np.genfromtxt(utils.resource_path(tableFile, pkg_name = PKG_NAME), dtype=('S20', float, float, float, float, float, float, float, float, float), delimiter='\t')
#rawTable = np.genfromtxt(utils.resource_path(tableFile), dtype=('S20', float, float, float, float, float, float, float, float, float), delimiter='\t')

#elementKeys = rawTable[rawTable.dtype.names[0]]
#numerical values of the table
#zipped = np.array(list(izip(*rawTable)))
#elementKeys, values = zipped[0], list(izip(*zipped[1:]))
#values = np.array(values, dtype=float)

#coeffDict = dict((k, v) for (k, v) in izip(elementKeys, values))

def getFofQ(k): 
    u"""
    return function that evaluates atomic form factor corresponding to the
    element/ion key, based on tabulated approximations.
    """

    try:
        _, a1, b1, a2, b2, a3, b3, a4, b4, c = table.loc[k]
    except KeyError:
        raise ValueError("invalid key: %s" % k)
        
    def FofQ(q):
        singleQ = lambda x :  a1 * np.exp(-b1 * (x/(4 * np.pi))**2)  +\
             a2 * np.exp(-b2 * (x/(4 * np.pi))**2)  + \
             a3 * np.exp(-b3 * (x/(4 * np.pi))**2)  + \
             a4 * np.exp(-b4 * (x/(4 * np.pi))**2) + c
        if isinstance(q, collections.Iterable): 
            return np.array(list(imap(singleQ, q)))
        else:
            return singleQ(q)
    return FofQ
    

def getPhase(rr, qq): 
    u"""evaluate exp(2 pi i q . r)"""
    return np.exp(2j * np.pi * np.dot(qq, rr)) 

def validCheck(qvec):
    u"""
    helper function to check and massage, if needed, an input array 
    of momentum transfers
    """
    errmsg  = u"Momentum transfer must be represented by an array of length three"
    try: 
        if depth(qvec) == 1:
            qvec = [qvec]
        q11 = qvec[0][0]
    except:
        raise ValueError(errmsg)
    else: 
        if  len(qvec[0]) != 3:
            raise ValueError(errmsg)
    return qvec

def gaussianCloud(charge, x, y, z, sigma):
    u"""
    return function that evaluates the scattering amplitude of a 3d gaussian 
    charge distribution with standard deviation sigma, total charge charge, and
    centered at (x, y, z)
    """
    #the function that takes in a q vector and returns the amplitude
    tomap = lambda qq: charge * getPhase((x, y, z), qq) * \
            np.exp(- 0.5 * sigma**2 * np.dot(qq, qq))
    def amplitude(qvec):
        qvec = validCheck(qvec)
        return np.array(list(imap(tomap, qvec)))
    return amplitude

def fccStruct(a1, a2 = None, latticeConst = 1):
    u"""
    return function that evaluates the unit cell structure factor of 
    of an fcc material with two distinct species.
    The function expects the momentum transfer vector to be expressed in 
    terms of the reciprocal lattice basis vectors. 

    a1, a2: key strings for the elements
    a2 == None corresponds to an fcc structure with a one-atom basis, 
    rather than rock salt structure
    """
    part1 = structFact(a1, positions1, latticeConst = latticeConst)
    if a2 is None:
        return part1
    else:
        part2 = structFact(a2, positions2, latticeConst = latticeConst)
        return lambda x: part1(x) + part2(x)

def structFact(species, positions, latticeConst = 1):
    if latticeConst ==1:
        print u"Warning: defaulting to 1 Angstrom for lattice constant"
    reciprocalLatticeConst = 2 * np.pi / latticeConst
    #form factor of a single species
    #q_hkl is momentum transfer magnitude in units of the 
    #reciprocal lattice constant
    f = lambda q_hkl: getFofQ(species)(np.array(q_hkl) * reciprocalLatticeConst)
    #function to evaluate amplitude contribution of a single atom
    def oneatom(formfactor, positions):
        return lambda qq: getPhase(positions, qq) * formfactor(list(imap(np.linalg.norm, qq)))
    #function to evaluate total amplitude for this strucure
    def amplitude(qvec):
        qvec = validCheck(qvec)
        return np.sum( np.array([oneatom(f, x)(qvec) for x in positions]), axis = 0) 
    return amplitude


def heat(qTransfer, structfact, donor = u'F', sigma = 0.05, latticeConst = 1):
    u"""
    qTransfer: amount of charge to move
    structFac: structure factor functions. 

    the locations of the donor species are assumed to be positions2
    """
    perBond = float(qTransfer)/len(bondingLocations)
    gaussians = [gaussianCloud(perBond, x[0], x[1], x[2], sigma) for x in bondingLocations]
    donors = structFact(donor, positions2, latticeConst = latticeConst)
    donorCharge = donors([0, 0, 0])
    scale = float(qTransfer)/donorCharge
    return  (lambda x: (structfact(x) - scale * donors(x) + np.sum(np.array([gaussian(x) for gaussian in gaussians]), axis = 0)))
    
    
    
def normHKLs(charges, alkali = u'Li', halide = None, hkls = hklList, mode = u'amplitude', latticeConst = 1):
    baseline = fccStruct(alkali, halide, latticeConst = latticeConst)
    if mode ==u'amplitude':
        mapFunc = lambda x: round(abs(x), 2)
    elif mode ==u'intensity':
        mapFunc = lambda x: round(abs(x)**2, 2)
    formFactors = np.array([list(imap(mapFunc, heat(z, baseline, latticeConst = latticeConst)(hkls))) for z in charges])
    #normTable = lambda x: np.array(map(lambda y: abs(y), x))
    #return map(normTable, formFactors)
    return formFactors

def tableForm(charges, alkali = u'Li', halide = None, hkls = hklList, extracol = None, mode = u'amplitude', latticeConst = 1):
    if halide is None:
        print u"computing scattering amplitudes for fcc structure with single-atom basis"
    f_hkls = normHKLs(charges, alkali = alkali, halide = halide, hkls = hkls, mode = mode, latticeConst = latticeConst)
    hklstrlist = hklString(hkls)
    if extracol != None:
        return np.array(list(izip(*np.vstack((hklstrlist, f_hkls, extracol)))))
    else: 
        return np.array(list(izip(*np.vstack((hklstrlist, f_hkls)))))

#def hklPermutations(hmax):
#    ipdb.set_trace()
#    return _hklPermutations(0, 0, 0, hmax, [[0, 0, 0]])

#def _hklPermutations(h, k, l, hmax, acc):
#    if l < k:
#        l += 1
#        return _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    elif k < h: 
#        k += 1
#        return _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    elif h <= hmax: 
#        h += 1
#        return  _hklPermutations(h, k, l, hmax, acc + [[h, k, l]])
#    else:
#        return acc
#
        

def FCChkl(maxh, complement = False):
    u"""allowed fcc reflections, up to maxh maxh maxh"""
    outlist = []
    for i in xrange(maxh + 1):
        for j in xrange(i + 1):
            for k in xrange(j + 1):
                #allowed reflections: h, k, l all even or all odd
                if not complement: 
                    if (i%2 == 0 and j%2 == 0 and k%2 ==0 ) or \
                            (i%2 == 1 and j%2 == 1 and k%2 ==1 ):
                        outlist += [[i, j, k]]
                else: 
                    if not ((i%2 == 0 and j%2 == 0 and k%2 ==0 ) or \
                            (i%2 == 1 and j%2 == 1 and k%2 ==1 )):
                        outlist += [[i, j, k]]
    return outlist

def hklString(hkl):
    u"""string representation of list of three integers"""
    def stringify(x):
        hklstr = list(imap(unicode, x))
        return hklstr[0] + u';' + hklstr[1]  + u';' + hklstr[2]
    try: 
        hklStringList = list(imap(stringify, hkl))
    except: 
        return stringify(hkl)
    else:
        return hklStringList 

def sortHKL(hkllist):
    u"""sort a list of hkls by momentum transfer"""
    def qTrans(hkl):
        return sum([x**2 for x in hkl])
    return sorted(hkllist, key = qTrans)

def depth(l):
    u"""evaluate maximum depth of a list or numpy array"""
    if isinstance(l, (np.ndarray, list)):
        return 1 + max(depth(item) for item in l)
    else:
        return 0

def csvWrite(fname, arr):
    with open(fname, u'w') as fp:
        a = csv.writer(fp, delimiter=u',')
        a.writerows(arr)
        fp.close()
