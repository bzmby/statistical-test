import os
import glob

import numpy
from numpy import array
import pandas
import pymc

RS = numpy.random.RandomState()
RS.seed(42)

files = sorted(glob.glob("./output/feature_count_tables/*.tsv"))

ds = map(lambda fn: pandas.read_table(fn, sep="\t", comment="#").dropna(), files)

conds = map(lambda s: s.split(".")[0], map(lambda l: l[2], map(lambda s: s.split("_"), map(os.path.basename, files))))
conds = dict([(x, [i for (i, y) in enumerate(conds) if y == x]) for x in set(conds)])
conds = sorted(list(conds.items()))

d = numpy.array(map(lambda d_: d_[["fputr", "cds"]].values.astype("uint32"), ds))
nf = d.sum(axis=2).sum(axis=1) / 1e6
EP = numpy.power(10, numpy.round(numpy.log10(nf.mean() / d.shape[1] / 10)))

d = d[:, (d.sum(axis=2) >= 100).any(axis=0), :]
d = d[:, (d >= 10).all(axis=2).any(axis=0), :]

#NSUBSAMPLE = 15
#wr = RS.permutation(numpy.arange(d.shape[1]))[:NSUBSAMPLE]
#d = d[:, wr, :]
NGENES = 250
d = d[:, 0:NGENES, :]

Fmc = numpy.empty(shape=(len(conds), d.shape[1]), dtype="O")
for i in xrange(Fmc.shape[0]):
    for j in xrange(Fmc.shape[1]):
        #Fmc[i, j] = pymc.Uniform("Fmc_{%d,%d}" % (i, j), 0.0001, 1 - 0.0001, trace=False)
        Fmc[i, j] = pymc.Uniform("Fmc_{%d,%d}" % (i, j), 0.0001, 1 - 0.0001, trace=False)
Fmc = pymc.ArrayContainer(Fmc)

alphamc = pymc.Gamma("alphamc", 0.001, 0.001)

alphamc.value = 1.0
for i in xrange(Fmc.shape[0]):
    for j in xrange(Fmc.shape[1]):
        Fmc[i, j].value = 0.5

Cm = numpy.empty(shape=Fmc.shape, dtype="O")
for i in xrange(Cm.shape[0]):
    for j in xrange(Cm.shape[1]):
        Cm[i, j] = pymc.Exponential("Cm_{%d,%d}" % (i, j), 0.001, trace=False)
Cm = pymc.ArrayContainer(Cm)

Cmc = numpy.empty(shape=Fmc.shape + (2, ), dtype="O")
for i in xrange(Cmc.shape[0]):
    for j in xrange(Cmc.shape[1]):
        condidx = conds[i][1]
        cg = d[condidx, j]
        for k in xrange(Cmc.shape[2]):
            if k == 0:
                Cmc[i, j, k] = pymc.NegativeBinomial("Cmc_{%d,%d,%d}" % (i, j, k), mu=Cm[i, j] * Fmc[i, j] * nf[condidx], alpha=alphamc, value=cg[:, k], observed=True, trace=False)
            else:
                assert k == 1
                Cmc[i, j, k] = pymc.NegativeBinomial("Cmc_{%d,%d,%d}" % (i, j, k), mu=Cm[i, j] * (1 - Fmc[i, j]) * nf[condidx], alpha=alphamc, value=cg[:, k], observed=True, trace=False)
Cmc = pymc.ArrayContainer(Cmc)

# interested in cond Cy73 vs others
ref_cond = [x[0] for x in conds].index("Cy73")
other_cond = range(len(conds))
other_cond.remove(ref_cond)
Fshift = numpy.empty(d.shape[1], dtype="O")
for i in xrange(Fshift.shape[0]):
    #Fshift[i] = pymc.Lambda("Fshift_{%d}" % (i,), lambda fref=Fmc.value[ref_cond, i].mean(axis=0).tolist(), f=Fmc.value[other_cond, i].mean(axis=0).tolist(): f-fref)
    #Fshift[i] = pymc.Lambda("Fshift_{%d}" % (i,), lambda fall=Fmc, i=i: float(fall[other_cond, i].mean() - fall[ref_cond, i].mean()))
    Fshift[i] = pymc.Lambda("Fshift_{%d}" % (i,), lambda fall=Fmc, i=i: float((fall[other_cond, i].mean() - fall[ref_cond, i].mean()) / max(0.2, float(fall[ref_cond, i].std()))))
Fshift = pymc.ArrayContainer(Fshift)

numpy.random.seed(42)  # set random seed for pymc

inp = (alphamc, Fmc, Fshift, Cm, Cmc)
#N = pymc.MAP(input=inp)
#N.fit()

#M = pymc.MCMC(input=inp, db="hdf5", dbname="./output/mcmc/traces/shift_20131129.hdf5", dbcomplevel=9, dbcomplib="bzip2")
M = pymc.MCMC(input=inp, db="sqlite", dbname="./output/mcmc/traces/shift_20131129.sqlite")
Nthin = 5
Nsample = 20000 * Nthin
Nburn = 2500 * Nthin
Nchains = 10

for i in xrange(Nchains):
    N = pymc.MAP(input=inp)
    N.fit()
    M.sample(Nsample + Nburn, burn=Nburn, thin=Nthin)

M.db.close()

#import scipy
#[(i, scipy.stats.mstats.mquantiles(Fshift[i].trace()[:], [0.25, 0.75])) for i in xrange(len(Fshift))]

#from matplotlib import pyplot
#pyplot.hist(Fshift[2].trace()[:], alpha=0.66, color="red", normed=True)
#pyplot.show()
#pyplot.close("all")
#
#pyplot.hist(Fmc[0, 2].trace()[:], alpha=0.66, color="blue", normed=True)
#pyplot.hist(Fmc[1, 2].trace()[:], alpha=0.66, color="orange", normed=True)
#pyplot.hist(Fmc[2, 2].trace()[:], alpha=0.66, color="red", normed=True)
#pyplot.show()

# analysis of hits
#filter(lambda x: x[1], [(i, scipy.stats.mstats.mquantiles(Fshift[i].trace()[:], [0.01, 0.99]).searchsorted(0) != 1) for i in xrange(len(Fshift))])
# per hit:
#[d[condidx, 164, :] for condidx in [x[1] for x in conds]]  # get raw counts
#map(lambda x: (x[:, 0] / x.sum(axis=1).astype(float)).mean(), [d[condidx, 164, :] for condidx in [x[1] for x in conds]]) # get Fhats
#ds[0].ix[numpy.arange(ds[0].shape[0])[dcutoff1][dcutoff2][wr[164]]]  # get gene id data
