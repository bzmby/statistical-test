import numpy
from numpy import array, zeros, empty

import pymc
import h5py


h5file = h5py.File("./output/KR20140911.2625_autophagy_torin.counts.hdf5", 'r')
samplelabels = sorted(h5file["counts"].keys())

MINREADS = 100

anno = {"ctrl": [0, 3], "torin": [1, 4], "autophagy": [2, 5]}
annolabels = ["ctrl", "torin", "autophagy"]
annoidx = dict(((x[1], x[0]) for x in enumerate(annolabels)))

d = array(map(lambda s: h5file["counts"][s][:], samplelabels))

ws = map(lambda idx: d[idx].sum(axis=2).sum(axis=0) >= MINREADS, map(lambda k: anno[k], annolabels))

ctrl = "ctrl"
cond = "torin"

w = ws[annoidx[ctrl]] & ws[annoidx[cond]]

ds = d[:, w, :]
#hits = zeros((ds.shape[1], 4, 2), dtype="bool")
hpds = empty((ds.shape[1], 4, 2), dtype="float32")

#for igene in xrange(ds.shape[1]):
for igene in xrange(500):
    if igene % 25 == 0:
        print(igene)

    dir_1 = pymc.Dirichlet("dir_1", theta = numpy.repeat(1.0, d.shape[2]), trace=True)
    dir_2 = pymc.Dirichlet("dir_2", theta = numpy.repeat(1.0, d.shape[2]), trace=True)

    ddiff = pymc.Lambda("ddiff", 
                        lambda dir_1=pymc.CompletedDirichlet("cdir_1", dir_1, trace=False), 
                            dir_2=pymc.CompletedDirichlet("cdir_2", dir_2, trace=False): 
                                    dir_2[0] - dir_1[0], trace=True)

    vals_1 = ds[anno[ctrl], igene]
    vals_2 = ds[anno[cond], igene]

    mn_1 = pymc.Multinomial("mn_1", value=vals_1, n=vals_1.sum(axis=1), p=dir_1, observed=True, trace=False)
    mn_2 = pymc.Multinomial("mn_2", value=vals_2, n=vals_2.sum(axis=1), p=dir_2, observed=True, trace=False)

    #mp = pymc.MAP([dir_1, dir_2, mn_1, mn_2])
    #mp.fit()
    dir_1.value = array([0.25] * 3)
    dir_2.value = array([0.25] * 3)

    mcmc = pymc.MCMC([dir_1, dir_2, mn_1, mn_2, ddiff])
    mcmc.sample(iter=6000, burn=1000, thin=5, progress_bar=False)

    hpd = pymc.utils.hpd(ddiff.trace()[:], 0.1)
    hpds[igene] = hpd

    #for j in numpy.where((hpd > 0.05).all(axis=1))[0]:
    #    hits[igene, j, 1] = True

    #for j in numpy.where((hpd < -0.05).all(axis=1))[0]:
    #    hits[igene, j, 0] = True
