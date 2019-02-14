import numpy
from numpy import array
import scipy.stats
import pymc

numpy.random.seed(42)

#NGENES = 25
NFP = 230
NTP = 20
NGENES = NFP + NTP

nlibs = array([2, 4])
#libidxs = map(lambda x: slice(x[0], x[1]), zip(numpy.r_[0, nlibs.cumsum()[:-1]], nlibs.cumsum()))
libidxs = numpy.repeat(numpy.arange(len(nlibs)), nlibs)

fbase = array([0.25, 0.05, 0.65, 0.05])
conc = scipy.stats.gamma.rvs(1. / (NGENES / 250.), scale=1. / 250., size=NGENES)

mu_libsize = NGENES * 250
var_libsize = mu_libsize**1.75
libsizes = numpy.random.negative_binomial((mu_libsize ** 2 / var_libsize) / (1 - (mu_libsize / var_libsize)), mu_libsize / var_libsize, size=sum(nlibs))

gamma = ((fbase * conc[:, numpy.newaxis]) * libsizes[:, numpy.newaxis, numpy.newaxis])

# insert true positives
for i in xrange(NFP, NFP + NTP):
    #fbase_tp_0 = numpy.random.dirichlet([1.0] * 4)
    #fbase_tp_1 = numpy.random.dirichlet([1.0] * 4)
    if (i - NFP) < (NTP / 2):
        fbase_tp_0 = numpy.random.dirichlet([5., 2., 8., 1.])
        fbase_tp_1 = numpy.random.dirichlet([8., 2., 5., 1.])
    else:
        fbase_tp_0 = numpy.random.dirichlet([8., 2., 5., 1.])
        fbase_tp_1 = numpy.random.dirichlet([5., 2., 8., 1.])

    gamma[libidxs == 0, i] = (fbase_tp_0 * conc[i]) * libsizes[libidxs == 0][:, numpy.newaxis]
    gamma[libidxs == 1, i] = (fbase_tp_1 * conc[i]) * libsizes[libidxs == 1][:, numpy.newaxis]

var = gamma + 0.3 * gamma + 0.15 * gamma ** 2

p = gamma / var
n = (gamma * p) / (1 - p)
d = numpy.random.negative_binomial(n, p)

w = d.sum(axis=2).mean(axis=0) >= 50
print(w.sum())

libsizes_hat = d.sum(axis=2).sum(axis=1)


## normalize data
nd = (d.T / libsizes_hat.astype(float)).T

g = numpy.array([nd[libidxs == i].mean(axis=0) for i in xrange(libidxs.max() + 1)])
## impute zeros
m = numpy.ma.masked_where(g==0, g).min(axis=0).min(axis=0).data / 2
for i in xrange(g.shape[2]):
    mask = g[:, :, i] == 0
    numpy.place(g[:, :, i], mask, m[i])

mu = (g[libidxs].T * libsizes_hat).T


# pt I
# estimate overdispersion

# var == mu + t0 * mu + t1 * mu ** 2
theta0 = pymc.Exponential("theta0", 1. / 10, trace=True)
theta1 = pymc.Exponential("theta1", 1. / 10, trace=True)
alpha = mu / (mu * theta1 + theta0)
dnb = pymc.NegativeBinomial("dnb", mu, alpha, observed=True, value=d)

# manual init.
theta0.value = 0.5
theta1.value = 0.5

inp = (dnb, alpha, theta0, theta1)
mcmc = pymc.MCMC(input=inp)
NBURN = 20000
mcmc.sample(25000 + NBURN, NBURN, thin=10)
print("")

#t0 = numpy.median(theta0.trace()[:])
#t1 = numpy.median(theta1.trace()[:])
t0 = scipy.stats.mstats.mquantiles(theta0.trace(), [0.667])[0]
t1 = scipy.stats.mstats.mquantiles(theta1.trace(), [0.667])[0]

## or just use NNLS...
## (which seems very unrobust and mostly focussed on the highly abundant outliers)
#m = numpy.r_[d[libidxs == 0].mean(axis=0).flatten(), d[libidxs == 1].mean(axis=0).flatten()]
#v = numpy.r_[d[libidxs == 0].var(axis=0).flatten(), d[libidxs == 1].var(axis=0).flatten()]
#t1, t0 = scipy.optimize.nnls(numpy.column_stack((m ** 2, m)), v - m)[0]

#m = numpy.r_[d[libidxs == 0].mean(axis=0).flatten(), d[libidxs == 1].mean(axis=0).flatten()]
#v = numpy.r_[d[libidxs == 0].var(axis=0).flatten(), d[libidxs == 1].var(axis=0).flatten()]
#qlow, qhigh = scipy.stats.mstats.mquantiles(m, [0.25, 0.75])
#w = (m > qlow) & (m < qhigh) & (m > 0)
#t1, t0 = scipy.optimize.nnls(numpy.column_stack(((m ** 2)[w], m[w])), (v - m)[w])[0]

print("estimates: theta0: %.4f\ntheta1: %.4f" % (t0, t1))


# pt. II
# use overdispersion and run analysis per gene

NSAMPLES = 2500
NBURN = 20000
NTHIN = 10
traces = dict()
traces["y"] = numpy.empty((d.shape[1], 2, NSAMPLES))
traces["dr"] = numpy.empty((d.shape[1], 2, NSAMPLES, 4))
#ghat = g.sum(axis=2)

for i in xrange(d.shape[1]):
    print(i)
    dr = [pymc.Dirichlet("dir_%d" % j, theta = numpy.repeat(1.0, 4), trace=True) for j in xrange(2)]
    cdr = [pymc.CompletedDirichlet("cdir_%d" % j, dr[j]) for j in xrange(2)]
    y = [pymc.Exponential("g_%d" % j, 0.0001, trace=True) for j in xrange(2)]
    #y = ghat[:, i]
    mu = [pymc.Lambda("mu_%d" % j, lambda y=y[j], d=cdr[j], L=libsizes_hat[libidxs == j]: ((y * d.T) * L).T, trace=False) for j in xrange(2)]
    
    alpha = [(mu[j] / (mu[j] * t1 + t0)) for j in xrange(2)]
    dnb = [pymc.NegativeBinomial("d_%d" % j, mu[j], alpha[j], observed=True, value=d[libidxs == j, i], trace=False) for j in xrange(2)]

    # starting points
    for j in xrange(2):
        dr[j].value = numpy.repeat(0.25, 3)
        y[j].value = g[j, i].sum()

    inp = (dnb, alpha, mu, y, cdr, dr)
    mcmc = pymc.MCMC(input=inp)
    mcmc.sample(NSAMPLES * NTHIN + NBURN, NBURN, thin=NTHIN)
    print("")

    for j in xrange(2):
        traces["y"][i][j] = y[j].trace()[:]
        traces["dr"][i][j] = cdr[j].trace()[:].squeeze(1)

ALPHA = 0.1
hpds = pymc.utils.hpd(traces["dr"].swapaxes(2, 0).swapaxes(2, 1), ALPHA)
s_up = (hpds[:, 1, :, 0] - 0.05) > hpds[:, 0, :, 1]
s_down = (hpds[:, 1, :, 1] + 0.05) < hpds[:, 0, :, 0]

# total false positives
print(s_up.sum())
print(s_down.sum())

# genewise false positives
print((s_up.sum(axis=1) != 0).sum())
print((s_down.sum(axis=1) != 0).sum())


# scoring
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

y_true = numpy.r_[numpy.repeat(False, NFP), numpy.repeat(True, NTP)]
ALPHAS = numpy.arange(0.01, 0.5, 0.01)
scores = numpy.empty(ALPHAS.size)
for ialpha, alpha in enumerate(ALPHAS):
    hpds = pymc.utils.hpd(traces["dr"].swapaxes(2, 0).swapaxes(2, 1), alpha)
    s_up = (hpds[:, 1, :, 0] - 0.05) > hpds[:, 0, :, 1]
    s_down = (hpds[:, 1, :, 1] + 0.05) < hpds[:, 0, :, 0]

    #scores[ialpha] = roc_auc_score(y_true, (s_up.sum(axis=1) != 0) | (s_down.sum(axis=1) != 0))
    scores[ialpha] = precision_score(y_true, (s_up.sum(axis=1) != 0) | (s_down.sum(axis=1) != 0))
