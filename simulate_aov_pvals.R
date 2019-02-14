#!/usr/bin/env Rscript

# lessons learned:
# aov with Error is _more_ conservative
# aov seems to be overconservative when testing on H0 data if Error term includes IV
# aov is sensitive to class of variables: everything that needs to be interpreted as factor _needs_ to be a factor, including logical and strings!!!

library(reshape2)
library(ggplot2)

NSIM = 5000

pvals = list()
for (i in seq(NSIM)) {
    if ((i %% 100) == 0) { print(i) }

    # H0 data
    m = data.frame(melt(
                Map(function(mu) {
                    rnorm(6, mu, 0.1)
                    }, 
                    rnorm(13, 0.0, 0.4))), 
                isold = rep(c(rep(F, 4), rep(T, 2)), 13))
    m$L1 = factor(m$L1)
    m$isold = factor(m$isold)  # without this: wrong results

    ## true effect data
    #mu.d = rnorm(1, 0, 0.2)
    #m = data.frame(melt(
    #            Map(function(mu) {
    #                c(rnorm(4, mu, 0.1), rnorm(2, mu - mu.d, 0.1))
    #                }, 
    #                rnorm(13, 0.0, 0.4))), 
    #            isold = rep(c(rep(F, 4), rep(T, 2)), 13))
    #m$L1 = factor(m$L1)
    #m$isold = factor(m$isold)  # without this: wrong results

    # getting p-value out of model is... ugly. Was this never meant to be automated?

    mm = aov(value ~ isold + Error(L1 / isold), data=m)
    smm = summary(mm)
    pval = smm[[2]][[1]][["Pr(>F)"]][1]

    #mm = aov(value ~ isold, data=m)
    #smm = summary(mm)
    #pval = smm[[1]][["Pr(>F)"]][1]

    #mm = aov(value ~ isold + Error(L1), data=m)
    #smm = summary(mm)
    #pval = smm[[2]][[1]][["Pr(>F)"]][1]

    pvals[[i]] = pval
}

pvals = unlist(pvals)
plot(density(pvals))

print(sum(pvals <= 0.01))
print(sum(pvals <= 0.05))
print(sum(pvals <= 0.1))

ggplot(m, aes(x=rep(seq(1, 6), 13), fill=isold, y=value)) + geom_bar(position="dodge", stat="identity") + facet_wrap(~L1)
