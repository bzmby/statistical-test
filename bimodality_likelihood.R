#!/usr/bin/env Rscript

#install.packages("diptest")
#install.packages("mclust")

options(max.print=500)
options(width=160)

OUTDIR = "./output/bimodality_analysis/";
FILENAME_FORMAT = "./*/data/tophat_out/*%d*_%d_*/accepted_hits.htseq_count_tagged.locus_size_distribution.tsv"
RLOCCOV_FILENAME_FORMAT = "./*/data/tophat_out/*%d*_%d_*/accepted_hits.htseq_count_tagged.locus_coverage.tsv"

library(reshape2)
library(ggplot2)
library(plyr)
library(diptest)
library(mclust)
library(mixtools)

RNDSEED = 42;

source("./scripts/base/annotations.R");

chrmrlocs = read.table("chrm_pc_rloc.list")

da = dannotations[dannotations$isused & dannotations$libtype == "RP", ]


rlocinfofile = "/srv/data3/deepseq/linc/linc475kd_RNAseq/final_alignments/Ensembl_RefSeq_BroadLinc.gffread_merged.with_refseq_genenames.rloc_attr.tsv"
rlocinfo = read.table(rlocinfofile, sep="\t", header=F, stringsAsFactors=F)
colnames(rlocinfo) = c("rloc", "molecule_types", "features", "gene_name", "gene_ids", "transcript_ids")
cdsrlocs = rlocinfo[grep("CDS", rlocinfo$features, fixed=T), "rloc"]

for (i in seq(dim(da)[1])) {
    f = sprintf(FILENAME_FORMAT, da[i, "expnr"], da[i, "samplenr"]) 
    f = Sys.glob(f)[1]

    fcov = sprintf(RLOCCOV_FILENAME_FORMAT, da[i, "expnr"], da[i, "samplenr"]) 
    fcov = Sys.glob(fcov)[1]

    d = read.table(f)
    dcov = read.table(fcov)
    colnames(dcov) = c("rloc", "cov", "size", "covfrac")
    dcov = dcov[match(rownames(d), dcov$rloc), ]

    samplename = da[i, "samplename"]

    dcutoffcount = rowSums(d[, seq(20, 50)]) >= 100
    dcutoffcov = (dcov$size < 200 & dcov$covfrac >= 0.5) | (dcov$size >= 200 & dcov$cov >= 60 & dcov$covfrac >= 0.2)
    dcutoffcds = rownames(d) %in% cdsrlocs
    dcutoff = dcutoffcount & dcutoffcov & dcutoffcds
    dischrm = rownames(d) %in% chrmrlocs$V1

    # dip statistic
    dipvals = alply(d[, seq(20, 50)], 1, function(x) {dip(as.numeric(x))}, .progress="text")
    dipvals = unlist(dipvals)
    names(dipvals) = rownames(d)

    dipdata = rbind(data.frame(melt(dipvals[dcutoff & dischrm]), ischrm=T), data.frame(melt(dipvals[dcutoff & !dischrm]), ischrm=F))

    # log-likelihood of 1 vs 2 clusters
    llvals = alply(d[dcutoff, seq(20, 50)], 1, 
                function(x) { 
                    set.seed(RNDSEED);
                    y = rep(seq_along(x) + 20, x)
                    M2 = Mclust(y, G=2)
                    M1 = Mclust(y, G=1)
                    return((M2$loglik - M1$loglik) / M2$n)  # correct for unequal n of obs.
                }, 
                .progress="text")

    llvals = unlist(llvals)
    names(llvals) = rownames(d[dcutoff, ])

    lldata = rbind(data.frame(value=llvals[names(llvals) %in% chrmrlocs$V1], ischrm=T), data.frame(value=llvals[!(names(llvals) %in% chrmrlocs$V1)], ischrm=F))


    # fitting two gaussians with fixed mean
    # method: fit two gaussians with fixed mean to the data and test goodness of fit
    doFit = function(lengthcounts) {
        h = rep(seq(20, 50), as.numeric(lengthcounts)[seq(20, 50)])
        set.seed(RNDSEED)
        h.fit = normalmixEM(h, mean.constr=c(27, 33), k=2, verb=F, maxit=5000, maxrestarts=1000)
        return(list(h=h, h.fit=h.fit))
    }

    pmnorm <- function(x, mu, sigma, pmix) {
        pmix[1]*pnorm(x,mu[1],sigma[1]) + (1-pmix[1])*pnorm(x,mu[2],sigma[2])
    }

    testFit = function(h, h.fit) {
        ks.test(h, pmnorm, mu=h.fit$mu, sigma=h.fit$sigma, pmix=h.fit$lambda)
    }

    fstatvals = alply(d[dcutoff, ], 1, 
                    function(x) { 
                        retval = tryCatch( {
                            doFit(x)
                        }, error = function(e) {
                            NA
                        })
                        if (is.na(retval)) {
                            return(NA)
                        }
                        h = retval$h
                        h.fit = retval$h.fit
                        testval = testFit(h, h.fit)
                        teststat = as.numeric(testval$statistic)
                        return(teststat)
                    }, 
                    .progress="text")

    fstatvals = unlist(fstatvals)
    names(fstatvals) = rownames(d[dcutoff, ])

    fstatvals = rbind(data.frame(value=fstatvals[names(fstatvals) %in% chrmrlocs$V1], ischrm=T), data.frame(value=fstatvals[!(names(fstatvals) %in% chrmrlocs$V1)], ischrm=F))

    #fstatvals = fstatvals[!is.na(fstatvals$value), ]


    outdata = merge(
                merge(data.frame(dip=dipdata$value, rloc=rownames(dipdata)), data.frame(ll=lldata$value, rloc=rownames(lldata)), 
                        all=T, by="rloc"), 
                data.frame(fstat=fstatvals$value, rloc=rownames(fstatvals)), 
                all=T, by="rloc"
                )
    outdata = data.frame(outdata, ischrm = outdata$rloc %in% chrmrlocs$V1)
    write.table(outdata, file=paste(OUTDIR, sprintf("dip_ll_twogauss.data.%s.%d.%d.tsv", gsub(" ", "_", samplename), da[i, "expnr"], da[i, "samplenr"]), sep=""), sep="\t", col.names=T, row.names=F, quote=F)
}
