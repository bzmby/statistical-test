#!/usr/bin/env Rscript

library(edgeR)
library(plyr)

geneiddata = read.table("/srv/data/Homo_sapiens.GRCh37.65.chrnames_modified.only_valid_chrs.gene_id_2_gene_name.tsv")
names(geneiddata) = c("gene_id", "gene_name")

FILENAME_FORMAT = "./*/data/tophat_out/*%d*_%d_*/accepted_hits.htseq_count.CDS_union.tsv"

source("./scripts/base/annotations.R");
da = dannotations[dannotations$isused & dannotations$libtype == "RNAseq", ]

d = list()
for (i in seq(dim(da)[1])) {
    samplename = da[i, "samplename"]
    f = Sys.glob(sprintf(FILENAME_FORMAT, da[i, "expnr"], da[i, "samplenr"]))
    d_ = read.table(f)
    names(d_) = c("gene_id", samplename)
    d[[i]] = d_
}

d = Reduce(function(a, b) { join(a, b, by="gene_id") }, d)

isensg = regexpr("^ENSG", d$gene_id, perl=T) >= 0
hasdata = rowSums(d[, -1]) > 0
d = d[isensg & hasdata, ]

controls = regexpr("control$", da[, "samplename"]) >= 0

for (i in which(!controls)) {
    g = factor(c(rep(0, sum(controls)), 1))
    samplename = da[i, "samplename"]
    cols = c((which(controls) + 1), which(names(d) == samplename))  # + 1 for gene_id column in d

    y = DGEList(d[, cols], genes=d[, 1], group=g)
    y = calcNormFactors(y)
    y = estimateCommonDisp(y)
    y = estimateTagwiseDisp(y)

    et = exactTest(y)
    tt = topTags(et, n=Inf)

    lfc = data.frame(gene_id=y$genes$genes, 
                     gene_name=geneiddata[match(y$genes$genes, geneiddata$gene_id), "gene_name"],
                     pred.logFC=((edgeR::predFC(y, design=model.matrix(~ y$samples$group )))[, 2]),
                     sums=rowSums(y$counts))

    m = (5 * length(y$samples))
    lfc[lfc$sums >= m, "pred.logFC.scaled"] = scale(lfc[lfc$sums >= m, "pred.logFC"])
    # NOTE this leaves NAs in the dataframe column pred.logFC.scaled

    tttable = tt$table
    names(tttable)[names(tttable) == "genes"] = "gene_id"
    outtable = join(tttable, lfc, by="gene_id")
    write.table(outtable, sprintf("./output/response_analysis/%s_vs_ctrls.tsv", gsub(" ", "_", samplename, fixed=T)), sep="\t", col.names=T, row.names=F, quote=F)
}

# NOTE
# make rnk files for GSEA with
# ls -1 output/response_analysis/*.tsv | while read f; do of=${f/%tsv/rnk}; tail -n+2 "$f" | cut -f6,9 > "$of"; done
