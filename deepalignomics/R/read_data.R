# gwas <- read.csv('Downloads/NIHMS958804-supplement-Supplementary_Table.csv')[-c(1:5),17]
# gene <- read.table('Downloads/DER-08a_hg19_eQTL.significant.txt', header = TRUE)[,1:3]
# gene$gene_id <- sapply(gene$gene_id, function(x) gsub("\\..*", "", x))
# gene$gene_chr <- sapply(gene$gene_chr, function(x) gsub("chr", "", x))
# gene$snp_id <- paste(gene$gene_chr, as.character(gene$gene_start), sep=":")
# gene <- gene[,-c(2,3)]
# gene <- unique(gene)
# gene[gene$snp_id %in% as.character(gwas),]
# snp <- read.table('Downloads/Capstone4.eSNP.dosage.txt')
# snp <- snp[snp$V3 %in% as.character(gwas),]
# 
# eQTL <- read.table('Downloads/DER-08a_hg19_eQTL.significant.txt', header = TRUE)[,1:3]
# eQTL$gene_id <- sapply(eQTL$gene_id, function(x) gsub("\\..*", "", x))
# eQTL$gene_chr <- sapply(eQTL$gene_chr, function(x) gsub("chr", "", x))
# eQTL$snp_id <- paste(eQTL$gene_chr, as.character(eQTL$gene_start), sep = ":")
# eQTL <- eQTL[,-c(2,3)]
# eQTL <- unique(eQTL)

snp_2 <- read.table('Downloads/Capstone4.eSNP.dosage.txt')[,-c(1,2)]
# intersect(eQTL$snp_id, snp_2$V3)

eQTL2 <- read.table('Downloads/PEC_eQTL_p=0.01_MAF=0.01.txt', header = TRUE)[,1:2]
# length(intersect(eQTL2$rsid, as.character(gwas)))

gene_expr <- read.table('Downloads/DER-02_PEC_Gene_expression_matrix_TPM.txt', header = TRUE)

GRN <- read.csv('Downloads/INT-11_ElasticNet_Filtered_Cutoff_0.1_GRN_1.csv', header = TRUE)[,-c(3,4)]
