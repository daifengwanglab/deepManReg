library(plyr)
library(reshape2)
library(ManiNetCluster)
library(dplyr)

method = c('linear manifold','cca','manifold warping','nonlinear manifold aln','nonlinear manifold warp')

# visual
edata = read.csv('../data/efeature_filtered.csv',row.names = 1)
gdata = read.csv('../data/expMat_filtered.csv',row.names = 1)
label = read.csv('../data/label_visual.csv')

#e-feature
X = t(edata)
#g-feature
Y= gdata
corr = Correspondence(matrix = cor(t(X),t(Y)))
#corr=Correspondence(matrix=diag(nrow(X)))

#lma
NMA_res = ManiNetCluster(X,Y,nameX='Ephys',nameY='Expr',corr=corr,method = method[1],d=10L,k_NN=5L,k_medoids=5L)
x = NMA_res[3:12]
write.csv(x,'../data/ma_latent.csv',row.names = F)

#cca
NMA_res = ManiNetCluster(X,Y,nameX='Ephys',nameY='Expr',corr=corr,method = method[2],d=10L,k_NN=5L,k_medoids=5L)
x = NMA_res[3:12]
write.csv(x,'../data/cca_latent.csv',row.names = F)
