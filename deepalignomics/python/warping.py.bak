''' Warping aligners '''
import numpy as np
from distance import SquaredL2
from dtw import dtw
from embedding import isomap
from correspondence import Correspondence
from alignment import TrivialAlignment,CCA,ManifoldLinear,manifold_nonlinear


def ctw(X,Y,num_dims,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: CCA(A,B,corr,num_dims,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def manifold_warping_linear(X,Y,num_dims,Wx,Wy,mu=0.9,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: ManifoldLinear(A,B,corr,num_dims,Wx,Wy,mu=mu,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def manifold_warping_nonlinear(X,Y,num_dims,Wx,Wy,mu=0.9,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: manifold_nonlinear(A,B,corr,num_dims,Wx,Wy,mu=mu,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments_nonlinear(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def ctw_twostep(X,Y,num_dims,embedder=isomap,**kwargs):
  alt_aligner = lambda A,B,n,**kwargs: ctw(A,B,n,**kwargs)
  return twostep_alignment(X,Y,num_dims,embedder,alt_aligner)


def manifold_warping_twostep(X,Y,num_dims,Wx,Wy,embedder=isomap,**kwargs):
  alt_aligner = lambda A,B,n,**kwargs: manifold_warping_linear(A,B,n,Wx,Wy,**kwargs)
  return twostep_alignment(X,Y,num_dims,embedder,alt_aligner)


def twostep_alignment(X,Y,num_dims,embedder,alt_aligner):
  X_proj, Y_proj = embedder(X,num_dims,k=5), embedder(Y,num_dims,k=5)
  corr, aln = alt_aligner(X_proj,Y_proj,num_dims)
  X_proj, Y_proj = aln.project(X_proj,Y_proj)
  return corr, X_proj, Y_proj


def alternating_alignments(X,Y,proj_align,corr_align,threshold,max_iters):
  corr = Correspondence(pairs=np.array(((0,0),(X.shape[0]-1,Y.shape[0]-1))))
  aln = TrivialAlignment(X,Y)
  X_proj,Y_proj = X.copy(), Y.copy()  # same as aln.project(X,Y)
  for it in xrange(max_iters):
    aln.apply_transform(proj_align(X_proj,Y_proj,corr))
    X_proj,Y_proj = aln.project(X,Y)
    new_corr = corr_align(X_proj,Y_proj)
    if corr.dist_from(new_corr) < threshold:
      return new_corr, aln
    corr = new_corr
  return corr, aln


def alternating_alignments_nonlinear(X,Y,proj_align,corr_align,threshold,max_iters):
  corr = Correspondence(pairs=np.array(((0,0),(X.shape[0]-1,Y.shape[0]-1))))
  X_proj,Y_proj = proj_align(X,Y,corr)
  for it in xrange(max_iters):
    new_corr = corr_align(X_proj,Y_proj)
    if corr.dist_from(new_corr) < threshold:
      return new_corr, X_proj, Y_proj
    corr = new_corr
    X_proj,Y_proj = proj_align(X_proj,Y_proj,corr)
  return corr, X_proj, Y_proj
