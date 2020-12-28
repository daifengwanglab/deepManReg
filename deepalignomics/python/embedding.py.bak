import numpy as np
import scipy as sp
from sklearn.manifold import Isomap,LocallyLinearEmbedding
from neighborhood import neighbor_graph,laplacian


def lapeig(W=None, L=None, num_vecs=None, return_vals=False):
  tmp_L = (L is None)  # we can overwrite L if it's a tmp variable
  if L is None:
    L = laplacian(W, normed=True)
  vals,vecs = sp.linalg.eigh(L, overwrite_a=tmp_L)  # assumes L is symmetric!
  # not guaranteed to be in sorted order
  idx = np.argsort(vals)
  vecs = vecs.real[:,idx]
  vals = vals.real[idx]
  # discard any with really small eigenvalues
  for i in xrange(vals.shape[0]):
    if vals[i] >= 1e-8:
      break
  if num_vecs is None:
    # take all of them
    num_vecs = vals.shape[0] - i
  embedding = vecs[:,i:i+num_vecs]
  if return_vals:
    return embedding, vals[i:i+num_vecs]
  return embedding


def lapeig_linear(X=None,W=None,L=None,num_vecs=None,k=None,eball=None):
  if L is None:
    if W is None:
      W = neighbor_graph(X, k=k, epsilon=eball)
    L = laplacian(W)
  u,s,_ = np.linalg.svd(np.dot(X.T,X))
  Fplus = np.linalg.pinv(np.dot(u,np.diag(np.sqrt(s))))
  T = reduce(np.dot,(Fplus,X.T,L,X,Fplus.T))
  L = 0.5*(T+T.T)
  return lapeig(L=L,num_vecs=num_vecs)


def isomap(X=None,W=None,num_vecs=None,k=None):
  embedder = Isomap(n_neighbors=k, n_components=num_vecs)
  return embedder.fit_transform(X)


def lle(X=None,W=None,num_vecs=None,k=None):
  embedder = LocallyLinearEmbedding(n_neighbors=k, n_components=num_vecs)
  return embedder.fit_transform(X)


def slow_features(X=None,num_vecs=None):
  assert X.shape[0] >= 2, 'Must have at least 2 points to compute derivative'
  t_cov = np.cov(X, rowvar=False)  # variables are over columns
  dXdt = np.diff(X, axis=0)
  dt_cov = np.cov(dXdt, rowvar=False)
  if num_vecs is not None:
    num_vecs = (0,num_vecs-1)
  vals, vecs = sp.linalg.eigh(dt_cov, t_cov, eigvals=num_vecs, overwrite_a=True, overwrite_b=True)
  return vecs


if __name__ == '__main__':
  # simple usage example / visual test case
  from matplotlib import pyplot
  from viz import show_neighbor_graph
  from util import Timer
  from correspondence import Correspondence
  from synthetic_data import cylinder

  n = 300
  knn = 5
  out_dim = 2

  X = cylinder(np.linspace(0,4,n))
  W = neighbor_graph(X=X,k=knn)
  corr = Correspondence(matrix=W)

  with Timer('LapEig'):
    le_embed = lapeig(W=W, num_vecs=out_dim)
  with Timer('Linear LapEig'):
    # lapeig_linear returns a projector, not an embedding
    lel_embed = np.dot(X, lapeig_linear(X=X, W=W, num_vecs=out_dim, k=knn))
  with Timer('Isomap'):
    im_embed = isomap(X=X, num_vecs=out_dim, k=knn)
  with Timer('LLE'):
    lle_embed = lle(X=X, num_vecs=out_dim, k=knn)
  with Timer('SFA'):
    sfa_embed = np.dot(X, slow_features(X=X, num_vecs=out_dim))

  show_neighbor_graph(X, corr, 'Original space')

  fig, axes = pyplot.subplots(nrows=3,ncols=2)
  fig.tight_layout()  # spaces the subplots better
  show_neighbor_graph(le_embed, corr, 'Laplacian Eigenmaps', ax=axes[0,0])
  show_neighbor_graph(lel_embed, corr, 'Linear Laplacian Eigenmaps', ax=axes[0,1])
  show_neighbor_graph(im_embed, corr, 'Isomap', ax=axes[1,0])
  show_neighbor_graph(lle_embed, corr, 'Locally Linear Embedding', ax=axes[1,1])
  show_neighbor_graph(sfa_embed, corr, 'Slow Features Embedding', ax=axes[2,0])
  pyplot.show()
