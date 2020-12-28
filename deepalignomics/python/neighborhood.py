from distance import SquaredL2
import numpy as np


def neighbor_graph(X, metric=SquaredL2, k=None, epsilon=None, symmetrize=True):
  '''Construct an adj matrix from a matrix of points (one per row)'''
  assert (k is None) ^ (epsilon is None), "Must provide `k` xor `epsilon`"
  dist = metric.within(X)
  adj = np.zeros(dist.shape)  # TODO: scipy.sparse support, or at least use a smaller dtype
  if k is not None:
    # do k-nearest neighbors
    nn = np.argsort(dist)[:,:min(k+1,len(X))]
    # nn's first column is the point idx, rest are neighbor idxs
    if symmetrize:
      for inds in nn:
        adj[inds[0],inds[1:]] = 1
        adj[inds[1:],inds[0]] = 1
    else:
      for inds in nn:
        adj[inds[0],inds[1:]] = 1
  else:
    # do epsilon-ball
    p_idxs, n_idxs = np.nonzero(dist<=epsilon)
    for p_idx, n_idx in zip(p_idxs, n_idxs):
      if p_idx != n_idx:  # ignore self-neighbor connections
        adj[p_idx,n_idx] = 1
    # epsilon-ball is typically symmetric, assuming a normal distance metric
  return adj


def laplacian(W, normed=False, return_diag=False):
  '''Same as the dense laplacian from scipy.sparse.csgraph'''
  n_nodes = W.shape[0]
  lap = -np.asarray(W)  # minus sign leads to a copy
  # set diagonal to zero, in case it isn't already
  lap.flat[::n_nodes + 1] = 0
  d = -lap.sum(axis=0)  # re-negate to get positive degrees
  if normed:
    d = np.sqrt(d)
    d_zeros = (d == 0)
    d[d_zeros] = 1  # avoid div by zero
    # follow is the same as: diag(1/d) x W x diag(1/d) (where x is np.dot)
    lap /= d
    lap /= d[:, np.newaxis]
    lap.flat[::n_nodes + 1] = 1 - d_zeros
  else:
    # put the degrees on the diagonal
    lap.flat[::n_nodes + 1] = d
  if return_diag:
    return lap, d
  return lap


if __name__ == '__main__':
  from matplotlib import pyplot
  from synthetic_data import cylinder
  n,k = 300,5
  X = cylinder(np.linspace(0,4,n))
  adj = neighbor_graph(X=X,k=k)
  L = laplacian(adj,normed=False)
  Lhk = laplacian(adj,normed=True)
  _,axes = pyplot.subplots(1,3)
  labels = ('neighbors', 'laplacian', 'normed laplacian')
  for axis,mat,label in zip(axes.flat,(adj,L,Lhk),labels):
    axis.imshow(mat, interpolation='nearest')
    axis.set_title(label)
  pyplot.show()
