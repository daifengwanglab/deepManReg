''' Inter-data correspondences '''
import numpy as np


class Correspondence(object):

  def __init__(self, pairs=None, matrix=None):
    if pairs is None and matrix is None:
      raise ValueError('Must provide pairwise or matrix correspondences')
    self._pairs = pairs
    self._matrix = matrix

  def pairs(self):
    if self._pairs is None:
      self._pairs = np.column_stack(self._matrix.nonzero())
    return self._pairs

  def matrix(self):
    if self._matrix is None:
      self._matrix = np.zeros(self._pairs.max(axis=0)+1)
      i, j = self._pairs.T
      self._matrix[i,j] = 1
    return self._matrix

  def dist_from(self, other):
    '''Calculates the warping path distance from this correspondence to another.
       Based on the implementation from CTW.'''
    B1, B2 = self._bound_row(), other._bound_row()
    gap0 = np.abs(B1[:-1,1] - B2[:-1,1])
    gap1 = np.abs(B1[1:,0] - B2[1:,0])
    d = gap0.sum() + np.count_nonzero(gap0 != gap1) / 2.0
    return d / float(self.pairs()[-1,0]*other.pairs()[-1,0])

  def warp(self, A, XtoY=True):
    '''Warps points in A by pairwise correspondence'''
    P = self.pairs()
    if not XtoY:
      P = np.fliplr(P)
    warp_inds = np.zeros(A.shape[0], dtype=int)
    j = 0
    for i in range(A.shape[0]):
      while P[j,0] < i:
        j += 1
      warp_inds[i] = P[j,1]
    return A[warp_inds]

  def _bound_row(self):
    P = self.pairs()
    n = P.shape[0]
    B = np.zeros((P[-1,0]+1,2), dtype=int)
    head = 0
    while head < n:
      i = P[head,0]
      tail = head+1
      while tail < n and P[tail,0] == i:
        tail += 1
      B[i,:] = P[(head,tail-1),1]
      head = tail
    return B


if __name__ == '__main__':
  # simple sanity-check tests
  from neighborhood import neighbor_graph
  from viz import show_neighbor_graph, pyplot
  n = 500
  data = np.random.uniform(-1,1,(n,2))
  corr_k = Correspondence(matrix=neighbor_graph(data,k=3))
  corr_eps = Correspondence(matrix=neighbor_graph(data,epsilon=0.01))
  pyplot.subplot(1,2,1)
  show_neighbor_graph(data, corr_k, 'kNN graph, k = 3')
  pyplot.subplot(1,2,2)
  show_neighbor_graph(data, corr_eps,
                      '$\epsilon$-ball graph, $\epsilon$ = 0.1')()
