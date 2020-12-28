import numpy as np
import scipy.spatial.distance as sd
from sklearn.metrics.pairwise import euclidean_distances


'''Distance functions, grouped by metric.'''


class Metric(object):
  def __init__(self,dist,name):
    self.dist = dist  # dist(x,y): distance between two points
    self.name = name

  def within(self,A):
    '''pairwise distances between each pair of rows in A'''
    return sd.squareform(sd.pdist(A,self.name),force='tomatrix')

  def between(self,A,B):
    '''cartesian product distances between pairs of rows in A and B'''
    return sd.cdist(A,B,self.name)

  def pairwise(self,A,B):
    '''distances between pairs of rows in A and B'''
    return np.array([self.dist(a,b) for a,b in zip(A,B)])


class SparseL2Metric(Metric):
  '''scipy.spatial.distance functions don't support sparse inputs,
  so we have a separate SparseL2 metric for dealing with them'''
  def __init__(self):
    def _sparse_l2(A, B):
      diff = A - B
      if hasattr(diff, 'power'):
        sqdiff = diff.power(2)
      else:
        sqdiff = np.power(diff, 2)
      return np.sqrt(sqdiff.sum())

    self.dist = _sparse_l2
    self.name = 'sparseL2'

  def within(self, A):
    return euclidean_distances(A,A)

  def between(self,A,B):
    return euclidean_distances(A,B)

  def pairwise(self,A,B):
    '''distances between pairs of rows in A and B'''
    return Metric.pairwise(self, A, B).flatten()

# commonly-used metrics
L1 = Metric(sd.cityblock,'cityblock')
L2 = Metric(sd.euclidean,'euclidean')
SquaredL2 = Metric(sd.sqeuclidean,'sqeuclidean')
SparseL2 = SparseL2Metric()
