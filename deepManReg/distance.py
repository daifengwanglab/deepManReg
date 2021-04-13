''' adapted from https://github.com/all-umass/ManifoldWarping '''

import numpy as np
import scipy.spatial.distance as sd
from sklearn.metrics.pairwise import euclidean_distances
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

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
    return np.array([self.dist(a,b) for a,b in izip(A,B)])
    
class SparseL2Metric(Metric):
  '''scipy.spatial.distance functions don't support sparse inputs,
  so we have a separate SparseL2 metric for dealing with them'''
  def __init__(self):
    Metric.__init__(self, euclidean_distances, 'sparseL2')

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