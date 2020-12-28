''' miscellaneous utilities '''
import numpy
import scipy
import sys
import time

from distance import L2


def pairwise_error(A,B,metric=L2):
  ''' sum of distances between points in A and B, normalized '''
  return metric.pairwise(A/A.max(),B/B.max()).sum()


def block_antidiag(*args):
  ''' makes a block anti-diagonal matrix from the block matices given '''
  return numpy.fliplr(scipy.linalg.block_diag(*map(numpy.fliplr,args)))


class Timer(object):
  '''Context manager for simple timing of code:
  with Timer('test 1'):
    do_test1()
  '''
  def __init__(self, name, out=sys.stdout):
    self.name = name
    self.out = out

  def __enter__(self):
    self.start = time.time()

  def __exit__(self,*args):
    self.out.write("%s : %0.3f seconds\n" % (self.name, time.time()-self.start))
    return False
