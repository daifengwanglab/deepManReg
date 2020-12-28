import numpy as np
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix
from io import StringIO

import alignment
import correspondence
import distance
import dtw
import neighborhood
import util


# TODO: test embedding methods: lapeig, isomap, lle

class TestAlignment(unittest.TestCase):
  def setUp(self):
    self.points1 = np.array([[0,0],[2,4],[4,4],[6,9]])
    self.points2 = np.array([[0,0],[0,2],[2,4],[6,6]])

  def test_TrivialAlignment(self):
    proj = alignment.TrivialAlignment(self.points1, self.points2)
    p1,p2 = proj.project(self.points1, self.points2)
    assert_array_equal(self.points1, p1)
    assert_array_equal(self.points2, p2)
    p1,p2 = proj.project(self.points1, self.points2, num_dims=1)
    assert_array_equal(self.points1[:,:1], p1)
    assert_array_equal(self.points2[:,:1], p2)

  # TODO: test the other aligners


class TestCorrespondence(unittest.TestCase):
  def test_Correspondence(self):
    adj = np.array([[0,0,1],[0,0,1],[1,1,0]])
    pairs = np.array([[0,2],[1,2],[2,0],[2,1]])
    corr1 = correspondence.Correspondence(matrix=adj)
    corr2 = correspondence.Correspondence(pairs=pairs)
    assert_array_equal(corr1.pairs(), pairs)
    assert_array_equal(corr2.matrix(), adj)
    self.assertEqual(corr1.dist_from(corr2), 0)
    # TODO: test corr.warp()


class TestDistance(unittest.TestCase):
  def setUp(self):
    self.x = np.array([1,1])
    self.y = np.array([-1,4])
    self.A = np.array([self.x, self.y, [0,0]])
    self.B = np.array([self.y, [-3,5], [2,0]])

  def test_L1(self):
    self.assertEqual(distance.L1.dist(self.x, self.y), 5)
    expected = np.array([[0,5,2],[5,0,5],[2,5,0]])
    assert_array_almost_equal(distance.L1.within(self.A), expected)
    expected = np.array([[5,8,2],[0,3,7],[5,8,2]])
    assert_array_almost_equal(distance.L1.between(self.A, self.B), expected)
    assert_array_almost_equal(distance.L1.pairwise(self.A, self.B), np.diag(expected))

  def test_L2(self):
    self.assertAlmostEqual(distance.L2.dist(self.x, self.y), np.sqrt(13))
    expected = np.sqrt(np.array([[0,13,2],[13,0,17],[2,17,0]]))
    assert_array_almost_equal(distance.L2.within(self.A), expected)
    expected = np.sqrt(np.array([[13,32,2],[0,5,25],[17,34,4]]))
    assert_array_almost_equal(distance.L2.between(self.A, self.B), expected)
    assert_array_almost_equal(distance.L2.pairwise(self.A, self.B), np.diag(expected))

  def test_SquaredL2(self):
    self.assertEqual(distance.SquaredL2.dist(self.x, self.y), 13)
    expected = np.array([[0,13,2],[13,0,17],[2,17,0]])
    assert_array_almost_equal(distance.SquaredL2.within(self.A), expected)
    expected = np.array([[13,32,2],[0,5,25],[17,34,4]])
    assert_array_almost_equal(distance.SquaredL2.between(self.A, self.B), expected)
    assert_array_almost_equal(distance.SquaredL2.pairwise(self.A, self.B), np.diag(expected))

  def test_SparseL2(self):
    self.assertAlmostEqual(distance.SparseL2.dist(self.x, self.y), np.sqrt(13))
    sA = csr_matrix(self.A)
    sB = csr_matrix(self.B)
    expected = np.sqrt(np.array([[0,13,2],[13,0,17],[2,17,0]]))
    assert_array_almost_equal(distance.SparseL2.within(self.A), expected)
    assert_array_almost_equal(distance.SparseL2.within(sA), expected)
    expected = np.sqrt(np.array([[13,32,2],[0,5,25],[17,34,4]]))
    assert_array_almost_equal(distance.SparseL2.between(self.A, self.B), expected)
    assert_array_almost_equal(distance.SparseL2.between(sA, sB), expected)
    assert_array_almost_equal(distance.SparseL2.pairwise(self.A, self.B), np.diag(expected))
    assert_array_almost_equal(distance.SparseL2.pairwise(sA, sB), np.diag(expected))


class TestDTW(unittest.TestCase):
  def test_dtw(self):
    points1 = np.array([[0,0],[2,4],[4,4],[6,9]])
    points2 = np.array([[0,0],[0,2],[2,4],[6,6]])
    corr_fast = dtw.dtw(points1, points2)
    corr_slow = dtw.dtw(points1, points2, debug=True)
    assert_array_equal(corr_fast.matrix(), corr_slow.matrix())
    expected = np.array([[0,0],[0,1],[1,2],[2,2],[3,3]])
    assert_array_equal(corr_fast.pairs(), expected)


class TestNeighborhood(unittest.TestCase):
  def setUp(self):
    self.pts = np.array([[0,0],[1,2],[3,2],[-1,0]])

  def test_neighbor_graph(self):
    ngraph = neighborhood.neighbor_graph  # long names are long
    self.assertRaises(AssertionError, ngraph, self.pts)
    expected = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,1,0,0]])
    assert_array_equal(ngraph(self.pts, k=2, symmetrize=True), expected)
    assert_array_equal(ngraph(self.pts, epsilon=13), expected)
    expected = np.array([[0,1,0,1],[1,0,1,0],[1,1,0,0],[1,1,0,0]])
    assert_array_equal(ngraph(self.pts, k=2, symmetrize=False), expected)

  def test_laplacian(self):
    pass


class TestUtil(unittest.TestCase):
  def setUp(self):
    self.foo = np.array([[0,1],[2,3],[4,5]])

  def test_pairwise_error(self):
    self.assertEqual(util.pairwise_error(self.foo, self.foo), 0)
    self.assertEqual(util.pairwise_error(self.foo, self.foo, metric=distance.L1), 0)

  def test_block_antidiag(self):
    expected = np.array([[0,0,0,1],[0,0,2,3],[2,3,0,0],[4,5,0,0]])
    assert_array_equal(util.block_antidiag(self.foo[:2], self.foo[1:]), expected)

  def test_Timer(self):
    sysout = StringIO()
    with util.Timer('test', out=sysout):
      pass
    output = sysout.getvalue()
    # not really much we can test for here, without being too brittle
    assert output.startswith('test')


if __name__ == '__main__':
  unittest.main()
