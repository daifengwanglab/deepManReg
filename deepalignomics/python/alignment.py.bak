''' Alignment techniques '''
import numpy as np
import scipy as sp
from neighborhood import laplacian
from util import block_antidiag


def _manifold_setup(Wx,Wy,Wxy,mu):
  Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
  W = np.asarray(np.bmat(((Wx,Wxy),(Wxy.T,Wy))))
  return laplacian(W)


def _manifold_decompose(L,d1,d2,num_dims,eps,vec_func=None):
  vals,vecs = np.linalg.eig(L)
  idx = np.argsort(vals)
  for i in xrange(len(idx)):
    if vals[idx[i]] >= eps:
      break
  vecs = vecs.real[:,idx[i:]]
  if vec_func:
    vecs = vec_func(vecs)
  for i in xrange(vecs.shape[1]):
    vecs[:,i] /= np.linalg.norm(vecs[:,i])
  map1 = vecs[:d1,:num_dims]
  map2 = vecs[d1:d1+d2,:num_dims]
  return map1,map2


def _linear_decompose(X,Y,L,num_dims,eps):
  Z = sp.linalg.block_diag(X.T,Y.T)
  u,s,_ = np.linalg.svd(np.dot(Z,Z.T))
  Fplus = np.linalg.pinv(np.dot(u,np.diag(np.sqrt(s))))
  T = reduce(np.dot,(Fplus,Z,L,Z.T,Fplus.T))
  L = 0.5*(T+T.T)
  d1,d2 = X.shape[1],Y.shape[1]
  return _manifold_decompose(L,d1,d2,num_dims,eps,lambda v: np.dot(Fplus.T,v))


class LinearAlignment(object):
  def project(self,X,Y,num_dims=None):
    if num_dims is None:
      return np.dot(X,self.pX), np.dot(Y,self.pY)
    return np.dot(X,self.pX[:,:num_dims]), np.dot(Y,self.pY[:,:num_dims])

  def apply_transform(self, other):
    self.pX = np.dot(self.pX, other.pX)
    self.pY = np.dot(self.pY, other.pY)


class TrivialAlignment(LinearAlignment):
  def __init__(self, X, Y, num_dims=None):
    self.pX = np.eye(X.shape[1],num_dims)
    self.pY = np.eye(Y.shape[1],num_dims)


class Affine(LinearAlignment):
  ''' Solves for projection P s.t. Yx = Y*P '''
  def __init__(self, X, Y, corr, num_dims):
    c = corr.pairs()
    assert c.shape[0] > 0, "Can't align data with no correlation"
    Xtrain = X[c[:,0]]
    Ytrain = Y[c[:,1]]
    self.pY = np.linalg.lstsq(Ytrain,Xtrain)[0][:,:num_dims]
    self.pX = np.eye(self.pY.shape[0],num_dims)


class Procrustes:  # note: not a LinearAlignment because it requires mean centering
  ''' Solves for scaling k and rotation Q s.t. Yx = k*Y*Q '''
  def __init__(self, X, Y, corr, num_dims):
    c = corr.pairs()
    Xtrain = X[c[:,0]]
    Ytrain = Y[c[:,1]]
    mX = Xtrain - X.mean(0)
    mY = Ytrain - Y.mean(0)
    u,s,vT = np.linalg.svd(np.dot(mY.T,mX))
    k = s.sum() / np.trace(np.dot(mY.T,mY))
    self.pY = k*np.dot(u,vT)[:num_dims]

  def project(self,X,Y,num_dims=None):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    if num_dims is None:
      return mX, np.dot(mY,self.pY)
    return mX[:,:num_dims], np.dot(mY,self.pY[:,:num_dims])


class CCA(LinearAlignment):
  def __init__(self,X,Y,corr,num_dims,eps=1e-8):
    Wxy = corr.matrix()
    L = laplacian(block_antidiag(Wxy,Wxy.T))
    self.pX, self.pY = _linear_decompose(X,Y,L,num_dims,eps)


class CCAv2:  # same deal as with procrustes
  def __init__(self,X,Y,num_dims):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    Cxx = np.dot(mX.T,mX)
    Cyy = np.dot(mY.T,mY)
    Cxy = np.dot(mX.T,mY)
    d1,d2 = Cxy.shape
    if np.linalg.matrix_rank(Cxx) < d1 or np.linalg.matrix_rank(Cyy) < d2:
      lam = X.shape[0]/2.0
    else:
      lam = 0
    Cx = block_antidiag(Cxy,Cxy.T)
    Cy = sp.linalg.block_diag(Cxx + lam*np.eye(d1), Cyy + lam*np.eye(d2))
    vals,vecs = sp.linalg.eig(Cx,Cy)
    vecs = vecs[np.argsort(vals)[::-1]]  # descending order
    self.pX = vecs[:d1,:num_dims]
    self.pY = vecs[d1:d1+d2,:num_dims]

  def project(self,X,Y,num_dims=None):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    if num_dims is None:
      return np.dot(mX,self.pX), np.dot(mY,self.pY)
    return np.dot(mX,self.pX[:,:num_dims]), np.dot(mY,self.pY[:,:num_dims])


try:
  from sklearn import pls
  pls.CCA  # make sure it exists

  class CCAv3:
    def __init__(self,X,Y,num_dims):
      self._model = pls.CCA(n_components=num_dims)
      self._model.fit(X,Y)

    def project(self,X,Y,num_dims=None):
      pX,pY = self._model.transform(X,Y)
      if num_dims is None:
        return pX,pY
      return pX[:,:num_dims], pY[:,:num_dims]
except ImportError:
  pass


class ManifoldLinear(LinearAlignment):
  def __init__(self,X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8):
    L = _manifold_setup(Wx,Wy,corr.matrix(),mu)
    self.pX, self.pY = _linear_decompose(X,Y,L,num_dims,eps)


def manifold_nonlinear(X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8):
  L = _manifold_setup(Wx,Wy,corr.matrix(),mu)
  return _manifold_decompose(L,X.shape[0],Y.shape[0],num_dims,eps)
