''' helper functions for generating synthetic data '''
import numpy as np
from numpy.random import randn, uniform, normal


def add_noise(data, noise_stdv):
  return data + normal(scale=noise_stdv,size=data.shape)


# 2-d data generation functions #
def spiral(x):
  s = np.zeros((x.shape[0],2))
  s[:,0] = x*np.sin(x)
  s[:,1] = x*np.cos(x)
  return s


def squiggle(x):
  s = np.zeros((x.shape[0],2))
  s[:,0] = x
  s[:,1] = x*np.sin(x)**2
  return s


def gaussian_clusters(num_clusters, total_pts):
  n = total_pts // num_clusters
  clusters = [randn(n,2) + uniform(-9,9,2) for _ in xrange(num_clusters)]
  return np.concatenate(clusters,0)


# 3-d data generation functions #
def swiss_roll(x, rolled_func=None):
  sr = cylinder(x, rolled_func)
  sr[:,0] *= x
  sr[:,2] *= x
  return sr


def cylinder(x, rolled_func=None):
  cyl = np.zeros((x.shape[0],3))
  cyl[:,0] = np.cos(x)
  cyl[:,2] = np.sin(x)
  if rolled_func:
    cyl[:,1] = rolled_func(x)
  else:
    cyl[:,1] = uniform(-1,1,x.shape[0])
  return cyl


def plane(x, func=None):
  n = x.shape[0]
  plane = np.zeros((n,3))
  plane[:,0] = x
  plane[:,2] = x
  if func:
    plane[:,1] = func(x)
  else:
    plane[:,1] = uniform(-1,1,n)
  return plane
