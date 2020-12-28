import numpy as np
from correspondence import Correspondence
from distance import SquaredL2
from itertools import product


def dtw(X, Y, metric=SquaredL2, debug=False):
  '''Dynamic Time Warping'''
  dist = metric.between(X,Y)
  if debug:
    path = _python_dtw_path(dist)
  else:
    path = _dtw_path(dist)
  return Correspondence(pairs=path)


def _python_dtw_path(dist):
  '''Pure python, slow version of DTW'''
  nx,ny = dist.shape
  cost = np.zeros(dist.shape)
  trace = np.zeros(dist.shape,dtype=np.int)
  cost[0,:] = np.cumsum(dist[0,:])
  cost[:,0] = np.cumsum(dist[:,0])
  trace[0,:] = 1
  trace[:,0] = 0
  for i,j in product(range(1,nx),range(1,ny)):
    choices = dist[i,j] + np.array((cost[i-1,j], cost[i,j-1], cost[i-1,j-1]))
    trace[i,j] = choices.argmin()
    cost[i,j] = choices.min()
  path = [(nx-1,ny-1)]
  while not (i == 0 and j == 0):
    s = trace[i,j]
    if s == 0:
      i -= 1
    elif s == 1:
      j -= 1
    else:
      i -= 1
      j -= 1
    path.append((i,j))
  return np.array(path)[::-1]


# Shenanigans for running the fast C version of DTW,
# but falling back to pure python if needed
try:
  from weave import inline
  from weave.converters import blitz
except ImportError:
  print('weave not installed; falling back to slow Python DTW')
  _dtw_path = _python_dtw_path
else:
  def _dtw_path(dist):
    '''Fast DTW, with inlined C'''
    nx,ny = dist.shape
    path = np.zeros((nx+ny,2),dtype=np.int)
    code = '''
    int i,j;
    double* cost = new double[ny];
    cost[0] = dist(0,0);
    for (j=1; j<ny; ++j) cost[j] = dist(0,j) + cost[j-1];
    char** trace = new char*[nx];
    for (i=0; i<nx; ++i) {
      trace[i] = new char[ny];
      trace[i][0] = 0;
    }
    for (j=0; j<ny; ++j) {
      trace[0][j] = 1;
    }
    double diag,c;
    for (i=1; i<nx; ++i){
      diag = cost[0];
      cost[0] += dist(i,0);
      for (j=1; j<ny; ++j){
        // c <- min(cost[j],cost[j-1],diag), trace <- argmin
        if (diag < cost[j]){
          if (diag < cost[j-1]){
            c = diag;
            trace[i][j] = 2;
          } else {
            c = cost[j-1];
            trace[i][j] = 1;
          }
        } else if (cost[j] < cost[j-1]){
          c = cost[j];
          trace[i][j] = 0;
        } else {
          c = cost[j-1];
          trace[i][j] = 1;
        }
        diag = cost[j];
        cost[j] = dist(i,j) + c;
      }
    }
    delete[] cost;
    i = nx-1;
    j = ny-1;
    int p = nx+ny-1;
    for (;p>=0; --p){
      path(p,0) = i;
      path(p,1) = j;
      if (i==0 && j==0) break;
      switch (trace[i][j]){
        case 0: --i; break;
        case 1: --j; break;
        default: --i; --j;
      }
    }
    for (i=0; i<nx; ++i) delete[] trace[i];
    delete[] trace;
    return_val = p;
    '''
    p = inline(code,('nx','ny','dist','path'),type_converters=blitz)
    return path[p:]
