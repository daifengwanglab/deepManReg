from matplotlib import pyplot


def show_alignment(X,Y,title=None,marker='o-',legend=True):
  '''plot two data sets on the same figure'''
  dim = X.shape[1]
  if dim != Y.shape[1]:
    raise ValueError('dimensionality must match')

  if dim == 1:
    pyplot.plot(X[:,0],marker,label='X', alpha=0.5)
    pyplot.plot(Y[:,0],marker,label='Y', alpha=0.5)
  elif dim == 2:
    pyplot.plot(X[:,0],X[:,1],marker,label='X', alpha=0.5)
    pyplot.plot(Y[:,0],Y[:,1],marker,label='Y', alpha=0.5)
  elif dim == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.gcf()
    ax = Axes3D(fig)
    ax.plot(X[:,0],X[:,1],X[:,2],marker,label='X', alpha=0.5)
    ax.plot(Y[:,0],Y[:,1],Y[:,2],marker,label='Y', alpha=0.5)
  else:
    raise ValueError('can only plot 1, 2, or 3-dimensional data, '
                     'X has shape %s' % (X.shape),)

  if title:
    pyplot.title(title)
  if legend:
    pyplot.legend(loc='best')
  return pyplot.show


def show_neighbor_graph(X,corr,title=None,fig=None,ax=None):
  '''Plot the neighbor connections between points in a data set.
     Note: plotting correspondences for 3d points is slow!'''
  if X.shape[1] == 2:
    if ax is None:
      ax = pyplot.gca()
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], 'r-')
    ax.plot(X[:,0],X[:,1],'o')
  elif X.shape[1] == 3:
    if ax is None:
      from mpl_toolkits.mplot3d import Axes3D
      if fig is None:
        fig = pyplot.gcf()
      ax = Axes3D(fig)
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], X[pair,2], 'r-')
    ax.plot(X[:,0],X[:,1],X[:,2],'o')
  else:
    raise ValueError('can only show neighbor graph for 2d or 3d data')
  if title:
    ax.set_title(title)
  return pyplot.show
