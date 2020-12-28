''' adapted from https://github.com/all-umass/ManifoldWarping '''

from matplotlib import pyplot


# def show_alignment(X,Y,title=None,marker='o-',legend=True):
#   '''plot two data sets on the same figure'''
#   dim = X.shape[1]
#   assert dim == Y.shape[1], 'dimensionality must match'
#   assert dim in (1,2,3), ('can only plot 1, 2, or 3-dimensional data, X has shape %dx%d' % X.shape)
#   if dim == 1:
#     pyplot.plot(X[:,0],marker,label='X',hold=True)
#     pyplot.plot(Y[:,0],marker,label='Y',hold=True)
#   elif dim == 2:
#     pyplot.plot(X[:,0],X[:,1],marker,label='X',hold=True)
#     pyplot.plot(Y[:,0],Y[:,1],marker,label='Y',hold=True)
#   else:  # dim == 3
#     from mpl_toolkits.mplot3d import Axes3D
#     fig = pyplot.gcf()
#     ax = Axes3D(fig)
#     ax.plot(X[:,0],X[:,1],X[:,2],marker,label='X')
#     ax.plot(Y[:,0],Y[:,1],Y[:,2],marker,label='Y')
#   if title:
#     pyplot.title(title)
#   if legend:
#     pyplot.legend(loc='best')
#   return pyplot.show

def show_alignment(X,Y,titX=None,titY=None,title=None,legend=True):
  '''plot two data sets on the same figure'''
  dim = X.shape[1]
  assert dim == Y.shape[1], 'dimensionality must match'
  assert dim in (1,2,3), ('can only plot 1, 2, or 3-dimensional data, X has shape %dx%d' % X.shape)
  if dim == 1:
    pyplot.plot(X[:,0],label=titX,alpha=0.5)
    pyplot.plot(Y[:,0],label=titX,alpha=0.5)
  elif dim == 2:
    pyplot.scatter(X[:,0],X[:,1],label=titX,alpha=0.5)
    pyplot.scatter(Y[:,0],Y[:,1],label=titY,alpha=0.5)
  else:  # dim == 3
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.gcf()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],X[:,2],label=titX,alpha=0.5)
    ax.scatter(Y[:,0],Y[:,1],Y[:,2],label=titY,alpha=0.5)
  if title:
    pyplot.title(title)
  if legend:
    pyplot.legend(loc='best')
  return pyplot.show

def show_neighbor_graph(X,corr,title=None,fig=None,ax=None):
  '''Plot the neighbor connections between points in a data set.
     Note: plotting correspondences for 3d points is slow!'''
  assert X.shape[1] in (2,3), 'can only show neighbor graph for 2d or 3d data'
  if X.shape[1] == 2:
    if ax is None:
      ax = pyplot.gca()
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], 'r-')
    ax.plot(X[:,0],X[:,1],'o')
  else:
    if ax is None:
      from mpl_toolkits.mplot3d import Axes3D
      if fig is None:
        fig = pyplot.gcf()
      ax = Axes3D(fig)
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], X[pair,2], 'r-')
    ax.plot(X[:,0],X[:,1],X[:,2],'o')
  if title:
    ax.set_title(title)
  return pyplot.show