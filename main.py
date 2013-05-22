from numpy import *
from kohonen import *
from model import *

X = genfromtxt('iris.data', delimiter=',', usecols=(0,1,2,3))
y = genfromtxt('iris.data', delimiter=',', usecols=(4))
min_v = amin(X, axis=0)
max_v = amax(X, axis=0)

a = 0
b = 1


#Normalization
for i in xrange(len(X.T)):
	X.T[i] = (X.T[i] - min_v[i] * (b - a)) / (max_v[i] - min_v[i])

#k = Kohonen(H=30)
#Cik = k.cluster(X)
Cik = genfromtxt('valid_classes2.txt', delimiter=',').T
tsk = TSK(Cik, X, y)
#print tsk.Result(X[120,:])
#print tsk.Gauss(X[148,:])

#print reduce(lambda x,y: x+y, [tsk.Err(x,y) for x in X])

tsk.tune(X,y)
#print tsk.Answ(X[120,:], [0, 1, 2])
#print tsk.Err(X[120,:],y[120])

tmp = array([tsk.Result(_x) for _x,_y in zip(X,y)])
tmptmp = around(tmp) - y
print float(len(tmptmp[tmptmp == 0])) / len(y)
#print len(nonzero(tmp))/len(X)