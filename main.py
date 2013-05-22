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

k = Kohonen(H=20)
Cik = k.cluster(X)
print Cik
tsk = TSK(Cik, X, y)
print tsk.result(X[120,:])