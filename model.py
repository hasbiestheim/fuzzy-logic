from scipy.spatial.distance import *
from numpy import *

class TSK:
	def __init__(self, Cik, X, Y, r=1.5):
		self.__Cik = Cik
		self.__r = r
		d = cdist(Cik.T, Cik.T)
		d[d == 0] = float('inf')
		indexes = argmin(d, axis=0)[:-1]
		indexes = append(indexes, argmin(d,axis=1)[-1])

		self.__aik = zeros(len(Cik.T))
		for i in xrange(len(Cik.T)):
			self.__aik[i] = sum((Cik[:,i] - Cik[:,indexes[i]]) ** 2) / self.__r

		self.__bok = zeros(len(Cik.T))
		for i in xrange(len(Cik.T)):
			up, down = 0, 0
			for x,y in zip(X,Y):
				up += self.__alphak(x, i)*y
				down += self.__alphak(x, i)
			self.__bok[i] = up / down		

	def __muik(self, x, k):
		return exp(-((x - self.__Cik[:, k]) ** 2) / (2 * self.__aik[k] ** 2) )

	def __alphak(self, x, k):
		return reduce(lambda x,y: x*y, self.__muik(x,k))

	def result(self, x):
		l3ok = zeros(size(len(self.__Cik.T)))
		for i in xrange(len(l3ok)):
			l3ok[i] = self.__alphak(x, i)

		l4o1 = sum(l3ok * self.__bok)
		l402 = sum(l3ok)
		print l4o1, l402
		return l4o1 / l402
