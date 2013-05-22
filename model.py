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

		import pdb; pdb.set_trace()
		# dist = triu(d).flatten()
		# dist[dist == 0] = float('inf')
		# order = []
		# import pdb; pdb.set_trace()
		# while dist[dist != float('inf')].any():
		# 	ind = argmin(dist)
		# 	i = ind / len(Cik.T)
		# 	j = ind % len(Cik.T)
		# 	order.append((i,j))
		# 	dist[ind] = float('inf')
		# 	import pdb; pdb.set_trace()

		#d[d == 0] = float('inf')
		#row_ind = argmin(d, axis=0)
		#self.__aik = zeros(shape(Cik))
		#for i in xrange(len(Cik.T)):
		#	self.__aik[:, row_ind[i]] = Cik[:,i]
		#self.__muik = lambda x: exp(-()/2)
		

	def __muik(self, x, k):
		return exp(-((x - self.__Cik[:, k]) ** 2) / (2 * self.__aik[k] ** 2) )

	def __alphak(self, x, k):
		return reduce(lambda x,y: x*y, self.__muik(x,k))