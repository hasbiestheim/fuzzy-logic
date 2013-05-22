from numpy import *
from numpy.random import *

class Kohonen:
	def __init__(self, H, maxEpoch=10, eps=1e-4, alpha_w=0.2, alpha_r=0.1):
		self.__H = H
		self.__maxEpoch = maxEpoch
		self.__eps = eps
		self.__alpha_w = alpha_w
		self.__alpha_r = alpha_r

	def __d(self, x, Ck):
		#import pdb; pdb.set_trace()
		return (((ones((len(Ck.T), len(x)))*x).T - Ck) ** 2 ) * self.__Nk / sum(self.__Nk)

	def __argmin(self, x, Cik):
		d = self.__d(x, Cik)
		row_min = amin(d, axis=0)
		row_min_v = argmin(d, axis=0)
		#import pdb; pdb.set_trace()
		w = argmin(row_min)
		d[row_min_v[w], w] = float('inf')
		row_min = amin(d, axis=0)
		#import pdb; pdb.set_trace()
		r = argmin(row_min)
		return w,r

	def cluster(self, X):
		epochNumber = 0
		self.__Nk = ones(self.__H)
		Cik = rand(len(X.T)*self.__H).reshape(self.__H, len(X.T)).T
		for epochNumber in xrange(self.__maxEpoch):
			Cik_old = Cik.copy()
			for x in X:
				w, r = self.__argmin(x, Cik)
				self.__Nk[w] += 1
				#import pdb; pdb.set_trace()
				Cik[:,w] += self.__alpha_w*(x - Cik[:,w])
				Cik[:,r] -= self.__alpha_r*(x - Cik[:,r]) 
				#import pdb; pdb.set_trace()
				#print argmax(self.__Nk)
			self.__alpha_w -= self.__alpha_w*epochNumber/self.__maxEpoch
			#import pdb; pdb.set_trace()
			if sum((Cik - Cik_old)**2) / self.__H < self.__eps:
				break;
		tmp = Cik.T[~(Cik.T > 1).any(1)]
		return tmp[~(tmp < 0).any(0)].T
