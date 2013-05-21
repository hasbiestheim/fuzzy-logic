from numpy import *
from numpy.random import *

class Kohonen:
	def __init__(self, H, maxEpoch=5, eps=1e-4, alpha_w=0.7, alpha_r=0.2):
		self.__H = H
		self.__maxEpoch = maxEpoch
		self.__eps = eps
		self.__alpha_w = alpha_w
		self.__alpha_r = alpha_r

	def __d(self, x, Ck):
		pass

	def __argmin(self, x, Cik, w=None):
		if w is None:
			pass
		else:
			pass

	def cluster(self, X, y):
		epochNumber = 0
		Nk = ones(self.__H)
		Cik = rand(len(X.T)*self.__H).reshape(self.__H, len(X)).T
		for epochNumber in xrange(self.__maxEpoch):
			for x in X:
				w = self.__argmin(x, Cik)
				r = self.__argmin(x, Cik, w)
			import pdb; pdb.set_trace()
