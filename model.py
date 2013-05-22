from scipy.spatial.distance import *
from numpy import *

class TSK:
	def __init__(self, Cik, X, Y, eps=1e-4, r=3.41):
		self.__Cik = Cik
		self.__r = r
		self.__eps = eps
		d = cdist(Cik.T, Cik.T)
		d[d == 0] = float('inf')
		indexes = argmin(d, axis=0)[:-1]
		indexes = append(indexes, argmin(d,axis=1)[-1])
		#import pdb; pdb.set_trace()
		self.__aik = zeros((len(Cik.T), len(X.T)))
		for i in xrange(len(Cik.T)):
			self.__aik[i] = (sum((Cik[:,i] - Cik[:,indexes[i]]) ** 2) / self.__r) * ones(len(X.T))
		self.__aik = self.__aik.copy().T
		#import pdb; pdb.set_trace()
		self.__bok = zeros(len(Cik.T))
		for i in xrange(len(Cik.T)):
			up, down = 0, 0
			for x,y in zip(X,Y):
				up += self.__alphak(x, i)*y
				down += self.__alphak(x, i)
			self.__bok[i] = up / down		

		#import pdb; pdb.set_trace()

	def __muik(self, x, k):
		return exp(-((x - self.__Cik[:, k]) ** 2)) / (2 * self.__aik[:,k])

	def __alphak(self, x, k):
		#import pdb; pdb.set_trace()
		return reduce(lambda x,y: x*y, self.__muik(x,k))

	def Gauss(self, x):
		x_spread = (ones((len(self.__Cik.T), len(x)))*x).T
		#import pdb; pdb.set_trace()
		return exp(-(x_spread - self.__Cik) ** 2 ) / (2 * self.__aik)

	def Result(self, x):
		bok_spread = (ones((len(self.__Cik.T), len(x))).T*self.__bok)
		m = self.Gauss(x)
		#import pdb; pdb.set_trace()
		up = reduce(lambda x,y:x*y, sum(m*bok_spread, axis=1))
		down = reduce(lambda x,y:x*y, sum(m, axis=1))
		#import pdb; pdb.set_trace()
		return up / down

	def Err(self, x, y):
		return (self.Result(x) - y) ** 2

	def Err_Cik(self, x, y):
		y_star = (ones((len(self.__Cik.T), len(x)))*self.Result(x)).T
		y_spread = (ones((len(self.__Cik.T), len(x)))*y).T
		x_spread = (ones((len(self.__Cik.T), len(x)))*x).T
	 	bok_spread = (ones((len(self.__Cik.T), len(x))).T*self.__bok)
	 	alpha = ones(len(self.__Cik.T))
	 	for i in xrange(len(self.__Cik.T)):
	 		alpha[i] = self.__alphak(x, i)
	 	alpha_spread = (ones((len(self.__Cik.T), len(x))).T*alpha)
	 	return 2*(y_star - y_spread)*(bok_spread - y_star)*alpha_spread * (x_spread - self.__Cik) / (sum(alpha_spread) * self.__aik)

	def Err_aik(self, x, y):
		x_spread = (ones((len(self.__Cik.T), len(x)))*x).T
		return self.Err_Cik(x,y) * 0.5 *(x_spread - self.__Cik) / self.__aik

	def Err_bok(self, x, y):
		y_star = ones(len(self.__Cik.T))*self.Result(x)
		y_spread = ones(len(self.__Cik.T))*y
		alpha = ones(len(self.__Cik.T))
	 	for i in xrange(len(self.__Cik.T)):
	 		alpha[i] = self.__alphak(x, i)
	 	#import pdb; pdb.set_trace()
	 	return 2*(y_star - y_spread) * alpha / sum(alpha)

	def J(self, X, Y):
		result = 0
		for x,y in zip(X,Y):
			result += self.Err(x,y)
		return result / len(X)

	def tune(self, X, Y, N=50, eta=0.1):
		ind = 0
		while self.J(X,Y) > self.__eps and ind < N:
			for x,y in zip(X,Y):
				deltaCik = eta*self.Err_Cik(x, y)
				deltaaik = eta*self.Err_aik(x, y)
				deltabok = eta*self.Err_bok(x, y)
				self.__Cik -= deltaCik
				self.__aik -= deltaaik
				self.__bok -= deltabok
			#import pdb; pdb.set_trace()
			ind += 1
			if ind % 50 == 0:
				print "Steps :", ind