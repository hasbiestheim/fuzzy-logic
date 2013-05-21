from scipy.spatial.distance import *
from numpy import *

class TSK:
	def __init__(self, Cik, r=1.5):
		self.__Cik = Cik
		self.__r = r
		d = cdist(Cik.T, Cik.T)
		d[d == 0] = float('inf')
		row_ind = argmin(d, axis=0)
		self.__aik = zeros(shape(Cik))
		for i in xrange(len(Cik.T)):
			self.__aik[:, row_ind[i]] = Cik[:,i]
		#self.__muik = lambda x: exp(-()/2)
		import pdb; pdb.set_trace()

	def diff(self, x,y):
		return (x - y) ** 2