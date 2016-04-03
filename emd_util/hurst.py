
from numpy.fft import fft
from numpy.random import randn
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, \
				  ones, log2, std
from numpy.linalg import svd, lstsq
import time

######################## Functions contributed by Xin Liu #################

def hurst(X):
	""" Compute the Hurst exponent of X. If the output H=0.5,the behavior
	of the time-series is similar to random walk. If H<0.5, the time-series
	cover less "distance" than a random walk, vice verse. 

	Parameters
	----------

	X

		list    
		
		a time series

	Returns
	-------
	H
        
		float    

		Hurst exponent

	Examples
	--------

	>>> import pyeeg
	>>> from numpy.random import randn
	>>> a = randn(4096)
	>>> pyeeg.hurst(a)
	>>> 0.5057444
	
	"""
	
	N = len(X)
	T = array([float(i) for i in xrange(1,N+1)])
	Y = cumsum(X)
	Ave_T = Y/T
	
	S_T = zeros((N))
	R_T = zeros((N))
	for i in xrange(1,N):
		S_T[i] = std(X[:i+1])
		X_T = Y - T * Ave_T[i]
		R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

        R_T[0] = R_T[1]
        S_T[0] = S_T[1]
        
    
	R_S = R_T/S_T
	R_S = log(R_S)
	n = log(T).reshape(N, 1)
	H = lstsq(n[1:], R_S[1:])[0]
	return H[0]


if __name__=="__main__":
    a = randn(4096)
    print hurst(a)

