import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from numpy import exp,sqrt
from numpy.linalg import inv,norm
from statistics import mean, median,variance,stdev

"""
#linear regression
"""

def Phi_matrix(xx,basis):
	N = len(xx)
	H = len(xx[1])
	Phi_1 = np.array([1]*N).reshape(N,1)
	Phi_2 = np.array([basis(x) for x in xx]).reshape(N,H)
	return np.append(Phi_1,Phi_2,axis = 1)

def Phi_tilda(xx,basis_tilda):
	N = len(xx)
	H = len(xx[1])
	return  np.array([basis_tilda(x) for x in xx]).reshape(N,H)

def sigmoid_tilda(params):
	alpha = params
	return lambda x: 0.5*sigmoid(alpha)(x)*sigmoid(alpha)(x)-1

def Id(params):
	return lambda x:x

def sigmoid(params):
	alpha = params
	return lambda x: (1-np.exp(-x+alpha))/(1+np.exp(-x+alpha))

def gaussian(params):
	mu = params[0]
	sigma = params[1]
	return lambda x:np.exp(-(x - mu)**2 / (2*sigma**2))
	
def gaussian_mu(params):
	mu = params[0]
	sigma = params[1]
	return lambda x:gaussian(params)(x)*((x-mu)/(sigma**2))

def gaussian_sigma(params):
	mu = params[0]
	sigma = params[1]
	return lambda x:gaussian(params)(x)*((x-mu)**2)/(sigma**3)

class LinearReg:
	def __init__(self,X_train,y_train,basis):
		self.Phi = Phi_matrix(X_train,basis)
		self.y = y_train
		self.x = X_train
		self.basis = basis
		eta = 0.1
		N = len(self.Phi[1])
		self.w = inv(self.Phi.T.dot(self.Phi)+eta*np.eye(N)).dot(self.Phi.T).dot(self.y)
		
	def predict(self,X):
		Phi = Phi_matrix(X,self.basis)
		return Phi.dot(self.w)
		
	def relative_error_list(self,X,y):
		y_pred = self.predict(X)
		return abs(y-y_pred)/y
	
	def relative_error_average(self,X,y):
		return np.average(self.relative_error_list(X,y))
	
	def relative_error_max(self,X,y):
		return max(self.relative_error_list(X,y))
		
	def grad_params(self,X,y,basis_tilda):
		y_tilda = -y+self.predict(X)
		phi_tilda = Phi_tilda(X,basis_tilda)
		return (2.0*self.w[1:]*(phi_tilda.T.dot(y_tilda)))[0]
		

#linear regression(basis function;id)
"""
class LinearRegId:
	def __init__(self,X_train,y_train,alpha):
		self.w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train+alpha)),X_train.T),y_train)

	def predict(self,X):
		return np.dot(X,self.w)
	
	def relative_error_list(self,X,y):
		y_pred = self.predict(X)
		return abs(y-y_pred)/y
	
	def relative_error_average(self,X,y):
		return np.average(self.relative_error_list(X,y))
		
	def relative_error_max(self,X,y):
		return max(self.relative_error_list(X,y))
"""
#gaussian process
class GPR:
	def __init__(self,X_train,y_train,kernel,eta):
		K = kernel_matrix(X_train,kernel,eta)
		self.K_inv = inv(K)
		self.y = y_train
		self.X = X_train
		self.kernel = kernel
		
	def predict(self,X_test):
		y_pred = []
		for x in X_test:
			k = kv(x,self.X,self.kernel)
			y_pred.append(k.T.dot(self.K_inv).dot(self.y))
		return y_pred
	
	def relative_error_list(self,X,y):
		y_pred = self.predict(X)
		return abs(y-y_pred)/y
		
	def relative_error_average(self,X,y):
		return np.average(self.relative_error_list(X,y))

def kernel_matrix(xx,kernel,eta):
	N = len(xx)
	return np.array( [kernel(xi,xj) for xi in xx for xj in xx] ).reshape(N,N)+eta*np.eye(N)

def kv(x,xtrain,kernel):
	return np.array([kernel(x,xi) for xi in xtrain])

def kgauss(params):
	[tau,sigma] = params
	return lambda x,y: params[0]*np.exp(-norm(x-y,ord=2)**2/(2*params[1]*params[1]))

def kgauss_tau(params):
	[tau,sigma] = params
	return lambda x,y: params[0]*np.exp(-norm(x-y,ord=2)**2/(2*params[1]*params[1]))

def kgauss_sigma(params):
	[tau,sigma] = params
	return lambda x,y: params[0]*np.exp(-norm(x-y,ord=2)**2/(2*params[1]*params[1]))*(norm(x-y,ord=2)**2.0)/params[1]



"""
train:validation:test = 8:1:1
"""
file = open("mpg.txt")
x = file.readlines()
file.close()

n = len(x)
n_test = int(n*0.2)
K = 5
W = (n-n_test)//K
n_val = W
n_train = n-n_test-n_val

for i in range(n):
	x[i] = x[i].strip().split("\t")
	x[i] = [float(s) for s in x[i]]
	#x[i].append(1.0)
x = np.array(x)


shuffle(x)
x_test = x[0:n_test]


K_cross = 5

shuffle(x)
x_test = x[0:n_test]
y_test,X_test = np.hsplit(x_test,[1])
x_other = x[n_test:]
X_train = [[]]*K_cross
y_train = [[]]*K_cross
X_val = [[]]*K_cross
y_val = [[]]*K_cross
W = len(x_other)//K_cross
for k in range(K_cross):
	y_val[k],X_val[k] = np.hsplit(x_other[k*W:k*W+n_val],[1])
	ind = np.ones(len(x_other), dtype=bool)
	ind[np.arange(k*W,k*W+n_val)] = False
	y_train[k],X_train[k] = np.hsplit(x_other[ind],[1])


"""
training and validation
"""
accuracy_list = np.array([1.0]*K_cross)
#linear regression(basis function:id)
for k in range(K_cross):
	model = LinearReg(X_train[k],y_train[k],Id(0.0))
	accuracy_list[k] = model.relative_error_average(X_val[k],y_val[k])

print("basis function:id")
print("average:",average(accuracy_list))
print("stdev:",stdev(accuracy_list))

#linear regression(basis function:sigmoid)
for k in range(K_cross):
	H = len(X_train[k][1])
	alpha = rand(H)*30.0-15.0
	dalpha = np.array([0.0001]*H)
	for i in range(10000):
		model = LinearReg(X_train[k],y_train[k],sigmoid(alpha))
		alpha -= model.grad_params(X_train[k],y_train[k],sigmoid_tilda(alpha)).T*dalpha
	model = LinearReg(X_train[k],y_train[k],sigmoid(alpha))
	accuracy_list[k] = model.relative_error_average(X_val[k],y_val[k])

print("basis function:sigmoid")
print("average:",average(accuracy_list))
print("stdev:",stdev(accuracy_list))

#linear regression(basis function:gaussian)
for k in range(K_cross):
	H = len(X_train[k][1])
	mu = rand(H)*2.0-1
	sigma = rand(H)*5.0
	params = np.array([mu,sigma])
	dmu = np.array([0.0001]*H)
	dsigma = np.array([0.0001]*H)
	for i in range(10000):
		model = LinearReg(X_train[k],y_train[k],gaussian(params))
		mu -= model.grad_params(X_train[k],y_train[k],gaussian_mu(params)).T*dmu
		sigma -= model.grad_params(X_train[k],y_train[k],gaussian_sigma(params)).T*dsigma
		params = np.array([mu,sigma])
	model = LinearReg(X_train[k],y_train[k],gaussian(params))
	accuracy_list[k] = model.relative_error_average(X_val[k],y_val[k])
print("basis function:gaussian")
print("average:",average(accuracy_list))
print("stdev:",stdev(accuracy_list))

#GPR
tau = 1000.0
sigma = 3000.0
eta = 0.1
params = [tau,sigma]
for k in range(K_cross):
	model = GPR(X_train[k],y_train[k],kgauss(params),eta)
	accuracy_list[k] = model.relative_error_average(X_val[k],y_val[k])

print("Gaussian Process regression")
print("average:",average(accuracy_list))
print("stdev:",stdev(accuracy_list))

accuracy_test = model.relative_error_average(X_test,y_test)
print(accuracy_test)

