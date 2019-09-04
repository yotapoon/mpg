tau = 1000.0
sigma = 3000.0
eta = 0.1
params = [tau,sigma]
for k in range(K_cross):
	model = GPR(X_train[k],y_train[k],kgauss(params),eta)
	accuracy_list[k] = model.relative_error_average(X_val[k],y_val[k])
print("average:",average(accuracy_list))
print("stdev:",stdev(accuracy_list))

"""
eta = 0.1
dtau = 0.01
dsigma = 0.001
for i in range(100):
	params = [tau,sigma]
	print(i,params)
	for k in range(K_cross):
		K = kernel_matrix(X_val[k],kgauss(params),eta)
		K_tau = kernel_matrix(X_val[k],kgauss(params),0.0)
		grad_tau = -np.trace(inv(K).dot(K_tau))+(inv(K).dot(y_val[k])).T.dot(K_tau).dot(inv(K)).dot(y_val[k])
		K_sigma = kernel_matrix(X_val[k],kgauss_sigma(params),0.0)
		grad_sigma = -np.trace(inv(K).dot(K_sigma))+(inv(K).dot(y_val[k])).T.dot(K_sigma).dot(inv(K)).dot(y_val[k])
		tau += grad_tau*dtau/tau
		sigma += grad_sigma*dsigma/sigma
		
"""