import hw_utils as h
import math
from timeit import default_timer as timer


def main():
	X_train, y_train, X_test, y_test = h.loaddata("MiniBooNE_PID.txt")
	X_train_n,X_test_n = h.normalize(X_train, X_test)

	#a)linear activation
	h.testmodels(X_train_n, y_train, X_test_n, y_test, [[50, 2], [50, 50, 2], [50, 50, 50, 2], [50, 50, 50, 50, 2]],
			   actfn='linear', last_act='softmax', reg_coeffs=[0.0],
			   num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
			   sgd_Nesterov=False, EStop=False, verbose=0)
	start = timer()
	h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]],
			   actfn='linear', last_act='softmax', reg_coeffs=[0.0],
			   num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
			   sgd_Nesterov=False, EStop=False, verbose=0)
	end = timer()
	print "Time taken for linear activation:", (end - start)

# Sigmoid activation
	start = timer()
	h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]],
			   actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0],
			   num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
			   sgd_Nesterov=False, EStop=False, verbose=0)
	end = timer()
	print "Time taken for Sigmoid activation:", (end - start)

	# Relu activation
	start = timer()
	h.testmodels(X_train_n, y_train, X_test_n, y_test, [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2],[50, 800, 500, 300, 2],[50, 800, 800, 500, 300, 2]],
			   actfn='relu', last_act='softmax', reg_coeffs=[0.0],
			   num_epoch=30, batch_size=1000, sgd_lr=5 * math.pow(10,-4), sgd_decays=[0.0], sgd_moms=[0.0],
			   sgd_Nesterov=False, EStop=False, verbose=0)
	end = timer()
	print "Time taken for Relu activation:", (end - start)

	#L2-Regularization
	reg_coeffs = [math.pow(10,-7),5*math.pow(10,-7),math.pow(10,-6), 5 * math.pow(10,-6), math.pow(10,-5)]

	best_config = h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 800, 500, 300, 2]],
				actfn='relu', last_act='softmax', reg_coeffs=reg_coeffs,
				num_epoch=30, batch_size=1000, sgd_lr=5 * math.pow(10,-4), sgd_decays=[0.0], sgd_moms=[0.0],
				sgd_Nesterov=False, EStop=False, verbose=0)
	best_reg_coeff = best_config[1]
	print "Best Regularization coefficient is: ", best_reg_coeff

	#Early Stopping and L2-regularization
	best_config = h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 800, 500, 300, 2]],
			   actfn='relu', last_act='softmax', reg_coeffs=reg_coeffs,
			   num_epoch=30, batch_size=1000, sgd_lr=5 * math.pow(10,-4), sgd_decays=[0.0], sgd_moms=[0.0],
			   sgd_Nesterov=False, EStop=True, verbose=0)
	best_reg_coeff_ES = best_config[1]
	print "Best Regularization coefficient with early stopping is: ", best_reg_coeff_ES

	# SGD with weight decay
	decay = [math.pow(10, -5), 5 * math.pow(10, -5), math.pow(10, -4), 3 * math.pow(10, -4), 7 * math.pow(10, -4),
			 math.pow(10, -3)]
	best_config = h.testmodels(X_train_n, y_train, X_test_n, y_test,
							[[50, 800, 500, 300, 2]],
							actfn='relu', last_act='softmax', reg_coeffs=[5 * math.pow(10, -7)],
							num_epoch=100, batch_size=1000, sgd_lr=math.pow(10, -5), sgd_decays=decay, sgd_moms=[0.0],
							sgd_Nesterov=False, EStop=False, verbose=0)
	best_decay = best_config[2]
	print "Best weight decay is: ", best_decay

	# Momentum
	momentum = [0.99, 0.98, 0.95, 0.9, 0.85]
	best_config = h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 800, 500, 300, 2]],
			   actfn='relu', last_act='softmax', reg_coeffs=[0.0],
			   num_epoch=50, batch_size=1000, sgd_lr=math.pow(10, -5), sgd_decays=[best_decay], sgd_moms=momentum,
			   sgd_Nesterov=True, EStop=False, verbose=0)
	best_momentum = best_config[3]
	print "Best momentum is: ", best_momentum

	# Combining the above
	h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 800, 500, 300, 2]],
			   actfn='relu', last_act='softmax', reg_coeffs=[best_reg_coeff],
			   num_epoch=100, batch_size=1000, sgd_lr=math.pow(10, -5), sgd_decays=[best_decay], sgd_moms=[best_momentum],
			   sgd_Nesterov=True, EStop=True, verbose=0)


	# Grid search with cross-validation
	decay_1 = [math.pow(10, -5), 5 * math.pow(10, -5), math.pow(10, -4)]
	h.testmodels(X_train_n, y_train, X_test_n, y_test,
			   [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]],
			   actfn='relu', last_act='softmax', reg_coeffs=reg_coeffs,
			   num_epoch=100, batch_size=1000, sgd_lr=math.pow(10, -5), sgd_decays=decay_1, sgd_moms=[0.99],
			   sgd_Nesterov=True, EStop=True, verbose=0)
main()