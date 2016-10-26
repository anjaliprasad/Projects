from svmutil import *

#C1 = [pow(4, -3), pow(4, -2), pow(4, -1), pow(4, 0), pow(4, 1), pow(4, 2), pow(4, 3), pow(4, 4), pow(4, 5), pow(4, 6),
#      pow(4, 7)]
#gamma = [pow(4, -7), pow(4, -6), pow(4, -5), pow(4, -4), pow(4, -3), pow(4, -2), pow(4, -1)]

def main(prob,testingLabel,testingSet):
    print "\nPrediction using best model"
    param_str = svm_parameter('-s 0 -t 2 -c {0} -g {1} -q'.format(1, 4 ** -1))
    model = svm_train(prob, param_str)
    svm_predict(testingLabel[0], testingSet, model)
