import matplotlib.pyplot as plt
import operator
from sklearn.datasets import load_boston
import numpy as np
import numpy.matlib

def loadData(train_data,train_label,test_data,test_label):
    boston = load_boston()
    X = np.array(boston.data)
    y = np.array(boston.target)

    for i in range(0, len(X)):
        if i % 7 == 0:
            test_data.append(np.array(X[i, :]))
            test_label.append(np.array(y[i]))
        else:
            train_data.append(np.array(X[i, :]))
            train_label.append(np.array(y[i]))

def pearsonCorrelation(x, y):

    n = len(x)
    xy = np.dot(x, y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    square_x = np.dot(x,x)
    square_y = np.dot(y,y)
    square_sum_x = sum_x * sum_x
    square_sum_y = sum_y * sum_y

    pearson_coefficient = ((n * xy) - (sum_x * sum_y)) / np.sqrt((n*square_x - square_sum_x)*(n*square_y - square_sum_y))


    return pearson_coefficient


def plotHistogram(train_data, train_label):
    train_data = np.array(train_data)
    pearsonList = []
    print "3.1 - Linear Regression"
    for i in range(0,len(train_data[0])):
        col = train_data[:,i]
        plt.hist(col)
        plt.title("Histogram for feature")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        p = pearsonCorrelation(col, train_label)
        print "Pearson's Correlation for column ", i + 1, " is ", p
        pearsonList.append([i+1,np.absolute(p)])

    pearsonList.sort(key=operator.itemgetter(1))
    pearsonList.reverse()

    return pearsonList



def normalizeParams(trainingSet,testSet):
    stdev = np.std(trainingSet, axis=0)
    mean = np.mean(trainingSet, axis=0)
    for i in range(len(trainingSet)):
        for j in range(0,len(trainingSet[0])):
            trainingSet[i][j] = (1/stdev[j])*(trainingSet[i][j] - mean[j])
    for i in range(len(testSet)):
        for j in range(0,len(testSet[0])):
                testSet[i][j] = (1/stdev[j])*(testSet[i][j] - mean[j])


def MSE(y,prediction):
    mse = (sum(np.square(y - prediction)))/len(y)
    return mse[0,0]

def top_four(X,y,X1,y1):
    theta = (np.linalg.pinv((X.T * X))) * (X.T * y)

    prediction = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
    for i in range(0, X.shape[0]):
        prediction[i] = X[i] * theta

    MSE_train = MSE(y, prediction)
    prediction = np.mat(np.zeros(X1.shape[0])).transpose(((1, 0)))
    for i in range(0, X1.shape[0]):
        prediction[i] = X1[i] * theta

    MSE_test = MSE(y1, prediction)
    return MSE_train,MSE_test

def residual(X,y,X1,y1):
    theta = (np.linalg.pinv((X.T * X))) * (X.T * y)

    prediction1 = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
    for i in range(0, X.shape[0]):
        prediction1[i] = X[i] * theta

    residual_train = np.array(y - prediction1)
    #print "Residual for training set", residual_train
    prediction2 = np.mat(np.zeros(X1.shape[0])).transpose(((1, 0)))
    for i in range(0, X1.shape[0]):
        prediction2[i] = X1[i] * theta

    residual_test = np.array(y1 - prediction2)
    #print "Residual for testing set", residual_test

    return residual_train,residual_test



def pearsonCall(train_data, train_label):
    pearsonList = []
    for i in range(0,train_data.shape[1]):
        col = train_data[:,i]

        p = pearson(col, train_label)
        pearsonList.append([i,np.absolute(p)])

    pearsonList.sort(key=operator.itemgetter(1))
    pearsonList.reverse()
    return pearsonList

def pearson(x, y):
    n = x.shape[0]
    xy = np.sum(np.multiply(x, y))
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    square_x = np.sum(np.multiply(x, x))
    square_y = np.sum(np.multiply(y, y))
    square_sum_x = sum_x * sum_x
    square_sum_y = sum_y * sum_y
    pearson_coefficient = ((n * xy) - (sum_x * sum_y)) / np.sqrt((n * square_x - square_sum_x) * (n * square_y - square_sum_y))
    return pearson_coefficient

def findResidual(prediction,y):
    return y-prediction

def residualMethod(X,y,X1,y1):
    featureSet =[]

    pearsonsList = pearsonCall(X, y)
    theta = np.matlib.zeros((X.shape[1], 1))
    X_new = np.matlib.ones((X.shape[0], 1))
    while len(featureSet)<4:

        for i in range(0,len(pearsonsList)):
            if pearsonsList[i][0] not in featureSet:
                featureSet.append(pearsonsList[i][0])
                break
            else:
                i=i+1
        X_new = np.matlib.ones((X.shape[0],1))

        for i in range(0,len(featureSet)):
            col = X[:,featureSet[i]]
            X_new = np.hstack((X_new, col))
        #theta
        theta = (np.linalg.pinv((X_new.T * X_new))) * (X_new.T * y)
        #prediction
        prediction = np.mat(np.zeros(X_new.shape[0])).transpose(((1, 0)))
        for i in range(0, X_new.shape[0]):
            prediction[i] = X_new[i] * theta
        residual = findResidual(prediction,y)
        pearsonsList = pearsonCall(X, residual)

    print "\n"
    print "3.3b"
    print "Select 4 features iteratively and select top 4:"
    for i in range(0, len(featureSet)):
        print featureSet[i]+1
    X1_new = np.matlib.ones((X1.shape[0], 1))
    for i in range(0, len(featureSet)):
        col = X1[:, featureSet[i]]
        X1_new = np.hstack((X1_new, col))

    prediction1 = np.mat(np.zeros(X_new.shape[0])).transpose(((1, 0)))
    for i in range(0, X_new.shape[0]):
        prediction1[i] = X_new[i] * theta

    MSE_train = MSE(y, prediction1)
    print "MSE_train",MSE_train

    prediction2 = np.mat(np.zeros(X1_new.shape[0])).transpose(((1, 0)))
    for i in range(0, X1_new.shape[0]):
        prediction2[i] = X1_new[i] * theta

    MSE_test = MSE(y1, prediction2)
    print "MSE_test", MSE_test


def bruteForce(X,y,X1,y1):
    print "\n"
    print "3c: brute force"
    print "Brute force: Please wait for few seconds for this result"
    mse_list = []

    for i in range(1, X.shape[1]+1):
        for j in range(1,  X.shape[1]+1):
            if j != i and j > i:
                for k in range(1,  X.shape[1]+1):
                    if k != i and k != j and k > i and k > j:

                        for l in range(1, X.shape[1]):
                            if l != i and l != j and l != k and l > i and l > j and l > k:
                                X_top = X[:, [0, i, j, k, l]]
                                y_top = y
                                X1_top = X1[:, [0, i, j, k, l]]
                                y1_top = y1
                                MSE_train, MSE_test = top_four(X_top, y_top, X1_top, y1_top)
                                mse_list.append([MSE_train, MSE_test, i, j, k, l])

    mse_list.sort(key=operator.itemgetter(0, 1))
    print "MSE_train for brute force: ",mse_list[0][0]
    print "MSE_test for brute force: ", mse_list[0][1]
    "columns selected in brute force:"
    for i in range(2,6):
        print mse_list[0][i]


def polynomialFeatureExpansion(X,y,X1,y1):
    fea = X.shape[1] + (1 + X.shape[1]) * X.shape[1] / 2
    fea=fea-X.shape[1]
    examples = X.shape[0]
    testExamples=X1.shape[0]

    X_new =  np.matlib.zeros((examples, fea))
    k=0
    while(k<fea):
        for i in range(0,X.shape[1]):
            for j in range(i,X.shape[1]):
                X_new[:,k] = np.multiply(X[:,i], X[:,j])
                k=k+1

    #X_new=np.insert(X_new, 0, 1, axis=1)
    all_data = np.hstack((X, X_new))

    X1_new = np.matlib.zeros((testExamples, fea))
    k = 0
    while (k < fea):
        for i in range(0, X1.shape[1]):
            for j in range(i, X1.shape[1]):
                X1_new[:, k] = np.multiply(X1[:, i], X1[:, j])
                k = k + 1

    #X1_new = np.insert(X1_new, 0, 1, axis=1)
    all_data1 = np.hstack((X1, X1_new))
    all_data_arr = np.asarray(all_data)
    all_data1_arr = np.asarray(all_data1)
    normalizeParams(all_data_arr,all_data1_arr)
    all_data = np.mat(all_data_arr)
    all_data1 = np.mat(all_data1_arr)
    X1_concatenated = np.insert(all_data1, 0, 1, axis=1)
    X_concatenated = np.insert(all_data, 0, 1, axis=1)
    print "\n"
    print "3d:polynomial expansion"
    MSE_train, MSE_test = top_four(X_concatenated, y, X1_concatenated, y1)
    print "MSE ater polynomial feature expansion on train set:",MSE_train
    print "MSE ater polynomial feature expansion on test set:", MSE_test

def LR():
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    loadData(train_data,train_label,test_data,test_label)

    pearsonsList = plotHistogram(train_data, train_label)
    normalizeParams(train_data, test_data)
    X1_withoutOne = np.mat(test_data)
    X_withoutOne = np.mat(train_data)

    X =  np.mat(train_data)
    X = np.insert(X, 0, 1, axis=1)
    y = (np.mat(train_label)).transpose((1,0))


    X1 = np.mat(test_data)
    X1 = np.insert(X1, 0, 1, axis=1)
    y1 = (np.mat(test_label)).transpose((1, 0))
    MSE_train, MSE_test = top_four(X,y,X1,y1)
    print "\n"
    print "3.2 -  Linear Regression"
    print "MSE for training set", MSE_train
    print "MSE for testing set", MSE_test

    #---------------------------------------------------------
    print "\n"
    print "3.3a"
    print "Top four correlated columns with the target are",pearsonsList[0][0],pearsonsList[1][0],pearsonsList[2][0],pearsonsList[3][0]
    print "MSE after taking top four correlated columns with the target are:"
    X_top = X[:,[0,pearsonsList[0][0],pearsonsList[1][0],pearsonsList[2][0],pearsonsList[3][0]]]
    y_top = y
    X1_top = X1[:,[0,pearsonsList[0][0],pearsonsList[1][0],pearsonsList[2][0],pearsonsList[3][0]]]
    y1_top = y1
    MSE_train, MSE_test = top_four(X_top, y_top, X1_top, y1_top)
    print "MSE for training set", MSE_train
    print "MSE for testing set", MSE_test
    #-----------------------------------------------------------
    residualMethod(X_withoutOne,y,X1_withoutOne,y1)
    #-----------------------------------------------------------
    bruteForce(X,y,X1,y1)
    polynomialFeatureExpansion(X_withoutOne,y,X1_withoutOne,y1)





