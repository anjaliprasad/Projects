from sklearn.datasets import load_boston
import numpy as np
import operator


def loadData(train_data, train_label, test_data, test_label):
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


def normalizeParams(trainingSet, testSet):
    stdev = np.std(trainingSet, axis=0)
    mean = np.mean(trainingSet, axis=0)
    for i in range(len(trainingSet)):
        for j in range(0, len(trainingSet[0])):
            trainingSet[i][j] = (1 / stdev[j]) * (trainingSet[i][j] - mean[j])
    for i in range(len(testSet)):
        for j in range(0, len(testSet[0])):
            testSet[i][j] = (1 / stdev[j]) * (testSet[i][j] - mean[j])


def MSE(y, prediction):
    mse = (sum(np.square(y - prediction))) / len(y)
    return mse[0, 0]


def k_fold(train_data1, train_label1, X, y, X1, y1):
    # print "train_data 44", type(train_data1)
    fea = train_data1.shape[1]
    A = np.insert(train_data1, fea, train_label1, axis=1)
    np.random.shuffle(A)

    train_label_ran = A[:, fea]
    train_data_ran = A[:, [range(0,fea)]]
    MSE_test_avg_list = []
    minMSE = 99999
    minLam = -10

    lam = 0.0001
    while lam <= 10:
        train_data = []
        train_label = []
        test_data = []
        test_label = []

        data_set = np.array_split(train_data_ran, 10)
        data_target = np.array_split(train_label_ran, 10)
        MSE_test_avg = []

        for i in range(0, 10):
            # print "iter",i
            test_data = data_set[i]
            test_label = data_target[i]
            # print 'shape of testing', np.shape(test_data)

            train_data = np.concatenate(np.delete(data_set, i, 0))
            train_label = np.concatenate(np.delete(data_target, i, 0))
            # print 'training section ::', np.shape(train_data)


            X_train = np.mat(train_data)
            X_train = np.insert(X_train, 0, 1, axis=1)
            y_train = (np.mat(train_label)).transpose((1, 0))

            X1_test = np.mat(test_data)
            X1_test = np.insert(X1_test, 0, 1, axis=1)
            y1_test = (np.mat(test_label)).transpose((1, 0))

            theta = (np.linalg.pinv((X_train.T * X_train) + (lam * np.identity(X_train.shape[1])))) * (
            X_train.T * y_train)

            # prediction train
            prediction_train = np.mat(np.zeros(X_train.shape[0])).transpose(((1, 0)))
            for i in range(0, X_train.shape[0]):
                prediction_train[i] = X_train[i] * theta

            MSE_train = MSE(y_train, prediction_train)

            # print "For lambda = ", lam
            # print "MSE for training set", MSE_train


            # prediction test
            prediction_test = np.mat(np.zeros(X1_test.shape[0])).transpose(((1, 0)))
            for i in range(0, X1_test.shape[0]):
                prediction_test[i] = X1_test[i] * theta

            MSE_test = MSE(y1_test, prediction_test)
            MSE_test_avg.append(MSE_test)

        # print "MSE for testing set", MSE_test
        # print "MSE_test_avg",np.sum(MSE_test_avg)/10.0
        avg_mse = [np.sum(MSE_test_avg) / 10.0, lam]
        MSE_test_avg_list.append(avg_mse)

        if avg_mse < minMSE:
            minMSE = avg_mse
            minLam = lam
        lam = lam + 0.1

    MSE_test_avg_list.sort(key=operator.itemgetter(0))

    print "winning lambda = ", MSE_test_avg_list[0][1]
    w_lam = MSE_test_avg_list[0][1]

    theta = (np.linalg.pinv((X.T * X) + (w_lam * np.identity(X.shape[1])))) * (X.T * y)

    prediction1 = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
    for i in range(0, X.shape[0]):
        prediction1[i] = X[i] * theta
    MSE_train = MSE(y, prediction1)
    print "MSE for training set after using winning lambda", MSE_train

    prediction2 = np.mat(np.zeros(X1.shape[0])).transpose(((1, 0)))
    for i in range(0, X1.shape[0]):
        prediction2[i] = X1[i] * theta
    MSE_test = MSE(y1, prediction2)
    print "MSE for training set after using winning lambda", MSE_test


def RR():
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    loadData(train_data, train_label, test_data, test_label)
    normalizeParams(train_data, test_data)

    X = np.mat(train_data)
    X = np.insert(X, 0, 1, axis=1)
    y = (np.mat(train_label)).transpose((1, 0))

    X1 = np.mat(test_data)
    X1 = np.insert(X1, 0, 1, axis=1)
    y1 = (np.mat(test_label)).transpose((1, 0))
    lam = [0.01, 0.1, 1.0]
    for l in range(0, len(lam)):

        theta = (np.linalg.pinv((X.T * X) + (lam[l] * np.identity(X.shape[1])))) * (X.T * y)

        prediction = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
        for i in range(0, X.shape[0]):
            prediction[i] = X[i] * theta

        MSE_train = MSE(y, prediction)
        print "For lambda = ", lam[l]
        print "MSE for training set", MSE_train
        prediction = np.mat(np.zeros(X1.shape[0])).transpose(((1, 0)))
        for i in range(0, X1.shape[0]):
            prediction[i] = X1[i] * theta

        MSE_test = MSE(y1, prediction)
        print "MSE for testing set", MSE_test

    print "Please wait for few seconds for results"
    k_fold(np.asarray(train_data), np.asarray(train_label), X, y, X1, y1)






