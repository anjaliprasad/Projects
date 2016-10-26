import numpy as np
import csv
from svmutil import *
from timeit import default_timer as timer
import libsvm as lib

def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    dataSet=[]
    for i in range(0, len(data)):
        ex = (data[i][0]).split('\t')
        dataSet.append(ex)
    for i in range(0, len(dataSet)):
        dataSet[i] = [float(x) for x in dataSet[i]]
    return dataSet

def replaceBinary(column):
    for i in range(0,len(column)):
        if column[i]==-1:
            column[i]=0
    return column

def makeThreeColumns(column):
    column1 = np.zeros(2000)
    column1=column1.reshape((2000,1))
    column2 = np.zeros(2000)
    column2 = column2.reshape((2000, 1))
    column3 = np.zeros(2000)
    column3 = column3.reshape((2000, 1))
    for i in range(0,len(column)):
        if(column[i] == -1):
            column1[i] = 1
            column2[i] = 0
            column3[i] = 0

        elif(column[i] == 0):
            column1[i] = 0
            column2[i] = 1
            column3[i] = 0
        else:
            column1[i] = 0
            column2[i] = 0
            column3[i] = 1

    concatenatedCol=np.concatenate((column1, column2, column3), axis=1)
    return concatenatedCol

def preProcessing(set):
    processed_Training_Set = np.empty([2000, 1])
    for i in range(0,len(set[0])):
        if len(np.unique(set[:,i]))==2 and (-1 in np.unique(set[:,i])) and (1 in np.unique(set[:,i])):
            #print "yo"
            set[:, i] = replaceBinary(set[:,i])
            if(i==0):
                processed_Training_Set = set[:, i]
                processed_Training_Set = processed_Training_Set.reshape((2000, 1))
            else:
                col = set[:,i].reshape((2000,1))
                processed_Training_Set = np.concatenate((processed_Training_Set,col),axis=1)
        elif len(np.unique(set[:,i]))==2 and (0 in np.unique(set[:,i])) and (1 in np.unique(set[:,i])):
            #print "okey"
            col = set[:, i].reshape((2000, 1))
            processed_Training_Set = np.concatenate((processed_Training_Set, col), axis=1)
        elif len(np.unique(set[:,i]))==3 and (-1 in np.unique(set[:,i])) and (1 in np.unique(set[:,i])) and (0 in np.unique(set[:,i])):
            #print "angie"
            concatenatedCol = makeThreeColumns(set[:,i])
            processed_Training_Set = np.concatenate((processed_Training_Set, concatenatedCol), axis=1)

    return np.transpose(processed_Training_Set).tolist()



def main():
    # Read data
    trainingSet = loadData("phishing-train-features.txt")
    trainingSet=np.array(trainingSet)
    trainingLabel=loadData("phishing-train-label.txt")
    trainingLabel=np.transpose(np.array(trainingLabel))
    testingSet = loadData("phishing-test-features.txt")
    testingSet = np.array(testingSet)
    testingLabel = loadData("phishing-test-label.txt")
    testingLabel = np.transpose(np.array(testingLabel))

    #preprocessing
    trainingSet = preProcessing(trainingSet)
    trainingSet = map(list, zip(*trainingSet))
    trainingLabel = preProcessing(trainingLabel)
    testingSet = preProcessing(testingSet)
    testingSet = map(list, zip(*testingSet))
    testingLabel = preProcessing(testingLabel)

    prob = svm_problem(trainingLabel[0], (trainingSet))



    gamma = [pow(4,-7), pow(4,-6), pow(4,-5), pow(4,-4), pow(4,-3), pow(4,-2), pow(4,-1)]

    C = [pow(4,-6), pow(4,-5), pow(4,-4), pow(4,-3), pow(4,-2), pow(4,-1), pow(4,0), pow(4,1), pow(4,2)]
    C1 = [pow(4, -3), pow(4, -2), pow(4, -1), pow(4, 0), pow(4, 1), pow(4, 2),pow(4, 3),pow(4, 4),pow(4, 5),pow(4, 6),pow(4, 7)]
    d=[1,2,3]
    print "linear kernel"
    for i in range(0,len(C)):
        print "\nfor c =",C[i]
        params =  '-c '+str(C[i])+' -v 3 -q'


        param = svm_parameter(params)
        #svm_train(trainingLabel,trainingSet,'-t 0 -c 1' )
        start = timer()
        m = svm_train(prob, param)
        end = timer()
        print "Average time:", (end - start) / 3
    print "\nPolynomial kernel"
    for i in range(0,len(C1)):

        for j in range(0,len(d)):
            print "\nfor c,d = ", C1[i],d[j]
            params = '-c '+str(C1[i])+' -v 3 -q' +' -t 1 -d '+ str(d[j])
            param = svm_parameter(params)
            start = timer()
            m = svm_train(prob, param)
            end = timer()
            print "Average time:", (end - start) / 3

    print "\nRBF kernel"
    for i in range(0, len(C1)):

        for j in range(0, len(gamma)):
            print "\nfor c,gamma = ", C1[i], gamma[j]
            params = '-c ' + str(C1[i]) + ' -v 3 -q'  +' -t 2 -g '+ str(gamma[j])
            param = svm_parameter(params)
            start = timer()
            m = svm_train(prob, param)
            end = timer()
            print "Average time:", (end - start) / 3
    #predict
    lib.main(prob,testingLabel,testingSet)

