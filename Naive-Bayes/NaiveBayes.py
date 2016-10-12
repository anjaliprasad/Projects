import csv
import math
import numpy as np

def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def classDivide(testingSet=[]):
    separatedOnClass = {}
    for i in range(len(testingSet)):
        row = testingSet[i]
        curClass = row[-1]
        if (curClass not in separatedOnClass):
            separatedOnClass[curClass] = []
        separatedOnClass[curClass].append(row)
   #print "Separated on class", separatedOnClass
    return separatedOnClass

def mean(nums):
    mean = sum(nums) / float(len(nums))
    return mean

def stdev(nums):
    avg = mean(nums)
    var = 0
    for x in range(len(nums)):
        var = var + (sum([pow(nums[x] - avg, 2)]) / float(len(nums) - 1 ))
    standard_deviation = math.sqrt(var)
    return standard_deviation

def summarize(set):
    for s in range(len(set)):
        del set[s][-1]
        del set[s][0]
        summary = [(mean(col), stdev(col)) for col in zip(*set)]
    return summary


def summarizeByClass(dataSet):
    sepClasses = classDivide(dataSet)
    sum = {}
    countOnClass = {}
    for cV, inst in sepClasses.iteritems():
        sum[cV] = summarize(inst)
        countOnClass[cV] = len(inst)
    return  sum,countOnClass

def gaussianFormula(x,mean,stdev):
    e = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    gaussian = (1/(math.sqrt(2*math.pi)*stdev)) * e
    return  gaussian

def calculateClassProb(sum, testSet, countOnClass):
    probs = {}
    for cV, cS in sum.iteritems():
        feature = countOnClass[cV] / float(get_TestingSetLen())
        probs[cV] = feature
        for i in range(0,len(cS)):
            mean,stdev = cS[i]
            if mean == 0 and stdev == 0 and testSet[i] != 0 :
                probs[cV] = probs[cV] * 0
            elif mean == 0 and stdev == 0 and testSet[i] == 0:
                probs[cV] = probs[cV] * 1
            else:
                x = testSet[i]
                probs[cV] = probs[cV] * gaussianFormula(x,mean,stdev)
    return probs

def predict(sum, testSet, countOnClass):
    probs = calculateClassProb(sum, testSet, countOnClass)
    classi = -1
    prob = -1
    for cV, cP in probs.iteritems():
        if classi == -1 or prob < cP:
            classi = cV
            prob = cP
    return classi

def letsPrediction(sum, countOnClass, testSet = [],):
    predictions = []
    for ip in range(0,len(testSet)):
        prediction = predict(sum, testSet[ip], countOnClass)
        predictions.append(prediction)
    return predictions


def getAccuracyPercentage(trainingSet, pred):
    counter=0
    for x in range(0,len(trainingSet)):
        if trainingSet[x][-1] == pred[x]:
            counter += 1
    length = float(len(trainingSet))
    accuracy = (counter / length) * 100.0
    return accuracy

def normalizeParams(trainingSet,testSet):

    stdev = np.std(trainingSet, axis=0)
    mean = np.mean(trainingSet, axis=0)
    for i in range(len(trainingSet)):
        for j in range(1,len(trainingSet[0])-1):
            trainingSet[i][j] = (1/stdev[j])*(trainingSet[i][j] - mean[j])
    for i in range(len(testSet)):
        for j in range(1,len(testSet[0])-1):
                testSet[i][j] = (1/stdev[j])*(testSet[i][j] - mean[j])

def set_TrainTestSetLen (trainingSetLen,testingSetLen):
   global Training_set_Len
   global Testing_set_Len
   Training_set_Len = trainingSetLen
   Testing_set_Len = testingSetLen

def get_TrainSetLen ():
   global Training_set_Len
   return Training_set_Len

def get_TestingSetLen ():
   global Testing_set_Len
   return Testing_set_Len

def NB():
    trainingSet = loadData("train.txt")
    testTrainingSet =  loadData("train.txt")
    testSet = loadData("test.txt")

    #for training data------------------------------------------------------
    set_TrainTestSetLen(len(trainingSet),len(trainingSet))

    summary, countOnClass = summarizeByClass(trainingSet)
    for t in range(0,len(testTrainingSet)):
        del testTrainingSet[t][0]
        del testTrainingSet[t][-1]
    predictions = letsPrediction(summary, countOnClass,testTrainingSet)
    trainingSet = loadData("train.txt")
    accuracy = getAccuracyPercentage(trainingSet,predictions)
    print "accuracy for training set", accuracy
    # for testing data------------------------------------------------------
    set_TrainTestSetLen(len(trainingSet),len(testSet))

    summary1,countOnClass1 = summarizeByClass(trainingSet)
    for t in range(0, len(testSet)):
        del testSet[t][0]
        del testSet[t][-1]
    predictions = letsPrediction(summary1, countOnClass1, testSet)
    testSet = loadData("test.txt")
    accuracy1 = getAccuracyPercentage(testSet, predictions)
    print "Accuracy for testing set", accuracy1

Training_set_Len = 0
Testing_set_len = 0

