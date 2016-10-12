import csv
import math
import operator
import numpy as np


def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def euclideanDistance(inst1, inst2, len):
	dist = 0
	for x in range(1,len-1):
		dist += pow(((inst1[x]) - (inst2[x])), 2)

	return math.sqrt(dist)

def manhattanDistance(inst1, inst2, len):
	dist = 0
	for x in range(1,len-1):
         dist += abs(float(inst1[x]) - float(inst2[x]))
	return dist

def findNeighbor(trainingSet,testingSet, k, leaveOneOutFlag,distanceType):
    neighbor = []
    count=0
    cols = len(trainingSet[0])
    for x in range(0,len(testingSet)):
        dists = []
        for y in range(0,len(trainingSet)):
            if leaveOneOutFlag == 0 :
                if distanceType == 0:
                    edist = manhattanDistance(testingSet[x],trainingSet[y],cols)
                else:
                    edist = euclideanDistance(testingSet[x], trainingSet[y], cols)
                dists.append((testingSet[x], trainingSet[y], edist))
            else:
                if testingSet[x][0] != trainingSet[y][0]:
                    if distanceType == 0:
                        edist = manhattanDistance(testingSet[x], trainingSet[y], cols)
                    else:
                        edist = euclideanDistance(testingSet[x], trainingSet[y], cols)
                    dists.append((testingSet[x], trainingSet[y], edist))
        dists.sort(key=operator.itemgetter(2))
        for z in range(0,k):
            neighbor.append(dists[z])
    return neighbor

def getRes(neighbors):
    Vote = {}
    for x in range(0,len(neighbors)):
        resp = neighbors[x][-1]
        if resp in Vote:
            Vote[resp] += 1
        else:
            Vote[resp] = 1
    sortedVotes = sorted(Vote.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracyPercentage(trainingSet, pred):
    counter=0
    for x in range(0,len(trainingSet)):
        if trainingSet[x][-1] == pred[x]:
            counter += 1
    length = float(len(trainingSet))
    accuracy = (counter / length) * 100.0
    return accuracy


def printResult(k, accuracy1,accuracy2,accuracy3,accuracy4, accuracy5, accuracy6):
    print "K =", k
    print "Euclidean distance (L2):"
    print "Training set accuracy", accuracy1
    print "Test set accuracy", accuracy3
    print "WITHOUT leaving one out on training set", accuracy5
    print "Manhattan distance (L1):"
    print "Training set accuracy", accuracy2
    print "Test set accuracy", accuracy4
    print "WITHOUT leaving on out on training set", accuracy6

def testingResult(neighbor, k, testSet):
    votes = []
    finalClassSelectionList = []
    count = 0
    for r in range(0, len(neighbor)):
        if count < k:
            votes.append((neighbor[r][1]))
            count += 1
        else:
            resp = getRes(votes)
            finalClassSelectionList.append(resp)
            count = 1
            votes = []
            votes.append((neighbor[r][1]))
        if r == len(neighbor) - 1 and votes != []:
            response = getRes(votes)
            finalClassSelectionList.append(response)
    accuracyPercentage1 = getAccuracyPercentage(testSet, finalClassSelectionList)
    return accuracyPercentage1

def normalizeParams(trainingSet,testSet):

    stdev = np.std(trainingSet, axis=0)
    mean = np.mean(trainingSet, axis=0)
    for i in range(len(trainingSet)):
        for j in range(1,len(trainingSet[0])-1):
            trainingSet[i][j] = (1/stdev[j])*(trainingSet[i][j] - mean[j])
    for i in range(len(testSet)):
        for j in range(1,len(testSet[0])-1):
                testSet[i][j] = (1/stdev[j])*(testSet[i][j] - mean[j])

def set_TrainSetLen (trainingSetLen):
   global Training_set_Len
   Training_set_Len = trainingSetLen
   # do things to x
   return Training_set_Len

def get_TrainSetLen ():
   global Training_set_Len
   return Training_set_Len

def KNN1():

    kArray = [1,3,5,7]
    trainingSet = loadData("train.txt")
    trainingSet = np.array(trainingSet)

    testSet = loadData("test.txt")
    testSet = np.array(testSet)
    set_TrainSetLen(len(trainingSet))
    x=get_TrainSetLen()
    normalizeParams(trainingSet,testSet)
    #--------------------------------------------------------------------------
    for j in range(0,len(kArray)):
        k=kArray[j]
        #manhattan distance for testing set
        neighbor = findNeighbor(trainingSet, testSet, k, 0, 0)
        #Euclidean distance for testing set
        testingAccuracyM = testingResult(neighbor, k, testSet)
        neighbor = findNeighbor(trainingSet, testSet, k, 0, 1)
        testingAccuracyE = testingResult(neighbor, k, testSet)

        #----------------------------------------------------------------------
        # manhattan distance for training set
        neighbor1 = findNeighbor(trainingSet, trainingSet, k, 1, 0)
        trainingAccuracyM = testingResult(neighbor1, k, trainingSet)
        # Euclidean distance for training set
        neighbor1 = findNeighbor(trainingSet, trainingSet, k, 1, 1)
        trainingAccuracyE = testingResult(neighbor1, k, trainingSet)
        #-----------------------------------------------------------------------
        #Without leaving one out in training set
        # manhattan distance for training set
        neighbor1 = findNeighbor(trainingSet, trainingSet, k, 0, 0)
        noLeavetrainingAccuracyM = testingResult(neighbor1, k, trainingSet)
        # Euclidean distance for training set
        neighbor1 = findNeighbor(trainingSet, trainingSet, k, 0, 1)
        noLeavetrainingAccuracyE = testingResult(neighbor1, k, trainingSet)
        printResult(k, trainingAccuracyE ,trainingAccuracyM ,testingAccuracyE,testingAccuracyM, noLeavetrainingAccuracyE,noLeavetrainingAccuracyM )
Training_set_Len = 0


