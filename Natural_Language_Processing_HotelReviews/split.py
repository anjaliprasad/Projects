import numpy as np
from numpy import array

def loadData(fn):
    with open(fn) as f:
        mylist = f.read().splitlines()
    return mylist

def saveInFiles(train, test, train_l, test_l):
    with open("train-text1.txt", "w") as text_file:
        for i in range(0,len(train)):
            if(i==len(train)-1):
                text_file.write(train[i])
            else:
                text_file.write(train[i] + "\n")
    with open("train-text2.txt", "w") as text_file:
        for i in range(0,len(test)):
            if(i==len(test)-1):
                text_file.write(test[i])
            else:
                text_file.write(test[i] + "\n")

    with open("train-label1.txt", "w") as text_file:
        for i in range(0,len(train_l)):
            if(i==len(train_l)-1):
                text_file.write(train_l[i])
            else:
                text_file.write(train_l[i] + "\n")
    with open("train-label2.txt", "w") as text_file:
        for i in range(0,len(test_l)):
            if(i==len(test_l)-1):
                text_file.write(test_l[i])
            else:
                text_file.write(test_l[i] + "\n")
    return True

def NB():
    trainingSet = loadData("train-text.txt")
    testingSet = loadData("train-labels.txt")
    train = []
    test = []
    train_l = []
    test_l = []

    for i in range(0, 1280):
        if(i<960):
            train.append(trainingSet[i])
            train_l.append(testingSet[i])
        else:
            test.append(trainingSet[i])
            test_l.append(testingSet[i])
    flag = saveInFiles(train, test, train_l, test_l)

NB()