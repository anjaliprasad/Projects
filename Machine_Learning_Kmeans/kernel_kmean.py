# necessary imports
import csv
from pylab import *
import numpy as np


def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def initialize_centroids(points, k):
    return points[:k]

def meanofInstance(name, instList):
    numInst = len(instList)
    if (numInst == 0):
        return
    numAttr = len(instList[0])
    means = [name] + [0] * (numAttr)
    for inst in instList:

        for i in range(1, numAttr+1):
            means[i] += inst[i-1]
    for i in range(1, numAttr+1):
        means[i] /= float(numInst)
    return tuple(means)

def computeCenters(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanofInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids


def distance(inst1, inst2):
    sumSquares = 0
    if(len(inst1)==len(inst2)):
        for i in range(0, len(inst1)):
            sumSquares += ((inst1[i] - inst2[i])**2)
    else:
        for i in range(0, len(inst1)):
            sumSquares += ((inst1[i] - inst2[i+1])**2)
    return sumSquares

def assign(inst, center):
    minDist= distance(inst, center[0])
    minDistIndex = 0

    for i in range(1, len(center)):
        d = distance(inst, center[i])
        if (d < minDist):
            minDist = d
            minDistIndex = i

    return minDistIndex

def createEmptyList(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAllpoints(insts, centers):
    clusters = createEmptyList(len(centers))
    for inst in insts:
        clusterIndex = assign(inst, centers)
        clusters[clusterIndex].append(inst)
    return clusters

def kmeans(instances, initCenters):
    kresult = {}
    centers = initCenters
    prevCenters = []
    iteration = 0

    while (centers != prevCenters):

        iteration += 1
        clusters = assignAllpoints(instances, centers)
        prevCenters = centers
        centers = computeCenters(clusters)
        r = np.array(clusters)
    kresult["clusters"] = clusters
    kresult["centroids"] = centers
    return kresult

def MappingToOriginalSpace(d,points):
    x_rows = d[:,0].shape[0]
    x = np.reshape(np.power(d[:,0],0.5),(x_rows,1))
    y = np.reshape(np.power(d[:,1],0.5),(x_rows,1))
    originalPoints = np.hstack((x, y))
    list_new = []
    for i in range(0,originalPoints.shape[0]):
        for i1 in range(0,points.shape[0]):
            if((originalPoints[i] == np.absolute(points[i1])).all()):
                list_new.append(points[i1])
    return list_new

def main(file):
    points = loadData(file)
    points = np.array(points)
    xSqr= np.reshape(np.power(points[:,0],2),(points.shape[0],1))
    ysqr = np.reshape(np.power(points[:,1],2),(points.shape[0],1))
    xysum=np.reshape(xSqr+ysqr,(points.shape[0],1))
    kernelPoints = np.hstack((xSqr, ysqr, xysum))

    KList=[2]
    for k in KList:
        centroids = initialize_centroids(kernelPoints, k)
        result =kmeans(kernelPoints,  centroids)
        for i in range(0,k):
            d = result["clusters"][i]
            d = np.array(d)
            originalPoints = np.array(MappingToOriginalSpace(d,points))
            plot(originalPoints[:, 0], originalPoints[:, 1], '+')
        show()


