import csv
from pylab import *

def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def initialize_centers(points, k):
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

def createEmptyLists(subList):
    myList = []
    for i in range(subList):
        myList.append([])
    return myList

def assignAllpoints(insts, centers):
    clusters = createEmptyLists(len(centers))
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



def main(file):
    data_points = loadData(file)
    data_points = np.array(data_points)
    KList=[2,3,5]
    for k in KList:
        centers = initialize_centers(data_points,k)
        result =kmeans(data_points,  centers)
        for i in range(0,k):
            d = result["clusters"][i]
            d = np.array(d)
            plot(d[:, 0], d[:, 1], '+')
        show()



