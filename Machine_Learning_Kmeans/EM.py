import csv
from pylab import *
import random
from scipy.stats import multivariate_normal

def loadData(fn):
    file = csv.reader(open(fn, "rb"))
    data = list(file)
    for i in range(0, len(data)):
        data[i] = [float(x) for x in data[i]]
    return data

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def initialize(k,points):
    meanList=[]
    centroids = points.copy()
    np.random.shuffle(centroids)
    p_z = np.zeros(k)
    covList = []
    initial_p_z = 1.0/k
    for i in range(0,len(p_z)):
        p_z[i] = initial_p_z

    meanList=centroids[:k]
    meanList = np.array(meanList)
    cov = np.cov(points.transpose(((1, 0))))


    for i in range(0,k):
        covList.append(cov)

    return p_z, meanList, covList

def multivariate_gaussian(mean,cov,points, p_z):
    #print cov
    f = multivariate_normal.pdf(points,mean,cov)
    return (f * p_z).reshape((points.shape[0],1))

def E_Step(p_z, meanList, covList, k, points):
    w_j = []
    sum_gaussian = np.zeros((points.shape[0], 1))
    for i in range(0,k):
        gaussian =  multivariate_gaussian(meanList[i],covList[i], points, p_z[i])
        sum_gaussian = sum_gaussian + gaussian

    for i in range(0,k):
        w_j.append(multivariate_gaussian(meanList[i],covList[i], points, p_z[i])/sum_gaussian)
    w_j_combined = np.hstack((w_j[0], w_j[1], w_j[2]))
    return w_j_combined

def M_Step(w_j_combined,points,k):
    covList = []
    mean = np.zeros((k,len(points[0])))
    p_z = np.zeros(k)
    for i in range(0,k):
        p_z[i]=np.sum(w_j_combined[:,i])/len(points)
    for i in range(0,k):
        for j in range(0,len(points[0])):
           mean[i][j] = np.dot(w_j_combined[:,i],points[:,j])/np.sum(w_j_combined[:,i])

    for i in range(0,k):
        cov=np.zeros((2,2))
        for j in range(0,len(points)):
            cov =cov + (w_j_combined[j,i] * ((points[j]-mean[i]).reshape((2,1)).transpose((1,0)) * (points[j]-mean[i]).reshape((2,1))))
        totalCov=np.sum(w_j_combined[:,i])
        cov=cov/totalCov
        covList.append(cov)
    total_log_li = log_liklihood(k,mean,covList,points,p_z)
    return  p_z,mean,covList, total_log_li

def log_liklihood(k,meanList,covList,points,p_z):
    sum_gaussian = np.zeros((points.shape[0], 1))
    total_log_li=0
    for i in range(0,k):
        gaussian =  multivariate_gaussian(meanList[i],covList[i], points, p_z[i])
        sum_gaussian = sum_gaussian + gaussian
    log_li = np.log(sum_gaussian)
    for j in range(0,len(sum_gaussian)):
        total_log_li = total_log_li + log_li[j]

    return total_log_li[0]

def cluster_assign(instance, w_j):
    max_prob = w_j[0]
    max_index = 0

    for i in range(1, len(w_j)):
        if (max_prob < w_j[i]):
            max_prob = w_j[i]
            max_index = i
    return max_index

def createClusters( points,k,w_j):
    clusters = createEmptyListOfLists(k)
    for i in range(0,len(points)):
        clusterIndex = cluster_assign(points[i], w_j[i])
        clusters[clusterIndex].append(points[i])
    return clusters

def main(file):
    points = loadData(file)
    points = np.array(points)
    k=3
    Log_list=[]
    min_iter = 1000
    best_sol = 0
    clusters_list=[]
    total_mean_list = []
    total_cov_list =[]
    for it in range(0,5):
        previous_mean=np.zeros((k,points.shape[1]))
        p_z, meanList, covList = initialize(k,points)
        iter=0
        liklihood = []
        w_j=[]
        while ((previous_mean!=meanList).any()):
            iter = iter+1
            #print iter
            previous_mean=meanList
            w_j_combined = E_Step(p_z, meanList, covList, k, points)
            p_z, meanList, covList, total_log_li = M_Step(w_j_combined, points, k)
            w_j=w_j_combined
            liklihood.append([iter,total_log_li])
        if(iter< min_iter):
            min_iter = iter
            best_sol = it

        clusters = createClusters(points,k,w_j)
        liklihood=np.array(liklihood)
        Log_list.append(liklihood)
        clusters_list.append(clusters)
        total_mean_list.append(meanList)
        total_cov_list.append(covList)
    print "Best iteration" , best_sol+1
    for i in range(0,len(Log_list)):
        plot(Log_list[i][:, 0], Log_list[i][:, 1])
    show()
    clusters=clusters_list[best_sol]
    print"covariance",total_cov_list[best_sol]
    print"mean",total_mean_list[best_sol]
    for i in range(0, k):
        d = clusters[i]
        d = np.array(d)
        plot(d[:, 0], d[:, 1], '+')
    show()

