

import random
import matplotlib.pyplot as plt

import pandas
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing.data import StandardScaler

import numpy as np


# from keras.utils.np_utils import to_categorical
cols = ['qid', 'uid']

for i in range(6902):
    cols.append(str(i))

cols.append('upvotes')
cols.append('answers')
cols.append('top_quality_answers')
cols.append('label')


training_set = pandas.read_csv("training.txt", sep=',',names=cols[2:])

del training_set['answers']
del training_set['upvotes']
del training_set['top_quality_answers']
del training_set['label']

print 'train read done'

pca = IncrementalPCA(n_components=1000, batch_size=30000).fit(training_set)

print 'Fitted'

data = pca.transform(training_set)
# training_set = training_set.sample(frac = 0.02)

# print 'train read done'





# X_std = StandardScaler().fit_transform(training_set)
# 
# #  
# print len(X_std[0])

'''
#  
cov_mat = np.cov(X_std.T)
#  
print len(cov_mat)
eigval, eigvec = np.linalg.eig(cov_mat)
  
print len(eigvec)
  
eigpairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]
  
print len(eigval)
print len(eigvec[0])

eigpairs.sort(key = lambda x : x[0], reverse = True)
#  
# srt = np.argsort(eigval)[::-1]
# eigvec = np.matrix(eigvec[:,srt])
# eigval = eigval[srt]
# 
# print eigval
 
 
# pca = PCA(n_components = 2)
# pca.fit_transform(user_ques_arr)
'''
# print pca.explained_variance_ratio_ 


vars = pca.explained_variance_ratio_
print len(vars)
comps = [i for i in range(len(vars))]
 
plt.bar(comps, vars)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

df = pandas.DataFrame(data, index= None, columns=None)

df.to_csv('pca.txt', sep=',', index=False, header=False)

print 'Finished'

# weight = np.hstack((eigpairs[0][1].reshape(4170,1), eigpairs[1][1].reshape(4170,1)))
# print weight