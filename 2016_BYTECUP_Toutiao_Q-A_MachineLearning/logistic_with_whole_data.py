'''
Created on Nov 18, 2016

@author: chira
'''
# from keras.utils.np_utils import to_categorical
import numpy
import pandas
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier


cols = ['qid', 'uid']

for i in range(6902):
    cols.append(str(i))

cols.append('upvotes')
cols.append('answers')
cols.append('top_quality_ans')
cols.append('label')


training_set = pandas.read_csv("training.txt", sep=',',names=cols[2:])

training_set['upvotes'] = (training_set['upvotes'] - numpy.mean(training_set['upvotes']))/ numpy.std(training_set['upvotes'], ddof = 1)
training_set['answers'] = (training_set['answers'] - numpy.mean(training_set['answers']))/ numpy.std(training_set['answers'], ddof = 1)
training_set['top_quality_ans'] = (training_set['top_quality_ans'] - numpy.mean(training_set['top_quality_ans'])) / numpy.std(training_set['top_quality_ans'], ddof = 1)
labels = training_set['label']
del training_set['label']

training_set['intercept'] = 1.0

train_set = numpy.array([x[:len(x)] for x in training_set.values])
Y = numpy.array(labels)

print Y
print type(train_set)
print train_set.shape
print Y.shape

sgd = LogisticRegression(penalty='l1',fit_intercept=True, C=1e4, n_jobs=-1, verbose=1, solver='sag', max_iter=200)#, alpha=0.0000001)# learning_rate='constant', eta0=0.00001)
sgd.fit(train_set, Y)
print sgd.coef_
print sgd.intercept_


testing_set = pandas.read_csv("testing.txt", sep=',',names=cols[len(cols) - 1])

qid = testing_set['qid']
uid = testing_set['uid']

del testing_set['uid']
del testing_set['qid']

testing_set['upvotes'] = (testing_set['upvotes'] - numpy.mean(testing_set['upvotes'])) / numpy.std(testing_set['upvotes'], ddof = 1)
testing_set['answers'] = (testing_set['answers'] - numpy.mean(testing_set['answers'])) / numpy.std(testing_set['answers'], ddof = 1)
testing_set['top_quality_ans'] = (testing_set['top_quality_ans'] - numpy.mean(testing_set['top_quality_ans'])) / numpy.std(testing_set['top_quality_ans'], ddof = 1 )

testing_set['intercept'] = 1.0

print type(testing_set)

test_set = numpy.array([x[:len(x)] for x in testing_set.values])
print type(test_set)
print test_set.shape


prob = sgd.predict_proba(test_set)
probs = prob[:,1]
probs0 = prob[:,0]

outfile = open('SGD_WholeData_100Iter.txt','w')
outfile.write('qid,uid,label\n'.encode('utf8'))
for i in range(len(prob)):
    outfile.write("%s,%s,%s\n"%(str(qid[i]).encode('utf8'),str(uid[i]).encode('utf8'),str(probs[i]).encode('utf8')))

outfile.close()


