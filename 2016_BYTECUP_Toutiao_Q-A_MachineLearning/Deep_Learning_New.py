
'''
Created on Nov 12, 2016

@author: chira
'''
'''
u tags - 4, ques tag, u wid 25, q wid 25, upvotes, answers, top_quality_answers, label
'''

import random

from keras.utils.np_utils import to_categorical
import numpy
import pandas

import hw_utils

cols = ['qid', 'uid']

for i in range(6902):
    cols.append(str(i))

cols.append('upvotes')
cols.append('answers')
cols.append('top_quality_answers')
cols.append('label')


training_set = pandas.read_csv("training_withZero.txt", sep=',',names=cols[2:])

print 'train read done'

X_tr = []
Y = []

X_tr = numpy.array([x[:len(x)-1] for x in training_set.values])
Y = numpy.array([x[len(x)-1] for x in training_set.values])

print 'train format done'

y_tr = to_categorical(Y)

print 'train categorical done'

testing_set = pandas.read_csv("testing_withZero.txt", sep=',',names=cols[:len(cols)-1])

print 'test Read Done'

qid = testing_set['qid']
uid = testing_set['uid']


del testing_set['qid'] 
del testing_set['uid']
# print testing_set

# print testing_set['qid']



# for col in testing_set.columns:
#     if numpy.count_nonzero(testing_set[col]) == 0:
#         del testing_set[col]
        

print ' test read done'

X_te = numpy.array([x[:len(x)] for x in testing_set.values])

# X_tr, X_te = hw_utils.normalize(numpy.array(X_tr), numpy.array(X_te))

print len(X_tr)
print len(X_te)

print len(X_tr[0])
print len(X_te[0])

training_set = None
testing_set = None

print len(y_tr)

random.shuffle(X_tr)

arch = [[len(X_te[0]),10000,2]]
p_ndarray = hw_utils.testmodels(numpy.array(X_tr), y_tr, numpy.array(X_te), numpy.array([]), arch, actfn='relu', last_act='softmax', reg_coeffs=[0.000001], 
                num_epoch=2, batch_size=1000, sgd_lr=0.0001, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=False, verbose=1)


probs = p_ndarray[:,1]
 
outfile = open('final1.txt','w')

outfile.write('qid,uid,label\n'.encode('utf8'))
for i in range(len(probs)):
    outfile.write("%s,%s,%s\n"%(str(qid[i]).encode('utf8'),str(uid[i]).encode('utf8'),str(probs[i]).encode('utf8')))
 
outfile.close()