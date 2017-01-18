'''
Created on Nov 12, 2016

@author: chira
'''

import random

from keras.utils.np_utils import to_categorical
import numpy
import pandas
import hw_utils

def create_data_dict(data, col, uniq_tags, uniq_cid):
    obj_dict = {}

    for i in range(len(data)):
        obj = data[col][i]

        chars = str(data['cid'][i]).split('/')
        tags = str(data['tags'][i]).split('/')

        for c in chars:
            uniq_cid.add(c)

            if obj in obj_dict:
                obj_dict[obj][0][c] = True
            else:
                l = [{}, {}]
                l[0][c] = True
                obj_dict[obj] = l

        for t in tags:
            uniq_tags.add(t)

            if obj in obj_dict:
                obj_dict[obj][1][t] = True
            else:
                l = list[{}, {}]
                l[1][t] = True
                obj_dict[obj] = l

    return obj_dict


user_uniq_tags = set()
user_uniq_cid = set()
print 'User'
user_file_name = "user_info.txt"
user_data = {}
user_data = pandas.read_csv(user_file_name, sep='\t', header=None, names=["uid", "tags", "wid", "cid"])
user_data_dict = create_data_dict(user_data, 'uid', user_uniq_tags, user_uniq_cid)

user_uniq_tags = sorted(user_uniq_tags)
user_uniq_cid = sorted(user_uniq_cid)

print len(user_uniq_tags)
print len(user_uniq_cid)

ques_uniq_tags = set()
ques_uniq_cid = set()
print 'Ques'
ques_file_name = "question_info.txt"
ques_data = {}
ques_data = pandas.read_csv(ques_file_name, sep='\t', header=None,
                            names=["qid", "tags", "wid", "cid", "upvotes", "answers", "top_quality_ans"])
ques_data_dict = create_data_dict(ques_data, 'qid', ques_uniq_tags, ques_uniq_cid)

ques_uniq_tags = sorted(ques_uniq_tags)
ques_uniq_cid = sorted(ques_uniq_cid)

print len(ques_uniq_tags)
print len(ques_uniq_cid)

del ques_data['wid']
del user_data['wid']

del ques_data['cid']
del user_data['cid']

del ques_data['tags']
del user_data['tags']

train_data = pandas.read_csv("invited_info_train.txt", sep='\t', names=['qid', 'uid', 'label'])
train_data.drop_duplicates(subset=['qid', 'uid', 'label'], keep='last', inplace=True)

groups = train_data.groupby(by=['qid', 'uid'])
train_data = groups.apply(lambda train_data: train_data[train_data['label'] == train_data['label'].max()])

df_intermediate = pandas.merge(train_data, user_data, how='inner', on=['uid'], sort=False)
user_ques_data = pandas.merge(df_intermediate, ques_data, how='inner', on=['qid'], sort=False)

# cols.append('upvotes')
# cols.append('answers')
# cols.append('top_quality_answers')
# cols.append('label')


training_set = pandas.read_csv("pca_train.txt", sep=',')


upvotes = user_ques_data['upvotes']
answers = user_ques_data['answers']
tqa = user_ques_data['top_quality_ans']
labels = user_ques_data['label']

del labels[len(labels)-1]
del upvotes[len(upvotes)-1]
del answers[len(answers)-1]
del tqa[len(tqa)-1]

training_set['upvotes'] = upvotes
training_set['answers'] = answers
training_set['top_quality_ans'] = tqa
training_set['label'] = labels

print len(training_set)
print len(training_set.columns)

user_ques_data = None

print 'train read done'

X_tr = []
Y = []

X_tr = numpy.array([x[:len(x)-1] for x in training_set.values])
Y = numpy.array([x[len(x)-1] for x in training_set.values])

print 'train format done'

y_tr = to_categorical(Y)

print 'train categorical done'

test_data = pandas.read_csv('validate_nolabel.txt',  sep = ',', names = ["qid","uid","label"])

df_intermediate_test = pandas.merge(test_data, user_data, how='inner', on=['uid'], sort= False)
df_test = pandas.merge(df_intermediate_test, ques_data, how='inner', on=['qid'], sort= False)

testing_set = pandas.read_csv("pca_test.txt", sep=',')


qid = df_test['qid']
uid = df_test['uid']


upvotes = df_test['upvotes']
answers = df_test['answers']
tqa = df_test['top_quality_ans']

del upvotes[len(upvotes)-1]
del answers[len(answers)-1]
del tqa[len(tqa)-1]

testing_set['upvotes'] = upvotes
testing_set['answers'] = answers
testing_set['top_quality_ans'] = tqa

print len(testing_set)
print len(testing_set.columns)


df_test = None
# print testing_set

# print testing_set['qid']



# for col in testing_set.columns:
#     if numpy.count_nonzero(testing_set[col]) == 0:
#         del testing_set[col]


print ' test read done'

X_te = numpy.array([x[:len(x)] for x in testing_set.values])

#X_tr, X_te = hw_utils.normalize(numpy.array(X_tr), numpy.array(X_te))

print len(X_tr)
print len(X_te)

print len(X_tr[0])
print len(X_te[0])

training_set = None
testing_set = None

print len(y_tr)

random.shuffle(X_tr)

arch = [[len(X_te[0]),500, 2]]
p_ndarray = hw_utils.testmodels(numpy.array(X_tr), numpy.array(y_tr), numpy.array(X_te), numpy.array([]), arch, actfn='relu', last_act='softmax', reg_coeffs=[0.00],
                num_epoch=5, batch_size=1000, sgd_lr=0.0001, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=1)


probs = p_ndarray[:,1]
# probs0 = p_ndarray[:,0]

outfile = open('New2500hidden10epochs0001lr0reg.txt','w')

outfile.write('qid,uid,label\n'.encode('utf8'))
for i in range(len(probs)):
    outfile.write("%s,%s,%s\n"%(str(qid[i]).encode('utf8'),str(uid[i]).encode('utf8'),str(probs[i]).encode('utf8')))

outfile.close()