'''
Created on Nov 12, 2016

@author: chira
'''

import numpy
import pandas
from sklearn.linear_model import LogisticRegression


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


training_set = pandas.read_csv("pca_train.txt", sep=',', header=None)

print len(training_set)
print len(training_set.columns)

training_set['upvotes'] = (user_ques_data['upvotes'] - numpy.mean(user_ques_data['upvotes']))/ numpy.std(user_ques_data['upvotes'], ddof = 1)
training_set['answers'] = (user_ques_data['answers'] - numpy.mean(user_ques_data['answers']))/ numpy.std(user_ques_data['answers'], ddof = 1)
training_set['top_quality_ans'] = (user_ques_data['top_quality_ans'] - numpy.mean(user_ques_data['top_quality_ans'])) / numpy.std(user_ques_data['top_quality_ans'], ddof = 1)
training_set['intercept'] = 1.0

labels = user_ques_data['label']
print 'train read done'
train_set = numpy.array([x[:len(x)] for x in training_set.values])
Y = numpy.array(labels)
# y_tr = to_categorical(Y)
ou = open("coef.txt", 'w')

print Y
print type(train_set)
print train_set.shape
print Y.shape
skmodel = LogisticRegression(fit_intercept=True, C=1e8, n_jobs=-1, verbose=1, solver='sag')
skmodel.fit(train_set, Y)
print skmodel.coef_
for y in skmodel.coef_:
    ou.write(str(y).encode('utf8'))
    ou.write('\n')
print skmodel.classes_

# logit = sm.Logit(labels, training_set)
# result= logit.fit(maxiter=4)

print 'Model Fitted'

# print result.summary()

test_data = pandas.read_csv('validate_nolabel.txt', sep=',', names=["qid", "uid", "label"])

df_intermediate_test = pandas.merge(test_data, user_data, how='inner', on=['uid'], sort=False)
df_test = pandas.merge(df_intermediate_test, ques_data, how='inner', on=['qid'], sort=False)

testing_set = pandas.read_csv("pca_test.txt", sep=',', header=None)

print len(testing_set)
print len(testing_set.columns)

qid = df_test['qid']
uid = df_test['uid']

testing_set['upvotes'] = (df_test['upvotes'] - numpy.mean(df_test['upvotes'])) / numpy.std(df_test['upvotes'], ddof = 1)
testing_set['answers'] = (df_test['answers'] - numpy.mean(df_test['answers'])) / numpy.std(df_test['answers'], ddof = 1)
testing_set['top_quality_ans'] = (df_test['top_quality_ans'] - numpy.mean(df_test['top_quality_ans'])) / numpy.std(df_test['top_quality_ans'], ddof = 1 )
testing_set['intercept'] = 1.0



print type(testing_set)

test_set = numpy.array([x[:len(x)] for x in testing_set.values])
print type(test_set)
print test_set.shape
# df_test['upvotes'] = (df_test['upvotes'] - df_test['upvotes'].mean()) / (df_test['upvotes'].max() - df_test['upvotes'].min())
# df_test['answers'] = (df_test['answers'] - df_test['answers'].mean()) / (df_test['answers'].max() - df_test['answers'].min())
# df_test['top_quality_ans'] = (df_test['top_quality_ans'] - df_test['top_quality_ans'].mean()) / (df_test['top_quality_ans'].max() - df_test['top_quality_ans'].min())

# testing_set['labels'] = result.predict(testing_set)
# testing_set['labels'] = skmodel.predict(test_set)
prob = skmodel.predict_proba(test_set)

probs = prob[:, 1]
probs0 = prob[:, 0]

outfile = open('LogisticProbs.txt', 'w')
outfile.write('qid,uid,label\n'.encode('utf8'))
for i in range(len(prob)):
    outfile.write("%s,%s,%s\n" % (str(qid[i]).encode('utf8'), str(uid[i]).encode('utf8'), str(probs[i]).encode('utf8')))
