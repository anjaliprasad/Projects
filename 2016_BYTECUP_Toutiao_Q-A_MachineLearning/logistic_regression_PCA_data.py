'''
Created on Nov 3, 2016

@author: chira
'''

import pandas

import statsmodels.api as sm


#df_norm = (df - df.mean()) / (df.max() - df.min())
remove = ['upvotes', 'top_quality_answers','answers','10','2000']
cols = ['qid', 'uid']

for i in range(4166):
    cols.append(str(i))

cols.append('upvotes')
cols.append('answers')
cols.append('top_quality_answers')
cols.append('labels')

# df_train = pandas.read_csv("training.txt", sep=',', names = train_cols) 

training_set = pandas.read_csv("training.txt", sep=',',names=cols)

print training_set['labels']
exit()

del training_set['qid'] 
del training_set['uid']
del training_set['upvotes']
del training_set['top_quality_answers']
del training_set['3000']

training_set['intercept'] = 1.0

print 'train read done'

label_col = training_set['labels']
del training_set['labels']


# data_label_col = 'labels'
# data_train_cols = train_cols[2:len(train_cols)-1]
# data_train_cols.append('intercept')

print 'Train Read Done'


# uq = {}
# f = open('corr.txt', 'w')
# for i in range(len(df_train['qid'])):
#     qid = df_train['qid'][i]
#     uid = df_train['uid'][i]
#     
#     if uid in uq:
#         uq[uid][1]+=df_train['labels'][i]
#         uq[uid][0]+=1
#     else:
#         l =[1,df_train['labels'][i]]
#         uq[uid]=l
# 
# f.write('uid, q_count, ans_count\n')
# for i in uq:
#     f.write('%s - %s - %s\n'%(i, uq[i][0], uq[i][1]))
# f.close()
        



# f = open('corr.txt', 'w')
# cor =  df_train.corr()
# # f.write(str(cor['u0']))
# 
# cols = cor.columns

# 
# 
# print cols
# print len(train_cols)
# 

# cls = set()
# j=0
# for col in cols:
#     for i in range(len(cor[col])):
#         if(cor[col][i] > 0.7 and cor[col][i] < 1):
# #             if i >= j:
# #             cls.add(cols[i])
#                 cls.add(cols[i])
#                 if cols[i] not in cls and col not in cls:
#                     cls.add(col)
#                 elif cols[i] in cls and col in cls:
#                     cls.remove(col)
#     j+=1
#   
# remove.extend(list(cls))
# 
# for i in range(len(cor)):
#     print cor[i]


#  
# for r in set(remove):
#     data_train_cols.remove(r)
# 
# # for j in range(20):
# #     data_train_cols.remove('q'+str(j))
#  

# clf = RandomForestClassifier(n_jobs=2)
# clf.fit(df_train[data_train_cols], df_train[data_label_col])

logit = sm.Logit(label_col, training_set)
result= logit.fit(maxiter=4)

print 'Model Fitted'

print result.summary()
   
testing_set = pandas.read_csv("testing.txt", sep=',',names=cols[:len(cols)-1])

print 'test Read Done'

qid = testing_set['qid']
uid = testing_set['uid']


del testing_set['qid'] 
del testing_set['uid']
del testing_set['upvotes']
del testing_set['top_quality_answers']
del testing_set['10']
del testing_set['3000']

testing_set['intercept'] = 1.0

# df_test['upvotes'] = (df_test['upvotes'] - df_test['upvotes'].mean()) / (df_test['upvotes'].max() - df_test['upvotes'].min())
# df_test['answers'] = (df_test['answers'] - df_test['answers'].mean()) / (df_test['answers'].max() - df_test['answers'].min())
# df_test['top_quality_ans'] = (df_test['top_quality_ans'] - df_test['top_quality_ans'].mean()) / (df_test['top_quality_ans'].max() - df_test['top_quality_ans'].min()) 
 
 
      
testing_set['labels'] = result.predict(testing_set)

print 'Predicted'
 
 
out = open('tempp.csv', 'w')
out.write('qid,uid,label\n')
      
for i in range(len(testing_set)):
    out.write('%s,%s,%s\n'%(qid[i],uid[i],testing_set['labels'][i]))
out.close()