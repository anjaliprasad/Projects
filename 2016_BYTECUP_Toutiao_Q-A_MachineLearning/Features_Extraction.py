'''
Created on Nov 13, 2016

@author: chira
'''

import pandas
from sklearn.decomposition import IncrementalPCA


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
                l = [{},{}]
                l[0][c] = True
                obj_dict[obj] = l
        
        for t in tags:
            uniq_tags.add(t)
            
            if obj in obj_dict:
                obj_dict[obj][1][t] = True
            else:
                l = list[{},{}]
                l[1][t] = True
                obj_dict[obj] = l
    
    return obj_dict
    

user_uniq_tags = set()
user_uniq_cid = set()

print 'User'
user_file_name = "user_info.txt"
user_data = {}
user_data = pandas.read_csv(user_file_name, sep = '\t',header=None, names = ["uid","tags","wid","cid"])
user_data_dict = create_data_dict(user_data, 'uid', user_uniq_tags,user_uniq_cid)

user_uniq_tags = sorted(user_uniq_tags)
user_uniq_cid = sorted(user_uniq_cid)


print len(user_uniq_tags)
print len(user_uniq_cid)


ques_uniq_tags = set()
ques_uniq_cid = set()

print 'Ques'

ques_file_name = "question_info.txt"
ques_data = {}
ques_data = pandas.read_csv(ques_file_name, sep = '\t',header=None, names = ["qid","tags","wid","cid","upvotes","answers","top_quality_ans"])
ques_data_dict = create_data_dict(ques_data, 'qid',ques_uniq_tags, ques_uniq_cid)

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

# print ques_data_dict['e153659c6c654cd12122232fca89f4bc'][0]

print 'Reading Training Data'

train_data = pandas.read_csv("invited_info_train.txt", sep='\t',names=['qid', 'uid','label'])
train_data.drop_duplicates(subset = ['qid', 'uid','label'], keep = 'last', inplace = True)
  
groups = train_data.groupby(by = ['qid', 'uid'])
train_data = groups.apply(lambda train_data : train_data[train_data['label'] == train_data['label'].max()])
  
df_intermediate = pandas.merge(train_data, user_data, how='inner', on=['uid'], sort= False)
user_ques_data = pandas.merge(df_intermediate, ques_data, how='inner', on=['qid'], sort= False)


print len(user_ques_data)

print 'Preparing Training Data'

X_tr = []
# y_tr = []

# user_ques_arr = []
# f = open('training_withZero.txt', 'w')

for i in range(len(user_ques_data)):
    
    inner = []
    y  = []
    
    user = user_ques_data['uid'][i]
    ques = user_ques_data['qid'][i]
    
    for t in user_uniq_tags:
        val = -1
        
        if t in user_data_dict[user][1]:
            val = 1
#         
        inner.append(val)
    
    for t in ques_uniq_tags:
        val = -1
        
        if t in ques_data_dict[ques][1]:
            val =1
        
        inner.append(val)
        
    for c in user_uniq_cid:
        
        val = -1
        
        if c in user_data_dict[user][0]:
            val =1
        
        inner.append(val)
        
    for c in ques_uniq_cid:
        
        val = -1
        
        if c in ques_data_dict[ques][0]:
            val = 1
        
        inner.append(val)        
    
#     y.append(user_ques_data['upvotes'][i])
#     y.append(user_ques_data['answers'][i])
#     y.append(user_ques_data['top_quality_ans'][i])
#     y.append(user_ques_data['label'][i])    
#     inner.append(user_ques_data['upvotes'][i])
#     inner.append(user_ques_data['answers'][i])
#     inner.append(user_ques_data['top_quality_ans'][i])
#     inner.append(user_ques_data['label'][i])
#     y_tr.append(y)
    X_tr.append(inner)
    
#     print len(inner)
    
#     f.write(','.join(map(str, inner)))
#     f.write('\n')
#   
# f.close()

print 'Generating PCA'

pca = IncrementalPCA(n_components=5000, batch_size=40000)

print 'Generated'

print 'Fitting Data'

pca.fit(X_tr)

print 'Fitted'

print 'Transforming Training data'
data = pca.transform(X_tr)

print 'writing Training Data'
 
f = open('training_withPCA.txt', 'w')
 
for i in range(len(data)):
    
    inner = list(data[i])
#     inner.extend(y_tr[i])
    f.write(','.join(map(str, inner)))
    f.write('\n')
 
f.close()

print 'Finished Processing Training Data'

X_tr = None
y_tr = None
# 
user_ques_data = None
# 
data = None

print 'Reading Testing Data'

test_data = pandas.read_csv('validate_nolabel.txt',  sep = ',', names = ["qid","uid","label"])

df_intermediate_test = pandas.merge(test_data, user_data, how='inner', on=['uid'], sort= False)
df_test = pandas.merge(df_intermediate_test, ques_data, how='inner', on=['qid'], sort= False)

X_te = []
y_te = []

print 'Preparing Testing Data'

# f = open('testing_withZero.txt', 'w')

for i in range(len(df_test)):
    inner = []
    y = []
    
    user = df_test['uid'][i]
    ques = df_test['qid'][i]
    
    for t in user_uniq_tags:
        val = -1
        
        if t in user_data_dict[user][1]:
            val = 1
        
        inner.append(val)
    
    for t in ques_uniq_tags:
        val = -1
        
        if t in ques_data_dict[ques][1]:
            val = 1
        
        inner.append(val)
        
    for c in user_uniq_cid:
        
        val = -1
        
        if c in user_data_dict[user][0]:
            val = 1
        
        inner.append(val)
        
    for c in ques_uniq_cid:
        
        val = -1
        
        if c in ques_data_dict[ques][0]:
            val = 1
        
        inner.append(val)
    
    X_te.append(inner) 


#     y.append(df_test['upvotes'][i])
#     y.append(df_test['answers'][i])
#     y.append(df_test['top_quality_ans'][i])
#     
#     y_te.append(y)
#     
#     print len(inner)
     
#     f.write(','.join(map(str, inner)))
#     f.write('\n')
#    
# f.close()


print 'Transforming Testing data'

data = pca.transform(X_te)

print 'Writing Testing data'
 
f = open('testing_withPCA.txt', 'w')

for i in range(len(data)):
    
    inner = list(data[i])
#     inner.extend(y_te[i])
    f.write(','.join(map(str, inner)))
    f.write('\n')
 
f.close()


print 'Finished'