'''
Created on Nov 13, 2016

@author: chira
'''

import random

from keras.utils.np_utils import to_categorical
import numpy
import pandas

import hw_utils


def create_data_dict(data, col):
    
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
            
        
    

uniq_tags = set()
uniq_cid = set()
print 'User'
user_file_name = "user_info.txt"
user_data = {}
user_data = pandas.read_csv(user_file_name, sep = '\t',header=None, names = ["uid","tags","wid","cid"])

user_data_dict = create_data_dict(user_data, 'uid')



print 'Ques'
ques_file_name = "question_info.txt"
ques_data = {}
ques_data = pandas.read_csv(ques_file_name, sep = '\t',header=None, names = ["qid","tags","wid","cid","upvotes","answers","top_quality_ans"])

ques_data_dict = create_data_dict(ques_data, 'qid')


del ques_data['wid']
del user_data['wid']

uniq_tags = sorted(uniq_tags)
uniq_cid = sorted(uniq_cid)

del uniq_cid[3000]
del uniq_tags[10]

print len(uniq_cid)
print len(uniq_tags)


train_data = pandas.read_csv("invited_info_train.txt", sep='\t',names=['qid', 'uid','label'])
train_data.drop_duplicates(subset = ['qid', 'uid','label'], keep = 'last', inplace = True)
 
groups = train_data.groupby(by = ['qid', 'uid'])
train_data = groups.apply(lambda train_data : train_data[train_data['label'] == train_data['label'].max()])
 
df_intermediate = pandas.merge(train_data, user_data, how='inner', on=['uid'], sort= False)
user_ques_data = pandas.merge(df_intermediate, ques_data, how='inner', on=['qid'], sort= False)

X_tr = []
y_tr = []

# user_ques_arr = []
# f = open('training.txt', 'w')

for i in range(len(user_ques_data)):
    inner = []
    
    user = user_ques_data['uid'][i]
    ques = user_ques_data['qid'][i]
    
#     inner.append(ques)
#     inner.append(user)
    
    for t in uniq_tags:
        val = 0
        
        if t in user_data_dict[user][1]:
            val += 1
        
        if t in ques_data_dict[ques][1]:
            val += 1
            
        if val == 0:
            val = -1
        
        inner.append(val)
    
    for c in uniq_cid:
        
        val = 0
        
        if c in user_data_dict[user][0]:
            val += 1
        
        if c in ques_data_dict[ques][0]:
            val += 1
        
        if val == 0:
            val = -1
        
        inner.append(val)
    
    inner.append(user_ques_data['upvotes'][i])
    inner.append(user_ques_data['answers'][i])
    inner.append(user_ques_data['top_quality_ans'][i])
    
    X_tr.append(inner)
    
    if user_ques_data['label'][i] == 0:
        y_tr.append([1,-1])
    else:
        y_tr.append([-1,1])
    
#     f.write(','.join(map(str, inner)))
#     f.write('\n')
# 
# f.close()

print 'Read Done'
print len(X_tr)
print len(X_tr[0])

qid = []
uid = []

test_data = pandas.read_csv('validate_nolabel.txt',  sep = ',', names = ["qid","uid","label"])

df_intermediate_test = pandas.merge(test_data, user_data, how='inner', on=['uid'], sort= False)
df_test = pandas.merge(df_intermediate_test, ques_data, how='inner', on=['qid'], sort= False)

X_te = []

# f = open('testing.txt', 'w')

for i in range(len(df_test)):
    inner = []
    
    user = df_test['uid'][i]
    ques = df_test['qid'][i]
    
    qid.append(ques)
    uid.append(user)
    
    for t in uniq_tags:
        val = 0
        
        if t in user_data_dict[user][1]:
            val += 1
        
        if t in ques_data_dict[ques][1]:
            val += 1
        
        if val == 0:
            val = -1
        
        inner.append(val)
    
    for c in uniq_cid:
        
        val = 0
        
        if c in user_data_dict[user][0]:
            val += 1
        
        if c in ques_data_dict[ques][0]:
            val += 1
        
        if val == 0:
            val = -1    
        
        inner.append(val)
    
    inner.append(df_test['upvotes'][i])
    inner.append(df_test['answers'][i])
    inner.append(df_test['top_quality_ans'][i])
    
    X_te.append(inner)
    


print 'Test Done'
print len(X_te)
print len(X_te[0])

# X_tr, X_te = hw_utils.normalize(numpy.array(X_tr), numpy.array(X_te))

random.shuffle(X_tr)

arch = [[len(X_te[0]),200,2]]
p_ndarray = hw_utils.testmodels(numpy.array(X_tr), numpy.array(y_tr), numpy.array(X_te), numpy.array(y_tr), arch, actfn='relu', last_act='softmax', reg_coeffs=[0.0001], 
                num_epoch=50, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
                    sgd_Nesterov=False, EStop=True, verbose=0)

 
probs = p_ndarray[:,1]
 
outfile = open('final.txt','w')

outfile.write('qid,uid,label\n'.encode('utf8'))
for i in range(len(probs)):
    outfile.write("%s,%s,%s\n"%(str(qid[i]).encode('utf8'),str(uid[i]).encode('utf8'),str(probs[i]).encode('utf8')))
 
outfile.close()
    
#     f.write(','.join(map(str, inner)))
#     f.write('\n')
# 
# f.close()




# X_std = StandardScaler().fit_transform(user_ques_arr)
# 
# # print len(X_std[0])
# 
# cov_mat = np.cov(X_std.T)
# 
# print len(cov_mat)
# eigval, eigvec = np.linalg.eig(cov_mat)
# 
# print len(eigvec)

# eigpairs = [(eigval[i], eigvec[i])]

# print len(eigval)
# print len(eigvec[0])
# print eigvec[0]

# srt = np.argsort(eigval)[::-1]
# eigvec = np.matrix(eigvec[:,srt])
# eigval = eigval[srt]
# 
# print eigval


# pca = PCA(n_components = 2)
# pca.fit_transform(user_ques_arr)
# print pca.explained_variance_ratio_ 
# pca = PCA().fit(X_std)
# vars = pca.explained_variance_ratio_
# print len(vars)
# comps = [i for i in range(len(vars))]
# 
# plt.bar(comps, vars)
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()



