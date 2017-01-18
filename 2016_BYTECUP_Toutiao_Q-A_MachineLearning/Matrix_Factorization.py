'''
Created on Nov 19, 2016

@author: chira
'''
import numpy
import pandas
import sklearn.decomposition as dcom

user_uniq_tags = set()
user_uniq_cid = set()

print 'User'
user_file_name = "user_info.txt"

user_data = {}
user_data = pandas.read_csv(user_file_name, sep = '\t',header=None, names = ["uid","tags","wid","cid"])

user_dict = {}

for i in range(len(user_data)):
    user_dict[user_data['uid'][i]] = i


print 'Ques'

ques_file_name = "question_info.txt"
ques_data = {}
ques_data = pandas.read_csv(ques_file_name, sep = '\t',header=None, names = ["qid","tags","wid","cid","upvotes","answers","top_quality_ans"])

ques_dict = {}

for i in range(len(ques_data)):
    ques_dict[ques_data['qid'][i]] = i

print 'Reading Training Data'

train_data = pandas.read_csv("invited_info_train.txt", sep='\t',names=['qid', 'uid','label'])
train_data.drop_duplicates(subset = ['qid', 'uid','label'], keep = 'last', inplace = True)
  
groups = train_data.groupby(by = ['qid', 'uid'])
train_data = groups.apply(lambda train_data : train_data[train_data['label'] == train_data['label'].max()])
  
df_intermediate = pandas.merge(train_data, user_data, how='inner', on=['uid'], sort= False)
user_ques_data = pandas.merge(df_intermediate, ques_data, how='inner', on=['qid'], sort= False)

user_ques_dict = {}
user_ques_list = []

print 'Creating Dictionary'

for i in range(len(user_ques_data)):
    
    u = user_ques_data['uid'][i]
    q = user_ques_data['qid'][i]
    
    if u in user_ques_dict:
        user_ques_dict[u].append(q)
    else:
        l = [q]
        user_ques_dict[u] = l 


print 'Creating Matrix'

for i in range(len(user_data)):
    usl = []
    u = user_data['uid'][i]
    for j in range(len(ques_data)):
        q = ques_data['qid'][j]
        
        if u in user_ques_dict:
            if q in user_ques_dict[u]:
                usl.append(1)
            else:
                usl.append(0)
        else:
            usl.append(0)
    user_ques_list.append(numpy.array(usl))
    

print user_ques_dict[0]

'''

# print user_ques_dict

print 'Creating NMF'

nmf = dcom.NMF(tol = 0.001, init='nndsvd')

print 'Creating W'


W = nmf.fit_transform(numpy.array(user_ques_list))

print 'Creating H'

H = nmf.components_

print 'Creating Dot'

nR = numpy.dot(W, H)

print 'Writing File'

print nR[0]

f = open('nmf.txt','w')

for r in range(len(nR)):
    
    f.write(','.join(map(str, r)))
    f.write('\n')
#    
f.close()
    
'''



        
        

