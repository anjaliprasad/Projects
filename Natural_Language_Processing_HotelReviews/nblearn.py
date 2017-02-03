import collections
import re
import json
import sys

filename1 = sys.argv[1]
filename2 = sys.argv[2]

def loadData(fn):
    with open(fn) as f:
        mylist = f.read().splitlines()
    return mylist

def tokenize1(set):
    punctuation = re.compile(r'[-.?!,":;()|0-9]')
    list_of_tokens = []
    stopWords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                 "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                 "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
                 "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
                 "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                 "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
                 "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
                 "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out",
                 "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
                 "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
                 "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
                 "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
                 "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
                 "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
                 "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "Subject:"]

    tokens = set.lower().split()
    tok=[]
    #tokens = set.split()
    for i in range(0, len(tokens)):
        if (tokens[i] not in stopWords):
            tok.append(punctuation.sub("", tokens[i]))
    list_of_tokens.append(tok)
    #print tokens
    list_of_tokens.append(tokens)
    return list_of_tokens

def getVocabulary(vset):
    vocabulary = []
    for s in range(0,len(vset)) :
        for k in vset[s]:
            vocabulary.append(k)
    vocabulary1 = set(vocabulary)
    vocabulary2=list(vocabulary1)
    return vocabulary2

def getVocabulary1(vset):
    #print "vset",vset[1]
    vocabulary = []
    for k in range(0,len(vset)):
        vocabulary.append(vset[k])

    #vocabulary1 = set(vocabulary)
    #vocabulary2=list(vocabulary1)
    return vocabulary

def column_extract(training,column_to_extract):
    column = []
    for i in range(0,len(training)):
        column.append(training[i][column_to_extract])
    return column

def NB():
    trainingSet = loadData(filename1)
#-----new
    training_set1 = []
    training_labels1 = []
    trainingLabel1 = loadData(filename2)
    for i in range(0, len(trainingSet)):
        training_labels1.append(trainingLabel1[i].split(' ', 2))
        training_set1.append(trainingSet[i].split(' ', 1))

    Token_list1 = []
    for i in range(0, len(training_set1)):
        Tokens1 = tokenize1(str(training_set1[i][1]))
        Token_list1.append(Tokens1)
    for i in range(0, len(training_set1)):
        training_labels1[i][0] = Token_list1[i][0]

    for i in range(0,len(training_labels1)):
        training_labels1[i][0],training_labels1[i][2]=training_labels1[i][2],training_labels1[i][0]
        training_labels1[i][0], training_labels1[i][1] = training_labels1[i][1], training_labels1[i][0]

    #Creating datastructures
    #Create dictionary of words with key as word and value as index of the array

    #----new
    vocabulary1=[]
    for i in range(0, len(training_labels1)):
        Tokens1 = getVocabulary1(training_labels1[i][2])
        for tok in Tokens1:
            vocabulary1.append(tok)
    #print "vocabus",len(vocabulary1)
    vocabulary2 = set(vocabulary1)
    vocabulary1 = list(vocabulary2)
    #print "vocabulary1",len(vocabulary1),vocabulary1[0],vocabulary1[2],vocabulary1[3]
    #-----

    codes = dict();
    codes["positive"] = 0
    codes["negative"] = 1
    codes["deceptive"] = 2
    codes["truthful"] = 3

    index_dict = dict();
    count = 0
    for i in vocabulary1:
        index_dict[i] = count
        count = count + 1
    total_unique_words = len(vocabulary1)
    word_count_array1=[ [0]*total_unique_words for _ in xrange(len(codes)) ]
    for i in range(0,len(training_labels1)):
        for j in training_labels1[i][2]:
            word_count_array1[codes[training_labels1[i][0]]][index_dict[j]] = word_count_array1[codes[training_labels1[i][0]]][index_dict[j]] + 1.0
            word_count_array1[codes[training_labels1[i][1]]][index_dict[j]] = word_count_array1[codes[training_labels1[i][1]]][index_dict[j]] + 1.0

    total_words_each_class = [0] * len(codes)

    for i in range(0, len(codes)):
        for j in range(0,total_unique_words):
            total_words_each_class[i] += word_count_array1[i][j]

    #------------------------------------------------
    word_prob_array1 = [ [0]*total_unique_words for _ in xrange(len(codes)) ]
    for i in range(0,len(codes)):
        for j in range(0,total_unique_words):
            word_prob_array1[i][j]=(word_count_array1[i][j] + 1)/(total_words_each_class[i] + total_unique_words)


    #------------------------------------------------
    prior_prob=[0] * len(codes)
    for i in range(0, len(training_labels1[0])-1):
        class_column = column_extract(training_labels1,i)
        unique_count = collections.Counter(class_column)
        unique_words = set(class_column)
        unique_words = list(unique_words)
        for j in range(0,len(unique_words)):
            prior_prob[codes[unique_words[j]]]=unique_count[unique_words[j]]/float(len(training_labels1))
    #----------------------------------------------------------------------------------------------------
    #with open("model.txt", "w") as text_file:
    #    text_file.write("Prior Probabilities \n")
    #    for i in codes:
    #        text_file.write(i + "\t\t")
    #    text_file.write("\n")
    #    for i in range(0,len(codes)):
    #        text_file.write(str(prior_prob[i]) + "\t")
    #    text_file.write("\n\n\t")
    #    for i in index_dict:
    #        text_file.write(i + "\t\t=" + str(word_prob_array1[0][index_dict[i]] ))
    print_data = dict();
    print_data["prior_prob"] = prior_prob
    print_data["word_prob_array1"] = word_prob_array1
    print_data["index_dict"] = index_dict
    print_data["codes"] = codes
    print_data["vocabulary1"]=vocabulary1

    with open('nbmodel.txt', 'w') as outfile:
        #json.dump(prior_prob, outfile)
        json.dump(print_data, outfile)
        #outfile.write("\n\n")
        #json.dump(print_data["word_prob_array1"], outfile)
    #-------------------------------------------------------------------------------------------------------
    #nbclassify.classify()


NB()