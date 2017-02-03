import math
import json
import re

import sys

filename1 = sys.argv[1]

def loadData(fn):
    with open(fn) as f:
        mylist = f.read().splitlines()
    return mylist

def tokenize(set):
    punctuation = re.compile(r'[-.?!,":;()|0-9]')

    stopWords =["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
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

    list_of_tokens = []
    #for s in set:
    #tokens = set.split()
    tokens = set.lower().split()
    tok=[]
    for i in range(0,len(tokens)):
        if(tokens[i] not in stopWords):
            tok.append(punctuation.sub("", tokens[i])  )
    list_of_tokens.append(tok)
    return list_of_tokens

def classify():
    data_dictionary = []
    with open("nbmodel.txt") as outputfile:
        data_dictionary = json.load(outputfile)

    prior_prob = data_dictionary["prior_prob"]
    word_prob_array = data_dictionary["word_prob_array1"]
    index_dict=data_dictionary["index_dict"]
    codes=data_dictionary["codes"]
    vocabulary=data_dictionary["vocabulary1"]
    testSet = loadData(filename1)
    #testLabel = loadData("train-label2.txt")
    testing_set = []
    testing_labels = []
    for i in range(0,len(testSet)):
        #testing_labels.append(testLabel[i].split(' ',2))
        testing_set.append(testSet[i].split(' ', 1))

    Token_list=[]
    for i in range(0,len(testing_set)):
        Tokens = tokenize(str(testing_set[i][1]))
        Token_list.append(Tokens)
    for i in range(0,len(testing_set)):
        testing_set[i][1]=Token_list[i][0]
    #-------------------------------------------------------
    #for column 1
    testing_labels_final = []

    for i in range(0,len(testing_set)):
        prob = []
        for pri in range(0, len(prior_prob)):
            prob.append(math.log(prior_prob[pri]))
        instance_label_c1 = []

        for j in range(0,len(testing_set[i][1])):
            for k in range(0,len(codes)):
                if(testing_set[i][1][j] in vocabulary):
                    prob[k] = prob[k] + math.log(word_prob_array[k][index_dict[testing_set[i][1][j]]])
                #if(i==3):
                    #print "ggjgh", word_prob_array[k][index_dict[testing_set[i][1][j]]]
        instance_label_c1.append(testing_set[i][0])
        if (prob[2] > prob[3]):
            instance_label_c1.append("deceptive")
        elif (prob[2] <= prob[3]):
            instance_label_c1.append("truthful")
        if (prob[0] > prob[1]):
            instance_label_c1.append("positive")
        elif (prob[0] <= prob[1]):
            instance_label_c1.append("negative")
        testing_labels_final.append(instance_label_c1)

    with open('nboutput.txt', 'w') as outfile:
        #json.dump(prior_prob, outfile)
        for i in range (0,len(testing_labels_final)):
            if i==(len(testing_labels_final) - 1):
                outfile.write(testing_labels_final[i][0] + " " + testing_labels_final[i][1] + " " + testing_labels_final[i][2])
            else:
                outfile.write(testing_labels_final[i][0] + " " + testing_labels_final[i][1] + " " + testing_labels_final[i][2] + "\n")


classify()


