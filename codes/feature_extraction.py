# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:07:20 2019

@author: daniel
"""
import numpy as np
import nltk
import csv
import editdistance

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def word_sim(token1, token2):
#    print(token1)
#    print(token2)
    """Convert all characters to lowercase from list of tokenized words"""
    try:
        syn1 = wn.synset(token1[0] + '.' + penn2morphy(token1[1])+'.01')
        syn2 = wn.synset(token2[0] + '.' + penn2morphy(token2[1])+'.01')
    except WordNetError:
        return 0
    
    simi_score = 0.0
    if syn1 and syn2: 
        simi_score = syn1.wup_similarity(syn2)                    
        if simi_score is None:
            tdist = editdistance.eval(token1[0], token2[0])
            kata1 = len(token1[0])
            kata2 = len(token2[0])
            simi_score = 1 - (tdist / max(kata1, kata2))
    else:
        tdist = editdistance.eval(token1[0], token2[0])
        kata1 = len(token1[0])
        kata2 = len(token2[0])
        simi_score = 1 - (tdist / max(kata1, kata2))

    if simi_score>0.8:
        #print(word1,' ', word2,' ', simi_score)
        return 1
    return 0

def find_filter(thelist, tup):
    for key,val in thelist:
#        print(key)
        if (key==tup):
            return key
#    return filter(lambda s: s[1] == tup or s[2] == tup, thelist) 
#    return [item for item in thelist if item[0] == tup]

#Requirements
default_stops = set(stopwords.words('english'))
new_stopwords = ['.', ',', '?', ':','-', '–', '/', 'the', ]
out_stopwords = {'our', 'you', 'we', 'they','would', 'should', 'but', 'will', 'how', 'all', 'who', 'ours',"wouldn't", "shouldn't", 'that', 'now', 'later''every',"tomorrow", "next", 'yesterday', 'it', 'she','he'}
add_stopwords = default_stops.union(new_stopwords)
#print(add_stopwords)
stops = set([word for word in add_stopwords if word not in out_stopwords])
#print(stops)
lemmatizer = WordNetLemmatizer()

import ast
bad_tags=[]
with open("dataset/DA/bad_tags_0p5.csv", "r") as f:
#    csvreader = csv.reader(f, delimiter=';', dialect='excel') 
    csvreader = csv.reader(f,delimiter=';', dialect='excel') 
    for row in csvreader:
        bad_row = []
        for item in row:
            bad_row.append(ast.literal_eval(item))
#            print(ast.literal_eval(item))
        bad_tags.append(bad_row)
#        print(bad_row)

bad_MD = bad_tags[0]
bad_NN = bad_tags[2]
bad_VB = bad_tags[4]
bad_RB = bad_tags[6]
bad_JJ = bad_tags[8]
bad_DT = bad_tags[10]
bad_punctuation = bad_tags[12]

filename = "dataset/DA/added/da-14-raw"
#f = open("dataset/DA/corpus_req.txt", "r")
#corpus = f.read()
#sentences = np.genfromtxt("dataset/DA/corpus_req.txt", delimiter ='\n', dtype=None)
sentences = np.genfromtxt(filename+ ".txt", delimiter ='\n', dtype=None)
results =""
for sentence in sentences:
    f_bad_MD = 0
    f_bad_NN = 0
    f_bad_VB = 0
    f_bad_punctuation = 0
    f_bad_RB = 0
    f_bad_JJ = 0
    f_bad_DT = 0
    try:
        words = nltk.tokenize.word_tokenize(sentence.lower().decode())
    except AttributeError:
        words = nltk.tokenize.word_tokenize(sentence.lower())

    tags = nltk.pos_tag(words)
    tags_remove_stopwords =[]
    for word,pos in tags:
        if word not in stops:
            tags_remove_stopwords.append((word,pos))
#    print('total setelah stopword: ' + str(len(tags_remove_stopwords)))
    
    del tags
    tags_lemmatized = []
    for w,t in tags_remove_stopwords:
        if (t in ('NN','NNS','NNP')):
            tags_lemmatized.append((lemmatizer.lemmatize(w,'n'),'NN')) 
        elif (t in ('VB','VBN','VBZ','VBG','VBP')):
            tags_lemmatized.append((lemmatizer.lemmatize(w,'v'),'VB')) 
        else:
            tags_lemmatized.append((lemmatizer.lemmatize(w),t))     
    del tags_remove_stopwords
#    print(tags_lemmatized)

    #generate arff
    
    for token in tags_lemmatized:
#        print(token)
        if token[1]=='MD':
            if any(token==d for d,s in bad_MD):
                f_bad_MD = f_bad_MD +1
            else:
                for bad,val in bad_MD:
                    if word_sim(token,bad)==1:
                        f_bad_MD=f_bad_MD+1
                        break                     
        elif token[1]=='NN':
            if any(token==d for d,s in bad_NN):
                f_bad_NN = f_bad_NN +1
            else:
                for bad,val in bad_NN:
                    if word_sim(token,bad)==1:
                        f_bad_NN=f_bad_NN+1
                        break                     
        elif token[1]=='VB':
            if any(token==d for d in bad_VB):
                f_bad_VB = f_bad_VB +1
            else:
                for bad,val in bad_VB:
                    if word_sim(token,bad)==1:
                        f_bad_VB=f_bad_VB+1
                        break                     
        elif token[1]=='RB':
            if any(token in d for d in bad_RB):
                f_bad_RB = f_bad_RB +1
            else:
                for bad,val in bad_RB:
                    if word_sim(token,bad)==1:
                        f_bad_RB=f_bad_RB+1
                        break                     
        elif token[1]=='JJ':
            if any(token in d for d in bad_JJ):
                f_bad_JJ = f_bad_JJ +1
            else:
                for bad,val in bad_JJ:
                    if word_sim(token,bad)==1:
                        f_bad_JJ=f_bad_JJ+1
                        break                     
        elif token[1]=='DT':
            if any(token in d for d in bad_DT):
                f_bad_DT = f_bad_DT +1
            else:
                for bad,val in bad_DT:
                    if word_sim(token,bad)==1:
                        f_bad_DT=f_bad_DT+1
                        break                     
        else: 
            if any(token in d for d in bad_punctuation):
                f_bad_punctuation = f_bad_punctuation +1
            else:
                for bad,val in bad_punctuation:
                    if word_sim(token,bad)==1:
                        f_bad_punctuation=f_bad_punctuation+1
                        break                     
                    
    #Hitung panjang kalimat
    try:
        num_sentences = len(nltk.sent_tokenize(sentence.decode())) 
    except AttributeError:
        num_sentences = len(nltk.sent_tokenize(sentence)) 

    num_words = len(words)
    
#    print(str(f_bad_MD) + "," + str(f_bad_NN) + ","+ str(f_bad_VB) + "," + str(f_bad_RB) + "," + str(f_bad_JJ)+ "," + str(f_bad_DT)+ "," + str(f_bad_punctuation) + "," + str(num_sentences) +  "," + str(num_words/num_sentences) + ",bad")    
    results = results + (str(f_bad_MD) + "," + str(f_bad_NN) + ","+ str(f_bad_VB) + "," + str(f_bad_RB) + "," + str(f_bad_JJ)+ "," + str(f_bad_DT)+ "," + str(f_bad_punctuation) + "," + str(num_sentences) +  "," + str(num_words/num_sentences) + ",badd\n")
    
#with open("dataset/DA/corpus_req_arff_0p5.txt", "w") as f1:    
with open(filename + "-slevel.txt" , "w") as f1:
    f1.write(results)
