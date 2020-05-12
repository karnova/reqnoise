# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:07:20 2019

@author: daniel
"""
import numpy as np
import math
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

filename = "dataset/DA/added/da-1-raw"
f = open(filename+ ".txt", "r")
corpus = f.read()
#corpus = np.genfromtxt("dataset/DA/da-1-raw.txt", delimiter ='\n', dtype=None)

#Requirements
default_stops = set(stopwords.words('english'))
new_stopwords = ['.', ',', '?', ':','-', '–', '/', 'the','’' ]
out_stopwords = {'our', 'you', 'we', 'they','would', 'should', 'but', 'will', 'how', 'all', 'who', 'ours',"wouldn't", "shouldn't", 'that', 'now', 'later''every',"tomorrow", "next", 'yesterday', 'it', 'she','he'}
add_stopwords = default_stops.union(new_stopwords)
#print(add_stopwords)
stops = set([word for word in add_stopwords if word not in out_stopwords])
#print(stops)
lemmatizer = WordNetLemmatizer()
sentences = corpus.splitlines()
dataset = []
for sent in sentences:    
    words = nltk.tokenize.word_tokenize(sent.lower())
    #del corpus
    tags = nltk.pos_tag(words)
    #print("total token: " + str(len(tags)))
    del words
    tags_remove_stopwords =[]
    for word,pos in tags:
        if word not in stops:
            tags_remove_stopwords.append((word,pos))
    #tags_remove_stopwords = [(word,pos) for word,pos in tags if not word in stops] 
    #print(len(tags_remove_stopwords))
    #print('total setelah stopword: ' + str(len(tags_remove_stopwords)))
    del tags
    word_lemmatized = ""
    for w,t in tags_remove_stopwords:
        if (t in ('NN','NNS','NNP')):
            word_lemmatized = word_lemmatized + lemmatizer.lemmatize(w,'n') + " "
        elif (t in ('VB','VBN','VBZ','VBG','VBP')):
            word_lemmatized = word_lemmatized + lemmatizer.lemmatize(w,'v') + " "
        else:
            word_lemmatized = word_lemmatized + lemmatizer.lemmatize(w) + " "     
    dataset.append(word_lemmatized)

#print(dataset)
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(dataset)
#print(vectorizer.get_feature_names())

lenx = len(vectorizer.get_feature_names())
leny = len(X.toarray())
A = X.toarray()
tf = [0] * lenx
idf = [0.0] * lenx
for i in range(0,leny):
    for j in range(0,lenx):
        if A[i][j]>0:
            tf[j] = tf[j]+1  
for j in range(0,lenx):
    idf[j] = math.log(leny/tf[j])
#print(tf)
#print(idf)

Af = np.zeros((leny,lenx))
for i in range(0,leny):
    for j in range(0,lenx):
        Af[i][j] = A[i][j]*idf[j]

tfidf = np.zeros((leny,leny))
for i in range(1,leny):
    for j in range(0,i):
        sumproduct = np.sum(Af[j]*Af[i])
        sumsqA = np.sum(Af[j]*Af[j])
        sumsqB = np.sum(Af[i]*Af[i])
        tfidf[i][j] = sumproduct / (math.sqrt(sumsqA) * math.sqrt(sumsqB))
        tfidf[j][i] = tfidf[i][j]
#        print "result {:>2f} == {:>2f} == {:>2f} == {:>2f}".format(sumproduct,sumsqA,sumsqB,tfidf[i][j])
#        print(str(sumproduct) + " == " +str(sumsqA) + " == " +str(sumsqB) + " == " +str(tfidf[i][j]))
#        print(Af[j])
#        print(Af[i])

#print(tfidf)
import pandas as pd

'''

df = pd.DataFrame(Af)
df.columns = df.columns+1
df.index = df.index + 1
#print(df1)    
df.to_csv(filename + "-wii.csv" , sep=';', encoding='utf-8')
'''

df1 = pd.DataFrame(tfidf)
df1.columns = df1.columns+1
df1.index = df1.index + 1
#print(df1)    
df1.to_csv(filename + "-tfidf.csv" , sep=';', encoding='utf-8')
    
#with open("dataset/DA/corpus_req_arff_0p5.txt", "w") as f1:
#    f1.write(results)