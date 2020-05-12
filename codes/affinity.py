# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 21:01:59 2019

@author: daniel
"""
import pandas as pd
import numpy as np
import editdistance
from nltk.corpus import wordnet
from difflib import SequenceMatcher

from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


sentences = np.genfromtxt("dataset/DA/da-8.txt", delimiter ='\n', dtype=None)
dataset=[]
for sentence in sentences:
    token = sentence.decode("utf-8").split(";")
    dataset.append(token);

#for ds in dataset:
#    words = np.
#print(dataset)

fields = []
for row in dataset :
    for cell in row:
        try:
            idx = fields.index(cell)
        except ValueError:
            fields.append(cell)
#print(fields)
        
duplicates=[]
compare1 = fields
compare2 = fields
for x, word1 in enumerate(compare1):
    for y, word2 in enumerate(compare2):
        if x<y:
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            simi_score = 0
            if wordFromList1 and wordFromList2: 
                simi_score = wordFromList1[0].wup_similarity(wordFromList2[0])                    
                if simi_score is not None:
                    temp=0
                else:
                    tdist = editdistance.eval(word1, word2)
                    kata1 = len(word1)
                    kata2 = len(word2)
                    simi_score = 1 - (tdist / max(kata1, kata2))
            else:
                tdist = editdistance.eval(word1, word2)
                kata1 = len(word1)
                kata2 = len(word2)
                simi_score = 1 - (tdist / max(kata1, kata2))
            if simi_score>0.6:
                #print(word1,' ', word2,' ', simi_score)
                fields.remove(word2)
#print(fields)
        
#Aff = np.zeros(len(sentences), len(fields))
Aff = [ [0] * len(fields) for _ in range(len(sentences))]
for x, row in enumerate(dataset) :
    print(row)
    for z, field in enumerate(fields):          
        for y,cell in enumerate(row):
            wordFromList1 = wordnet.synsets(field)
            wordFromList2 = wordnet.synsets(cell)
            simi_score = 0
            if wordFromList1 and wordFromList2: 
                simi_score = wordFromList1[0].wup_similarity(wordFromList2[0])                    
                if simi_score is not None:
                    temp=0
                else:
                    tdist = editdistance.eval(field, cell)
                    kata1 = len(field)
                    kata2 = len(cell)
                    simi_score = 1 - (tdist / max(kata1, kata2))
            else:
                tdist = editdistance.eval(field, cell)
                kata1 = len(field)
                kata2 = len(cell)
                simi_score = 1 - (tdist / max(kata1, kata2))
            if simi_score>0.6:
                Aff[x][z] =Aff[x][z] + 1

#print(Aff)


#Normalization
scaler = MinMaxScaler()
Aff1 = scaler.fit_transform(Aff)
print(Aff1)
#Aff1= pd.DataFrame(ageAndFare, columns = ["age", "fare"])
#ageAndFare.plot.scatter(x = "age", y = "fare")
#print(Aff1)


#clustering and noise detection

outlier_detection = DBSCAN(
  eps = 0.5,
  metric="euclidean",
  min_samples = 3,
  n_jobs = -1)
clusters = outlier_detection.fit_predict(Aff1)
print("DBSCAN....")
print(clusters)

clusters2 = SpectralClustering(n_clusters=2,assign_labels="discretize", random_state=0).fit(Aff1)
print("SpectralClustering....")
print(clusters2.toarray())

#clustering dengan pynomaly
from PyNomaly import loop
m = loop.LocalOutlierProbability(Aff1).fit()
scores = m.local_outlier_probabilities
print("LocalOutlierProbability....")
print(scores)

#clustering dengan pynomaly with distance: Nearest Neighbour
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=3, metric='hamming')
neigh.fit(Aff1)
d, idx = neigh.kneighbors(Aff1, return_distance=True)

m = loop.LocalOutlierProbability(distance_matrix=d, neighbor_matrix=idx, n_neighbors=3).fit()
scores = m.local_outlier_probabilities
print("NearestNeighbor....")
print(scores)

