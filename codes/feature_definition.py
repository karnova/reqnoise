# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:07:20 2019

@author: daniel
"""
import nltk
import copy
import csv
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix


#Requirements
default_stops = set(stopwords.words('english'))
new_stopwords = ['.', ',', '?', ':','-', '–', '/', 'the','’' ]
out_stopwords = {'our', 'you', 'we', 'they','would', 'should', 'but', 'will', 'how', 'all', 'who', 'ours',"wouldn't", "shouldn't", 'that', 'now', 'later''every',"tomorrow", "next", 'yesterday', 'it', 'she','he'}
add_stopwords = default_stops.union(new_stopwords)
#print(add_stopwords)
stops = set([word for word in add_stopwords if word not in out_stopwords])
#print(stops)
lemmatizer = WordNetLemmatizer()

f = open("dataset/DA/corpus_req.txt", "r")
corpus = f.read()
words = nltk.tokenize.word_tokenize(corpus.lower())
del corpus
tags = nltk.pos_tag(words)
print("total token: " + str(len(tags)))
del words
tags_remove_stopwords =[]
for word,pos in tags:
    if word not in stops:
        tags_remove_stopwords.append((word,pos))
#tags_remove_stopwords = [(word,pos) for word,pos in tags if not word in stops] 
#print(len(tags_remove_stopwords))
print('total setelah stopword: ' + str(len(tags_remove_stopwords)))
del tags
tags_lemmatized = []
for w,t in tags_remove_stopwords:
    if (t in ('NN','NNS','NNP')):
        tags_lemmatized.append((lemmatizer.lemmatize(w,'n'),'NN')) 
    elif (t in ('VB','VBN','VBZ','VBG','VBP')):
        tags_lemmatized.append((lemmatizer.lemmatize(w,'v'),'VB')) 
    else:
        tags_lemmatized.append((lemmatizer.lemmatize(w),t))     
#print(tags_lemmatized)

count_good_raw = Counter(tags_lemmatized)
del tags_lemmatized
print('total counter: ' + str(len(count_good_raw)))
count_good = copy.deepcopy(count_good_raw)
for tupple in count_good_raw:
    if count_good_raw[tupple]<3:
        del count_good[tupple]
del count_good_raw        
#print(count_good.most_common())
print(len(count_good))


f = open("dataset/DA/corpus_ireq.txt", "r")
corpus = f.read()
words = nltk.tokenize.word_tokenize(corpus.lower())
del corpus
tags = nltk.pos_tag(words)
print("total token: " + str(len(tags)))
#print(tags)
del words
tags_remove_stopwords =[]
for word,pos in tags:
    if word not in stops:
        tags_remove_stopwords.append((word,pos))
#tags_remove_stopwords = [(word,pos) for word,pos in tags if not word in stops] 
#print(len(tags_remove_stopwords))
print('total setelah stopword: ' + str(len(tags_remove_stopwords)))
del tags
tags_lemmatized = []
for w,t in tags_remove_stopwords:
    if (t in ('NN','NNS','NNP')):
        tags_lemmatized.append((lemmatizer.lemmatize(w,'n'),'NN')) 
    elif (t in ('VB','VBN','VBZ','VBG','VBP')):
        tags_lemmatized.append((lemmatizer.lemmatize(w,'v'),'VB')) 
    else:
        tags_lemmatized.append((lemmatizer.lemmatize(w),t))     
#print(tags_lemmatized)

count_bad_raw = Counter(tags_lemmatized)
del tags_lemmatized
print('total counter: ' + str(len(count_bad_raw)))
count_bad = copy.deepcopy(count_bad_raw)
for tupple in count_bad_raw:
    if count_bad_raw[tupple]<3:
        del count_bad[tupple]
del count_bad_raw        
#print(count_bad.most_common())
print(len(count_bad))
f_good_MD = 0
f_good_NN = 0
f_good_VB = 0
f_good_punctuation = 0
f_good_RB = 0
f_good_JJ = 0
f_good_DT = 0
f_bad_MD = 0
f_bad_NN = 0
f_bad_VB = 0
f_bad_punctiation = 0
f_bad_RB = 0
f_bad_JJ = 0
f_bad_DT = 0

bad_MD = []
bad_NN = []
bad_VB = []
bad_punctiation = []
bad_RB = []
bad_JJ = []
bad_DT = []

for tup in count_good:    
    f_good = count_good[tup]
    f_bad = count_bad[tup]
    if (f_good+f_bad)>0:
        LR = f_bad / (f_good+f_bad)
    else:
        LR=0
#        print(str(tup) + " == " + str(f_bad) + " + " +str(f_good))        
    if tup[1]=='MD':
        f_bad_MD = f_bad_MD +f_bad
        f_good_MD = f_good_MD +f_good
        bad_MD.append((tup,LR))
    elif (tup[1]=='NN'):
        f_bad_NN = f_bad_NN +f_bad
        f_good_NN = f_good_NN +f_good
        bad_NN.append((tup,LR))
    elif (tup[1]=='VB'):
        f_bad_VB = f_bad_VB +f_bad
        f_good_VB = f_good_VB +f_good
        bad_VB.append((tup,LR))
    elif (tup[1]=='RB'):
        f_bad_RB = f_bad_RB +f_bad
        f_good_RB = f_good_RB +f_good
        bad_RB.append((tup,LR))
    elif (tup[1]=='JJ'):
        f_bad_JJ = f_bad_JJ +f_bad
        f_good_JJ = f_good_JJ +f_good
        bad_JJ.append((tup,LR))
    elif (tup[1]=='DT'):
        f_bad_DT = f_bad_DT +f_bad
        f_good_DT = f_good_DT +f_good
        bad_DT.append((tup,LR))
    else:
        f_bad_punctiation = f_bad_punctiation +f_bad
        f_good_punctuation = f_good_punctuation +f_good
        bad_punctiation.append((tup,LR))
print("jumlah words")
print(len(bad_MD))    
print(len(bad_NN))    
print(len(bad_VB))    
print(len(bad_RB))    
print(len(bad_JJ))    
print(len(bad_DT))    
print(len(bad_punctiation))    

#print(bad_NN)    
"""
threshold=0.5
bLR_MD = threshold
bLR_NN = threshold
bLR_VB = threshold 
bLR_RB = threshold 
bLR_JJ = threshold 
bLR_DT = threshold 
bLR_punctuation = threshold 
"""    
bLR_MD = f_bad_MD/(f_bad_MD+f_good_MD) 
bLR_NN = f_bad_NN/(f_bad_NN+f_good_NN) 
bLR_VB = f_bad_VB/(f_bad_VB+f_good_VB) 
bLR_RB = f_bad_RB/(f_bad_RB+f_good_RB) 
bLR_JJ = f_bad_JJ/(f_bad_JJ+f_good_JJ) 
bLR_DT = f_bad_DT/(f_bad_DT+f_good_DT) 
bLR_punctuation = f_bad_punctiation/(f_bad_punctiation+f_good_punctuation) 

bad_tags=[]
print(str(bLR_MD) + " = " + str(f_bad_MD) + " + " +str(f_good_MD))        
print(str(bLR_NN) + " = " + str(f_bad_NN) + " + " +str(f_good_NN))        
print(str(bLR_VB) + " = " + str(f_bad_VB) + " + " +str(f_good_VB))        
print(str(bLR_RB) + " = " + str(f_bad_RB) + " + " +str(f_good_RB))        
print(str(bLR_JJ) + " = " + str(f_bad_JJ) + " + " +str(f_good_JJ))        
print(str(bLR_DT) + " = " + str(f_bad_DT) + " + " +str(f_good_DT))        
print(str(bLR_punctuation) + " = " + str(f_bad_punctiation) + " + " +str(f_good_punctuation))        



fbad_MD = []
for tup,v in bad_MD:
    if v>bLR_MD:
        fbad_MD.append((tup,v))
del bad_MD
bad_tags.append(fbad_MD)
fbad_NN = []
for tup,v in bad_NN:
    if v>bLR_NN:
        fbad_NN.append((tup,v))
del bad_NN
bad_tags.append(fbad_NN)
fbad_VB = []
for tup,v in bad_VB:
    if v>bLR_VB:
        fbad_VB.append((tup,v))
del bad_VB
bad_tags.append(fbad_VB)
fbad_RB = []
for tup,v in bad_RB:
    if v>bLR_RB:
        fbad_RB.append((tup,v))
del bad_RB
bad_tags.append(fbad_RB)
fbad_JJ = []
for tup,v in bad_JJ:
    if v>bLR_JJ:
        fbad_JJ.append((tup,v))
del bad_JJ
bad_tags.append(fbad_JJ)
fbad_DT = []
for tup,v in bad_DT:
    if v>bLR_DT:
        fbad_DT.append((tup,v))
del bad_DT
bad_tags.append(fbad_DT)
fbad_punctiation = []
for tup,v in bad_punctiation:
    if v>bLR_punctuation:
        fbad_punctiation.append((tup,v))
del bad_punctiation
bad_tags.append(fbad_punctiation)

print("jumlah bad words")
print(len(fbad_MD))    
print(len(fbad_NN))    
print(len(fbad_VB))    
print(len(fbad_RB))    
print(len(fbad_JJ))    
print(len(fbad_DT))    
print(len(fbad_punctiation))    


with open("dataset/DA/bad_tags_0p0.csv", "w") as f:
    csvwriter = csv.writer(f, delimiter=';', dialect='excel') 
    csvwriter.writerows(bad_tags)

#print(fbad_MD)    
    
#from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(sentences)
#print(count_vect.get_feature_names())
#print(X_train_counts)

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)

"""

sentences = np.genfromtxt("dataset/DA/corpus_req.txt", delimiter ='\n', dtype=None)
dataset=[]
for sentence in sentences:
    #print(sentence.lower())
    token = sentence.lower()
    dataset.append(token);


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

sent = "This is a foo bar sentence."
text= pos_tag(word_tokenize(sent))
#print(text)

from collections import Counter
count= Counter([j for i,j in pos_tag(word_tokenize(sent))])
#print (count)
"""

#print(X_train_tf)