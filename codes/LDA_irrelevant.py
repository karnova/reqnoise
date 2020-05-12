# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:02:37 2019

@author: daniel
"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.tag import pos_tag

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from nltk import word_tokenize          


from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

#Requirements
from nltk.corpus import stopwords
default_stops = set(stopwords.words('english'))
new_stopwords = ['.', ',', '?', ':','-', '–', '/', 'the', 'a', 'us' , 'user' , '(',')','is','are','were' ]
#out_stopwords = {'our', 'you', 'we', 'they','would', 'should', 'but', 'will', 'how', 'all', 'who', 'ours',"wouldn't", "shouldn't", 'that', 'now', 'later''every',"tomorrow", "next", 'yesterday', 'it', 'she','he'}
stops= default_stops.union(new_stopwords)
pos_stops = {'DT', 'CC', 'CD','IN','MD','JJ'}


def removeStops(poss):
    for word,pos in reversed(poss):
        if word.lower() in stops:
            poss.remove((word,pos))
        elif pos in pos_stops:
            poss.remove((word,pos))            
    return poss

def doit(text):      
  import re
  matches=re.findall(r'\"(.+?)\"',text)
  # matches is now ['String 1', 'String 2', 'String3']
  return " ".join(matches)


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

    
# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health." 

filename = "dataset/DA/added/da-1-"
#File deskripsi
fdesc = open(filename+ "desc.txt", "r")
#file requirements
corpus_desc = fdesc.read()

f = open(filename+ "raw.txt", "r")
corpus = f.read()
num_req_topic = 4
num_doc_topic = 3

doc_set = corpus.split('\n')
texts_ref = []

# loop through document list
for i in doc_set:
    raw = i.lower()
    raw_tagged = pos_tag(word_tokenize(raw)) #use NLTK's part of speech tagger
    tagged = removeStops(raw_tagged)
    non_propernouns = [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'VB' or pos == 'VBN' or pos == 'VBZ' or pos == 'VBG' or pos == 'VBP']
    lemmatized_tokens = [lemmatizer.lemmatize(get_lemma(j)) for j in non_propernouns ]    
    texts_ref.append(lemmatized_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary_ref = corpora.Dictionary(texts_ref)
print(dictionary_ref)
corpus_ref = [dictionary_ref.doc2bow(text) for text in texts_ref]
print(corpus_ref)

# generate LDA model
ldamodel_req = gensim.models.ldamodel.LdaModel(corpus_ref, num_topics=num_req_topic, id2word = dictionary_ref, passes=20)
print(ldamodel_req.print_topics(num_topics=num_req_topic, num_words=3))

from gensim.models.coherencemodel import CoherenceModel


print('\nPerplexity: ', ldamodel_req.log_perplexity(corpus_ref))
coherence_model_lda = CoherenceModel(model=ldamodel_req, texts=texts_ref, dictionary = dictionary_ref, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


from nltk.tokenize import sent_tokenize
desc_set = sent_tokenize(corpus_desc)
desc = []

# loop through document list
for i in desc_set:
    raw = i.lower()
    
    raw_tagged = pos_tag(word_tokenize(raw)) #use NLTK's part of speech tagger
    tagged = removeStops(raw_tagged)
    non_propernouns = [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'VB' or pos == 'VBN' or pos == 'VBZ' or pos == 'VBG' or pos == 'VBP']
    lemmatized_tokens = [lemmatizer.lemmatize(get_lemma(j)) for j in non_propernouns ]    
    desc.append(lemmatized_tokens)

# turn our tokenized documents into a id <-> term dictionary
desc_dictionary = corpora.Dictionary(desc)
corpus_desc = [desc_dictionary.doc2bow(text) for text in desc]
#print(corpus)

# generate LDA model
ldamodel_desc = gensim.models.ldamodel.LdaModel(corpus_desc, num_topics=num_doc_topic, dictionary=dictionary_ref, passes=20)
#print(ldamodel_desc.print_topics(num_topics=num_doc_topic, num_words=3))
tword = ldamodel_desc.show_topics(num_topics=num_doc_topic, num_words=3)
desc_topics = []
for iid,iitem in tword:
    desc_topics.append(doit(iitem))
    
#print(desc_topics)    


for iitem in desc_topics:
    doc_topic = ldamodel_req.get_document_topics(dictionary_ref.doc2bow(iitem.split()))
    #print(iitem)
    #print(doc_topic)

selected_topic = []
thresh = 1 / num_req_topic    
for x in doc_topic:
    if x[1]>thresh:
        if x[0] not in selected_topic:
            selected_topic.append(x[0])
        

'''
for txt in corpus_ref:    
    sts = 1
    vec = ldamodel_req[txt]    
    for x in vec:
        if x[1]>thresh:
            if x[0] in selected_topic:
                sts=0
                break
    if sts==1:
        print(1)
    else:    
        print(0)
'''
#import pyLDAvis.gensim
#lda_display = pyLDAvis.gensim.prepare(ldamodel_req, corpus_ref, dictionary, sort_topics=False)
#pyLDAvis.show(lda_display)

#doc_topic = ldamodel_req.get_document_topics(dictionary.doc2bow(topic4.split()))
#print(doc_topic)


'''

#for i in doc_set:

uknown_text = "The user monitors the status of submitted jobs."
doc_topic = ldamodel.get_document_topics(dictionary.doc2bow(uknown_text.split()))

tagged = pos_tag(uknown_text.split()) #use NLTK's part of speech tagger
uknown_propernouns = [word for word,pos in tagged if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS' or pos == 'VB' or pos == 'VBN' or pos == 'VBZ' or pos == 'VBG' or pos == 'VBP']
print(uknown_propernouns)
print(dictionary.doc2bow(uknown_propernouns ))
print(doc_topic)


'''