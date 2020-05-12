from sklearn.cluster import KMeans
import numpy as np
import string
import nltk
import editdistance

from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
import matplotlib.pyplot as plt


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def eigenDecomposition(A, plot = True):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

def vectorizeBoW(allsentences):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words="english", max_df=0.5, input='content', encoding='utf-8',
                                                           decode_error='strict', strip_accents=None, lowercase=True,
                                                           preprocessor=None, tokenizer=LemmaTokenizer(),token_pattern='(?u)\\b\\w\\w+\\b',
                                                           ngram_range=(1, 1), analyzer='word',  min_df=10, max_features=None,
                                                           vocabulary=None, binary=False)
    X = vectorizer.fit_transform(allsentences)
    print("Features names....")
    print(vectorizer.get_feature_names())
    return X.toarray()


def tfidf(allsentences):
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False)
    X = tf_transformer.fit(allsentences)
    print(tf_transformer.get_feature_names())
    return X.toarray()


def createAffinityMatrix(vectors):
    from sklearn.metrics.pairwise import chi2_kernel
    K = chi2_kernel(vectors, gamma=.5)
    return K


def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def word_sim(token1, token2, idj):
    """Convert all characters to lowercase from list of tokenized words"""
    simi_score = 0.0    
    syn1 = None
    syn2 = None
    try:
        syn1 = wn.synset(token1[0] + '.' + penn2morphy(token1[1])+'.01')
        syn2 = wn.synset(token2[0] + '.' + penn2morphy(token2[1])+'.01')
    except WordNetError:
        simi_score=0.0
    
    if (syn1 is not None) and (syn2 is not None): 
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
    if simi_score>0.5:
        #print(word1,' ', word2,' ', simi_score)
        return 1
    return 0


    
stemmer = PorterStemmer()
pre_file = "dataset/DA/"
pre_file_des = "dataset/DA/added/"
filename = "da-7-"
#File actor
factor = open(pre_file + filename+ "actor.txt", "r")
corpus_actor = factor.read()
#fdesc = open(pre_file + filename+ "desc.txt", "r")

#File actions
#f = open(pre_file + filename+ "raw.txt", "r")
faction = open(pre_file + filename+ "action.txt", "r")
corpus_action = faction.read()
#print(corpus)


#Requirements
from nltk.corpus import stopwords
default_stops = set(stopwords.words('english'))
new_stopwords = ['','.', ',', '?', ':','-', '–', '/', 'the', 'a', 'us' , '(',')' ]
out_stopwords = {'our', 'you', 'we', 'they','would', 'should', 'but', 'will', 'how', 'all', 'who', 'ours',"wouldn't", "shouldn't", 'that', 'now', 'later''every',"tomorrow", "next", 'yesterday', 'it', 'she','he'}
stops= default_stops.union(new_stopwords)
pos_stops = {'DT', 'CC', 'CD','IN','MD','JJ'}
#print(stops)
#print(pos_stops)

import re
#actors
actors = corpus_actor.split('\n')
tokenized_actors = [re.split(r'[;]', sentence.lower()) for sentence in actors]
#tokenized_actors = [nltk.word_tokenize(sentence) for sentence in actors]

#actions
actions = corpus_action.split('\n')
tokenized_actions = [re.split(r'[;]', sentence.lower()) for sentence in actions]
#tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]



#actors
good_actors = []    
for tags in tokenized_actors:
    for word in reversed(tags):
        if word.lower() in stops:
            tags.remove(word)
        else:
            good_actors.append(word)
        

#actions
good_actions = []    
for tags in tokenized_actions:
    for word in reversed(tags):
        if word.lower() in stops:
            tags.remove(word)
        else:
            good_actions.append(word)


from collections import Counter
#count_good_raw = Counter(good_raw)
count_good_actors  = Counter(good_actors)
count_good_actions = Counter(good_actions)
#number of statements
nos = len(tokenized_actions)
#number of good actors
noga = len(count_good_actors)
#number of good actors
nogc = len(count_good_actions)


PICKLE = "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
import nltk.data
from nltk.tag import PerceptronTagger
_nltk_pos_tagger = PerceptronTagger(load=False)
_nltk_pos_tagger.load(PICKLE)
print(count_good_actors)
S = np.zeros(shape=(nos,noga+nogc))
i=0
for sent_pos in tokenized_actors:    
    for token1 in sent_pos:
        j=0
        tt1 = _nltk_pos_tagger.tag([token1])
        for feature in count_good_actors:  
            ft = _nltk_pos_tagger.tag([feature])
            simval = word_sim(tt1[0],ft[0] , i)
            S[i][j] = S[i][j] + simval  
            j=j+1
    i=i+1

i=0
for sent_pos in tokenized_actions:    
    for token1 in sent_pos:
        j=noga
        tt1 = _nltk_pos_tagger.tag([token1])
        for feature in count_good_actions:
            ft = _nltk_pos_tagger.tag([feature])
            simval = word_sim(tt1[0],ft[0] , i)
            S[i][j] = S[i][j] + simval  
            j=j+1
    i=i+1


print(S)


A = np.array(createAffinityMatrix(S))
#A = np.array([[0,1],[-2,-3]])
#print(A)


# find eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(A)



# diagonal matrix
D = np.diag(A.sum(axis=1))

# graph laplacian
L = D-A

# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(L)

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

#Calculate the optimal number of clusters
k, _,  _ = eigenDecomposition(A)
#print(f'Optimal number of clusters {k}')


nb_clusters = np.argmax(np.diff(vals))
#print(f'Optimal number of clusters {nb_clusters}')


k=11
# kmeans on first three vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters=k)
try:
    kmeans.fit(vecs[:,1:4])
except:
    vecs = vecs.real
    kmeans.fit(vecs[:,1:4])
colors = kmeans.labels_

print("Clusters:", colors)

# Clusters: [2 1 1 0 0 0 3 3 2 2]

#affinity_matrix = getAffinityMatrix(X, k = 10)
#k, _,  _ = eigenDecomposition(affinity_matrix)


Sum_of_squared_distances = []
K = range(1,len(S))
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(vecs[:,1:4])
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

