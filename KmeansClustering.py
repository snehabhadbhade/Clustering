from operator import itemgetter
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file
import nltk
import gensim
import string
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import math
from scipy.spatial import distance
import json

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

lemmatiser = WordNetLemmatizer()


output_file("test.html")

stemmer = nltk.SnowballStemmer("english")

with open("What.txt","r") as f:
	input_text = f.readlines()
f.close()

input_data = []


def tokenize(text):
    lemmatized_words = []
    text = text.lower()
    text = text.strip("\n")
    text = re.sub(r'\\n',' ',text)
    text = re.sub(r'[^\w\s]',' ',text)
    tokens = nltk.word_tokenize(text)
    tokens_pos = pos_tag(tokens)
    count = 0
    for token in tokens:
        pos = tokens_pos[count]
        pos = get_wordnet_pos(pos[1])
        if pos != '':
            lemma = lemmatiser.lemmatize(token, pos)
        else:
            lemma = lemmatiser.lemmatize(token)
        lemmatized_words.append(lemma)
        count+=1
    return lemmatized_words


for row in input_text:
    row = row.strip("\n")
    input_data.append(row)

count_vectorizer = CountVectorizer(encoding="latin-1", stop_words="english", tokenizer=tokenize, analyzer='word')
Tfidf_vectorizer = TfidfVectorizer(encoding="latin-1", use_idf=True, stop_words="english", tokenizer=tokenize, analyzer='word')
vectorizer = TfidfVectorizer(encoding="latin-1", use_idf=True, stop_words="english", tokenizer=tokenize, analyzer='word')
input_freq = Tfidf_vectorizer.fit_transform(input_data).toarray()
input_vector = Tfidf_vectorizer.fit_transform(input_data)

num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(input_vector)
clusters = clustering_model.fit_predict(input_vector)

count=1
cluster_text = defaultdict(list)
docs_list =[]

for cluster in clusters:
    print cluster

count_data = len(input_data)

for i in range(count_data):
    cluster_id = clusters[i]
    doc_id = i+1
    cluster_text[cluster_id].append(doc_id)

cluster_terms = defaultdict(list)

print("Top terms per cluster:")
centroids = clustering_model.cluster_centers_
order_centroids = centroids.argsort()[:, ::-1]
terms = Tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :50]:
        print ' %s' % terms[ind],
        cluster_terms[i].append((ind,terms[ind]))
    print

cluster_freq = defaultdict(dict)


for cluster_id, terms in cluster_terms.items():
    doc_list = cluster_text[cluster_id]
    term_freq = defaultdict(int)
    for (index, word) in terms:
        term_count = 0
        for doc in doc_list:
            doc_id = doc - 1
            term_count+= input_freq[doc_id][index]
        term_freq[word] = term_count
    cluster_freq[cluster_id] = term_freq

with open('cluster_freq.json', 'w') as f:
    json.dump(cluster_freq, f)


answers = defaultdict(list)
for i in range(num_clusters):
    #print "Cluster %d:" % i
    doc_list = cluster_text[i]
    #print doc_list
    distances = []
    for doc in doc_list:
        doc_id = doc - 1
        doc_vector = input_vector[doc_id]
        v1 = np.array(centroids[i])
        v2 = np.array(doc_vector.toarray())
        distances.append((doc, distance.euclidean(v1, v2)))

    distances.sort(key = itemgetter(1))
    count = len(distances)
    #print count
    doc_id, dis = distances[0]
    answers[i].append(('best',input_data[doc_id-1]))
    doc_id, dis = distances[1]
    answers[i].append(('best',input_data[doc_id-1]))

    doc_id, dis = distances[count-1]
    answers[i].append(('worst', input_data[doc_id-1]))
    doc_id, dis = distances[count-2]
    answers[i].append(('worst',input_data[doc_id-1] ))



with open("answers.txt" , "w") as f:
    for answers in answers.values():
        for answer in answers:
            output = answer[1] + "\n"
            f.write(output)




