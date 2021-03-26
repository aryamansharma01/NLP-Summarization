from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import math


def read_article(filename):
    file = open(filename, encoding="utf-8")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence)
    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    totalwords = list(set(sent1+sent2))

    v1 = [0] * len(totalwords)
    v2 = [0]*len(totalwords)

    for w in sent1:
        if w in stopwords:
            continue
        v1[totalwords.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        v2[totalwords.index(w)] += 1

    return 1 - cosine_distance(v1, v2)


def buildmatrix(sentences, stopwords):
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            matrix[i, j] = sentence_similarity(sentences[i], sentences[j])
    return matrix


def summary(filename, top_n=5):
    stop_words = stopwords.words('english')
    summary = []
    sentences = read_article(filename)
    top_n = math.ceil(0.2*len(sentences))
    matrix = buildmatrix(sentences, stopwords)
    graph = nx.from_numpy_array(matrix)
    scores = nx.pagerank(graph)
    rankedsentences = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                             reverse=True)
    #print("Indexes of top ranked sentences are : ", rankedsentences)
    for i in range(top_n):
        summary.append(rankedsentences[i][1])
    print("Summary is: \n", ". ".join(summary))


summary('finaltext', 5)
