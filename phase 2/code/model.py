from typing import List, Dict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import string
stemmer = SnowballStemmer("english")


def train(training_docs: List[Dict]):
    
    N = len(training_docs)
    stp_words = list(stopwords.words('english'))

    tf = [None for _ in range(N)]
    cat_train = np.zeros(N)
    vocab = list()
    remove_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))

    for i in range(N):
        cat_train[i] = training_docs[i]['category']
        words = (training_docs[i]['title']+" "+training_docs[i]['body']).lower().translate(remove_punc).split()
        words = [stemmer.stem(word) for word in words if word not in stp_words]

        unq, cnt = np.unique(words, return_counts=True)
        vocab += unq.tolist()
        tf[i] = dict(zip(unq, cnt))

    vocab = np.unique(vocab)
    vocab = dict(zip(vocab, range(len(vocab))))
    
    text_c = [{} for _ in range(5)]
    prior, numWords = np.zeros(5), np.zeros(5)
    N_term = len(vocab)
    condprob = np.zeros((N_term, 5))
    
    for i in range(N):
        for word in tf[i]:
            if word in text_c[int(cat_train[i])]:
                text_c[int(cat_train[i])][word] += tf[i][word]
            else:
                text_c[int(cat_train[i])][word] = tf[i][word]

            numWords[int(cat_train[i])] += tf[i][word]

        prior[int(cat_train[i])] += 1

    prior /= N
    alpha = 0.419
    
    for word in list(vocab):
        for class_ in range(1, 5):
            if word in text_c[class_]:
                condprob[vocab[word]][class_] = (text_c[class_][word] + alpha) / (numWords[class_] + alpha * N_term)
            else:
                condprob[vocab[word]][class_] = (alpha / (numWords[class_] + alpha * N_term))
                
    return prior, condprob, vocab


def classify(prior, condprob, vocab, doc: Dict) -> int:
    remove_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    words = (doc['title']+" "+doc['body']).lower().translate(remove_punc).split()
    words = [stemmer.stem(word) for word in words]
    unq, cnt = np.unique(words, return_counts=True)
    tf = dict(zip(unq, cnt))
    
    score = np.array([0] + np.log10(np.array(prior[1:])).tolist())
    for class_ in range(1, 5):
        for term in tf:
            if term in vocab:
                score[class_] += np.log10(condprob[vocab[term]][class_])

    pred = np.argmax(score[1:]) + 1
    
    return pred



