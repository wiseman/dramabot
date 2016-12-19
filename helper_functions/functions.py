
import itertools
import logging
import pickle
import sys
import time

import gensim
import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re as reg
from timer import Timer

logger = logging.getLogger(__name__)

def get_soap_data(location):
    logger.info('Loading corpus %s', location)
    timer = Timer()
    with open(location, 'r') as f:
        content = f.readlines()
    logger.info('Loaded %s lines in %s s', len(content), timer.elapsed())
    return content


def grouper(iterable, n, fillvalue=None):
    """Collects data into fixed-length chunks or blocks.
    Short blocks are filled with fillvalue. For example,

      grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def grouper_nofill(iterable, n):
    """Collects data into fixed-length chunks or blocks.
    Short blocks are not filled. For example,

      grouper_nofill('ABCDEFG', 3) --> ABC DEF G
    """
    fillvalue = list([1])

    def pred(e):
        return not(e is fillvalue)
    return tuple(
        filter(pred, x) for x in grouper(iterable, n, fillvalue=fillvalue))


def average_vector(doc, embedding):  # just average all the words in a sentence
    words = doc.split()
    size = 0
    full_model = [0] * embedding.layer1_size
    for key in words:
        try:
            ary = embedding[key]
            size += 1
            full_model += ary
        except KeyError:
            pass

    if size != 0:
        full_model = np.array(full_model) / float(size)
    return full_model

def clean_text(self,text,labels):
        text=text.lower()


        text = reg.sub("[^a-z|A-Z|']", " ", text, max=100) #all non letter words
        text = reg.sub("\s+", " ", text) #take out all them sapces

        return text


def _get_w2v_embedding(name=None): #load embedding
    logger.info('Loading %s', name)
    timer = Timer()
    try:
        embedding = gensim.models.word2vec.Word2Vec.load_word2vec_format(
            name, binary=True)
    except:  # maybe it was saved in a different format
        embedding = gensim.models.word2vec.Word2Vec.load(name)
    logger.info('Loaded %s in %s s', name, timer.elapsed())
    return embedding


def quick_save(name, embedded_data): #quick save model
    pickle.dump(embedded_data, open(name + ".p", "wb"))
    result = pickle.load(open(name + ".p", "rb"))