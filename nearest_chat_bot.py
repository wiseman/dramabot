import itertools
import logging
import pickle
import sys
import time

import gensim
import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors

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


def _get_w2v_embedding(name=None):
    logger.info('Loading %s', name)
    timer = Timer()
    try:
        embedding = gensim.models.word2vec.Word2Vec.load_word2vec_format(
            name, binary=True)
    except:  # maybe it was saved in a different format
        embedding = gensim.models.word2vec.Word2Vec.load(name)
    logger.info('Loaded %s in %s s', name, timer.elapsed())
    return embedding


def quick_save(name, embedded_data):
    pickle.dump(embedded_data, open(name + ".p", "wb"))
    result = pickle.load(open(name + ".p", "rb"))


LOGGING_FORMAT = ('%(threadName)s:%(asctime)s:%(levelname)s:%(module)s:'
                  '%(lineno)d %(message)s')


def average_vectors(vecs, embedding):
    return [average_vector(v, embedding) for v in vecs]


class Timer(object):
    """Keeps track of wall-clock time."""
    def __init__(self):
        self.start_time = None
        self.reset()

    def reset(self):
        """Resets the timer."""
        self.start_time = time.time()

    def elapsed(self):
        """Returns the elapsed time in seconds.

        Elapsed time is the time since the timer was created or last
        reset.
        """
        return time.time() - self.start_time


def main():
    if len(sys.argv) != 3:
        sys.stderr.write('Error: wrong number of arguments.\n')
        sys.stderr.write(
            'Usage: %s <corpus path> <model path>\n' % (sys.argv[0],))
        return 1
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    text = get_soap_data(sys.argv[1])
    embedding = _get_w2v_embedding(sys.argv[2])
    data = pandas.DataFrame()
    data["Transcript"] = text[0:200000]

    data["Transcript"] = data["Transcript"].str.lower()
    data["index_value"] = data.index
    vals = data["Transcript"].values

    logger.info('Averaging')
    at = Timer()
    vector_rep = [average_vector(v, embedding) for v in vals]
    logger.info('Averaging took %s s', at.elapsed())
    # logger.info('Reassembling')
    # vector_rep = reduce(lambda a, b: a + b, vector_reps)
    # vector_rep = [average_vector(s, embedding) for s in vals]
    # logger.info('Saving vector...')
    # quick_save("big_ver", vector_rep)

    logger.info('Nearest neighbors fit')
    nnt = Timer()
    neighbors = NearestNeighbors(
        n_neighbors=10, metric="euclidean", algorithm='ball_tree')
    neighbors.fit(vector_rep)
    logger.info('Fitting took %s s', nnt.elapsed())

    threshold = .6  # Of the top N, take the longest response

    for i in range(5):
        t = Timer()
        embedded = average_vector(
            'how many women have you slept with', embedding)
        distance, indices = neighbors.kneighbors([embedded])
        print 'Query time: %s s' % (t.elapsed(),)

    while True:
        sentence = raw_input("Enter some text:\n")
        sentence = sentence.lower()
        embedded = average_vector(sentence, embedding)
        distance, indices = neighbors.kneighbors([embedded])
        for best in indices[0][0:5]:
            # Get the correct location
            best_match_index = data.iloc[best].index_value
            print 'Best match: %s' % (
                data['Transcript'][best_match_index],)
            print 'Response1:   %s' % (
                data['Transcript'][best_match_index + 1],)


if __name__ == '__main__':
    main()
