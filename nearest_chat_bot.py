import itertools
import logging
import pickle
import sys
import time

import gensim
import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors
from nearest_model_approaches.naive_approach_script import Naive_Approach
from helper_functions import functions

logger = logging.getLogger(__name__)


LOGGING_FORMAT = ('%(threadName)s:%(asctime)s:%(levelname)s:%(module)s:'
                  '%(lineno)d %(message)s')

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)




def main():

    if len(sys.argv) != 3:
        sys.stderr.write('Error: wrong number of arguments.\n')
        sys.stderr.write(
            'Usage: %s <corpus path> <model path>\n' % (sys.argv[0],))
        return 1

    text = functions.get_soap_data(sys.argv[1]) #soap directory

    naive = Naive_Approach()

    naive.load_embedding(sys.argv[2])
    naive.fit(text[0:30000])  # Cute baby model :)

    while True:
        sentence = raw_input("Enter some text:\n")
        sentence = sentence.lower()
        print "closest matching statment: \n %s" % naive.get_closest(sentence)

        print "closest response \n %s" % naive.get_response(sentence)


if __name__ == '__main__':
    main()
