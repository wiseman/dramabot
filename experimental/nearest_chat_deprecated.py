import gensim
import pandas
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle


def get_soap_data(location=None):
    if type(location) == type(None):
        location = "D:\dialogue_agent\dr_word2vec_code\input_data\soaps_all.txt"  # Change direcotry

    f = open(location, mode="r")
    content = f.readlines()
    f.close()
    return content


def average_vector(doc, embedding):  # just average all the words in a sentence
    if embedding == None:
        raise Exception("in average vector function, w2v embedding does not exist or has not been loaded ")
    doc = doc.rstrip()
    doc = doc.split()
    size = 0
    full_model = [0] * embedding.layer1_size
    for key in doc:
        try:
            ary = embedding[key]
            size += 1
            full_model += ary
        except:
            pass

    if size != 0: full_model = np.array(full_model) / float(size)
    return full_model


def _get_w2v_embedding(name=None):
    if name == None:
        name = "D:\dialogue_agent\GoogleNews-vectors-negative300.bin"  # change directory

    try:
        embedding = gensim.models.word2vec.Word2Vec.load_word2vec_format(name, binary=True)
    except:  # maybe it was saved in a different format

        embedding = gensim.models.word2vec.Word2Vec.load(name)

    return embedding


def quick_save(name, embedded_data):
    pickle.dump(embedded_data, open(name + ".p", "wb"))
    result = pickle.load(open(name + ".p", "rb"))


if __name__ == '__main__':
    text = get_soap_data()
    print len(text)
    embedding = _get_w2v_embedding()

    data = pandas.DataFrame()
    data["Transcript"] = text[0:2000000]

    data["Transcript"] = data["Transcript"].str.lower()
    data["index_value"] = data.index
    vals = data["Transcript"].values

    vector_rep = [average_vector(s, embedding) for s in vals]

    # quick_save("big_ver",vector_rep)

    neighbors = NearestNeighbors(n_neighbors=10, metric="euclidean")
    neighbors.fit(vector_rep)

    threshold = .6  # Of the top N, take the longest response

    while True:
        sentence = raw_input("ENTER YOUR SENTENCE - THIS WOULD NORMALLY BE AN API CALL")
        sentence = sentence.lower()
        embedded = average_vector(sentence, embedding)

        distance, indices = neighbors.kneighbors([embedded])

        best = indices[0][0]

        indice = data.iloc[best].index_value  # Get the correct location

        print distance
        print indice
        best_response = indice + 1

        print data["Transcript"][indice + 1]