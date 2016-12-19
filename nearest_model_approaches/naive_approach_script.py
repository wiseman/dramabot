import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  #reference parent directory



from sklearn.neighbors import NearestNeighbors
from helper_functions import functions as hf
from helper_functions.timer import Timer
import pandas
import logging
import pickle

logger = logging.getLogger(__name__)

class Naive_Approach(object):

    #naive approach
    def __init__(self):


        self._fitted_data=None #this is interesting in that this is specifically the subset of data that is going to be fitted and that i need to be able to track the indices


        self._embedding=None #assume gensim w2v model


        self._distance_metric="euclidean" #type of distance metric
        self._search_method="brute" #how you search
        self._closest_number_of_elements=10

        self._neighbors = NearestNeighbors(
            self._closest_number_of_elements, metric=self._distance_metric, algorithm=self._search_method) #Keep it very simple



    def fit(self,ary_str_data):

        #rebuild the model each time you fit new data - so essentially reparameterize
        self._neighbors = NearestNeighbors(
            self._closest_number_of_elements, metric=self._distance_metric,
            algorithm=self._search_method)  # Keep it very simple

        vector_rep = [hf.average_vector(v, self._embedding) for v in ary_str_data]  # All the data
        self._fit_embedding(ary_str_data,vector_rep)


    def _fit_embedding(self,ary_text_data,ary_embedded_data):
        '''

        :param ary_text_data: the text associated with the embedding
        :param ary_embedded_data:
        :return:
        '''

        self._fitted_data = pandas.DataFrame()
        self._fitted_data["Transcript"] = ary_text_data
        self._fitted_data["Transcript"] = self._fitted_data["Transcript"].str.lower()
        self._fitted_data["index_value"] = self._fitted_data.index
        self._neighbors.fit(ary_embedded_data)




    def load_embedding(self,file_name):
        '''
        load em
        :param file_name: str relevant file name - a w2v gensim associated file
        :return:
        '''
        self._embedding=hf._get_w2v_embedding(file_name)

    def set_embedding(self,embedding):
        '''

        :param embedding: w2v gensim, sets the relevant emebdding
        :return:
        '''
        self._embedding=embedding


    def get_closest(self,str_statement):
        results=self._get_closest(str_statement,1)
        return results[0][0] #get the best closest result with nothing else

    def _get_closest(self,str_statement,n_closest=None):
        '''
        Get the closest in cosine sim using w2v  for N features
        :param str_statement:
        :param n_closest:
        :return:
        '''
        if n_closest == None:
            n_closest = self._closest_number_of_elements

        avg_embedded_vector =  hf.average_vector(str_statement,self._embedding) #convert statement into embedding

        T = Timer()
        distance, indices = self._neighbors.kneighbors([avg_embedded_vector],n_neighbors=n_closest)

        logger.info('Query time: %s s' % (T.elapsed()))

        results=[]
        counter=0 #for iterating through the elements - quick hack
        for counter in xrange(n_closest):

            best= indices[0][counter]
            # Get the correct location
            best_match_index = self._fitted_data.iloc[best].index_value #use the original dataset
            best_distance=distance[0][counter]
            str_best=self._fitted_data['Transcript'][best_match_index]

            results.append([str_best,best_match_index,best_distance])
            counter+=1
        return results






    def get_responses(self,statement,number):
        '''
        Get N best responses
        :param statement:
        :param number:
        :return:
        '''

        ary_str_best_responses=[]
        best = self._get_closest(statement, number)
        for top_n in best:

            best_indice=top_n[1]

            str_best = self._fitted_data['Transcript'][best_indice+1] #get ther esponse to the closest response
            ary_str_best_responses.append(str_best)
        return ary_str_best_responses

    def get_response(self,statement):
        '''
        The best response to the closest statement - expect this to be overriden quite a bit
        :param statement: str statement you're going to use
        :return:
        '''
        return self.get_responses(statement,1)[0]


