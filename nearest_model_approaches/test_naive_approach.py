from naive_approach_script import Naive_Approach

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  #reference parent directory
from helper_functions import functions
import unittest

class naive_approach_test(unittest.TestCase):



    def setUp(self):

        self. sample_phrases=[
            "All of the women on The Apprentice flirted with me - consciously or unconsciously. That's to be expected."
        ]

        self.model=Naive_Approach()

        self.data_path = "../Data/soaps_all.txt"
        self.embedding_path ="../../Model_For_word2Vec/GoogleNews-vectors-negative300.bin"

        self.naive = Naive_Approach()
        self.naive._distance_metric = "cosine"
        self.naive.load_embedding(self.embedding_path)

        self.NAIVE_MODEL=self.naive

        used_data = functions.get_soap_data(self.data_path)
        self.naive.fit(used_data[0:10000])  # Cute baby dataset :)

    def test_closest(self):
        text=self.sample_phrases[0]
        str_closest=self.naive.get_closest(text)

        str_desired_text="well  you're this hotshot doctor who  if circumstances were different  would be fighting all the women off " \
        "with a stick  but mostly it's  like  this whole drugging people onboard this boat thing just to impress this woman "

        self.assertTrue(str_desired_text in str_closest)

    def test_best_response(self):
        pass

        text = self.sample_phrases[0]
        str_response = self.naive.get_response(text)

        str_desired_text="which one do you work for"

        self.assertTrue(str_desired_text in str_response)

if __name__ == '__main__':
    unittest.main()