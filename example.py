from nearest_model_approaches.naive_approach_script import Naive_Approach
from helper_functions import functions
if __name__ == '__main__':


    data_path="Data/soaps_all.txt"
    embedding_path="../../Model_For_word2Vec/GoogleNews-vectors-negative300.bin"

    used_data=functions.get_soap_data(data_path)

    naive=Naive_Approach()
    naive._distance_metric="cosine"
    naive.load_embedding(embedding_path)
    naive.fit(used_data[0:30000]) #Cute baby model :)

    sample_phrases=[
            "All of the women on The Apprentice flirted with me - consciously or unconsciously. That's to be expected."

        ]

    result= naive.get_closest(sample_phrases[0])
    print result

    print naive.get_response(sample_phrases[0])

