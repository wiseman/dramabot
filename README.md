# dramabot
Twitter bot that responds dramatically, like a character in a soap opera.


/Data:
    Contains relevant soap data - unzip to use
    I recommend google news w2v model for your embedding: https://code.google.com/archive/p/word2vec/
    You can also use the soap data and retrain a smaller model. Check out gensim for more details to do so.

example.py:
    An instance of calling the relevant embedding and dataset to get a response

nearest_chat_bot.py:
    Play around with a for loop entering arbitrary sentences to see how it feels :)