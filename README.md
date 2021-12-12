# CS6907 Neural Networks Project Submission

### Overview

This project will consist of implementing the Doc2Vec network in 


### Scope Changes

Initially I started using the tf doc2vec model, after having some struggles with initial tf doc2vec tests I found the following threads that indicated gensim may be better than tf for doc2vec.  I wanted to use tf initially because I have used gensim in the past more than tf, however it seemed like the best bet for this project.
- https://groups.google.com/g/gensim/c/0GVxA055yOU
- https://stackoverflow.com/questions/39843584/gensim-doc2vec-vs-tensorflow-doc2vec

As such I ended up using the DBOW + DM approach outlined in this notebook: https://github.com/RaRe-Technologies/gensim/blob/3c3506d51a2caf6b890de3b1b32a8b85f7566ca5/docs/notebooks/doc2vec-IMDB.ipynb


### Research/Supporting Documentation:
- Paper from instructions: https://arxiv.org/pdf/1405.4053.pdf
- Relevant Paper: https://arxiv.org/pdf/1301.3781.pdf
- Word2Vec Tensorflow docs: https://www.tensorflow.org/tutorials/text/word2vec
- Sample implementation:
https://github.com/wangz10/tensorflow-playground/blob/master/doc2vec.py
- Sample Implementation: https://github.com/PacktPublishing/TensorFlow-Machine-
Learning-Cookbook/blob/master/Chapter%2007/doc2vec.py

### Results
Results have been compiled in a deck for class presentation.  If that deck is not in this repo, contact @samodle to get a copy.