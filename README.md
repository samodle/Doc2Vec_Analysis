# CS6907 Neural Networks Project Submission

## Data
"The Consumer Complaint Database is a collection of complaints about consumer financial products and services that we sent to companies for response. Complaints are published after the company responds, confirming a commercial relationship with the consumer, or after 15 days, whichever comes first. Complaints referred to other regulators, such as complaints about depository institutions with less than $10 billion in assets, are not published in the Consumer Complaint Database. The database generally updates daily."
Source: https://catalog.data.gov/dataset/consumer-complaint-database

## Key Files
- d2v_tf_source.py: Tensorflow d2v model creation and evaluation
- main.py: Gensim d2v model creation and evaluation
- Any pdf with 'output' in the name: Contains console output from successful test run of one of the models.

### Overview

This project will consist of implementing the Doc2Vec network in Tensorflow in order to classify documents.
The doc2vec results from Tensorflow will be compared against multiple GenSim doc2vec implementations.

 
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