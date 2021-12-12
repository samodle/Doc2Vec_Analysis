import multiprocessing
from time import time  # To time our operations
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

# set up mechanism for timing how long the program takes to execute
t = time()

tqdm.pandas(desc="progress-bar")

if __name__ == '__main__':
    # freeze_support()  # include this if packaging as stand alone app/freezing

    print('Initialization Complete. Time: {} min'.format(round((time() - t) / 60, 2)))

    df = pd.read_csv('Consumer_Complaints.csv')
    df = df[['Consumer complaint narrative','Product']]
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    df.rename(columns = {'Consumer complaint narrative':'narrative'}, inplace = True)
    print(df.head(15))
