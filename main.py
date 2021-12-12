import multiprocessing
from time import time  # To time our operations
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import TaggedDocument
import Helpers
import nltk

nltk.download('punkt')  # This may not be necessary once you've done it or if you have it
verbose = True

# set up mechanism for timing how long the program takes to execute
t = time()

tqdm.pandas(desc="progress-bar")
cores = multiprocessing.cpu_count()

if __name__ == '__main__':
    # freeze_support()  # include this if packaging as stand alone app/freezing

    if verbose:
        print('Initialization Complete. Time: {} min'.format(round((time() - t) / 60, 2)))

    # First, import the dataset
    df = pd.read_csv('RawData/complaints.csv')
    df = df[['Consumer complaint narrative', 'Product']]
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    df.rename(columns={'Consumer complaint narrative': 'narrative'}, inplace=True)

    if verbose:
        print('Before Preprocessing:')
        print(df.head(5))
        print(f'   Dataframe Shape: {df.shape}')
        num_words = df['narrative'].apply(lambda x: len(x.split(' '))).sum()
        print(f'   Number of words in dataset: {num_words}')

    # Preprocess the text
    df['narrative'] = df['narrative'].apply(Helpers.clean_text)

    if verbose:
        print('Preprocessing Complete. Time: {} min'.format(round((time() - t) / 60, 2)))
        num_words_2 = df['narrative'].apply(lambda x: len(x.split(' '))).sum()
        print(f'   Number of words in dataset: {num_words_2}, delta: -{(num_words - num_words_2)}')

    # Train Test split for our data
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=Helpers.tokenize_text(r['narrative']), tags=[r.Product]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=Helpers.tokenize_text(r['narrative']), tags=[r.Product]), axis=1)

    # Setup the model
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    if verbose:
        print('Ready To Train Model. Time: {} min'.format(round((time() - t) / 60, 2)))

    num_epochs_dbow = 30
    i = 0
    for epoch in range(num_epochs_dbow):
        print(f'{i}/{num_epochs_dbow}: ', end='')
        i += 1
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                         epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
    print('Training Complete.')

    # Train the classifier
    y_train, X_train = Helpers.vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = Helpers.vec_for_learning(model_dbow, test_tagged)

    if verbose:
        print('Ready To Fit Classifier. Time: {} min'.format(round((time() - t) / 60, 2)))
    log_reg = LogisticRegression(n_jobs=1, C=1e5)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    if verbose:
        print('Results. Time: {} min'.format(round((time() - t) / 60, 2)))
        print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
        print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

    # Distributed Memory
    num_epochs_dmm = 30
    i = 0
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                        min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])
    for epoch in range(num_epochs_dmm):
        print(f'{i}/{num_epochs_dmm}: ', end='')
        i += 1
        model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                        epochs=1)
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    print('Training Complete.')

    # logistic regression classifier
    y_train, X_train = Helpers.vec_for_learning(model_dmm, train_tagged)
    y_test, X_test = Helpers.vec_for_learning(model_dmm, test_tagged)

    if verbose:
        print('Ready To Fit Classifier. Time: {} min'.format(round((time() - t) / 60, 2)))
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

    # clean it up
    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # concatenate distributed memory w/ bag of words
    model_combo = ConcatenatedDoc2Vec([model_dbow, model_dmm])

    y_train, X_train = Helpers.vec_for_learning(model_combo, train_tagged)
    y_test, X_test = Helpers.vec_for_learning(model_combo, test_tagged)

    if verbose:
        print('Ready To Fit Classifier. Time: {} min'.format(round((time() - t) / 60, 2)))

    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
    print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))