import collections
from time import time  # To time our operations
import Helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

DATASET_SIZE = 100000
TRAIN_PCT = 0.5
KEY_COLUMN_NAME = 'narrative'
LABEL_COLUMN_NAME = 'Product'

# set up mechanism for timing how long the program takes to execute
t = time()
np.random.seed(42)
verbose = True

if verbose:
    print('Initialization Complete. Time: {} min'.format(round((time() - t) / 60, 2)))

# First, import the dataset
df = pd.read_csv('RawData/complaints.csv')
df = df[['Consumer complaint narrative', LABEL_COLUMN_NAME]]
df = df[pd.notnull(df['Consumer complaint narrative'])]
df.rename(columns={'Consumer complaint narrative': KEY_COLUMN_NAME}, inplace=True)
df = df.head(DATASET_SIZE)

if verbose:
    print('Before Preprocessing:')
    print(df.head(3))
    print(f'   Dataframe Shape: {df.shape}')
    num_words = df[KEY_COLUMN_NAME].apply(lambda x: len(x.split(' '))).sum()
    print(f'   Number of words in dataset: {num_words}')

# Preprocess the text
df[KEY_COLUMN_NAME] = df[KEY_COLUMN_NAME].apply(Helpers.clean_text)

if verbose:
    print('Preprocessing Complete. Time: {} min'.format(round((time() - t) / 60, 2)))
    num_words_2 = df[KEY_COLUMN_NAME].apply(lambda x: len(x.split(' '))).sum()
    print(f'   Number of words in dataset: {num_words_2}, delta: -{(num_words - num_words_2)}')

df = df.reset_index()
df = df.drop('index',1)
#print(df.head(5))

num_sentences = int(round(DATASET_SIZE * TRAIN_PCT, 1))
sentences = df[KEY_COLUMN_NAME].iloc[1:num_sentences].values.tolist()
words = " ".join(map(str, sentences))

print(f'Sentences: {len(sentences)}. Time: {format(round((time() - t) / 60, 2))} min')
print(f'Words: {len(words)}. Time: {format(round((time() - t) / 60, 2))} min')

vocabulary_size_tf = 40000


def build_dataset(sentences):
    words = ''.join(sentences).split()
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size_tf - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0
    sent_data = []
    for sentence in sentences:
        data = []
        for word in sentence.split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count = unk_count + 1
            data.append(index)
        sent_data.append(data)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return sent_data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(sentences)
print('Most Common', count[:10])
print('Sample data', data[:2])

skip_window = 3
instances = 0

for i in range(len(data)):  # Pad sentence with skip_windows
    data[i] = [vocabulary_size_tf] * skip_window + data[i] + [vocabulary_size_tf] * skip_window

for sentence in data:  # Check how many training samples that we get
    instances += len(sentence) - 2 * skip_window

context = np.zeros((instances, skip_window * 2 + 1), dtype=np.int32)
labels = np.zeros((instances, 1), dtype=np.int32)
doc = np.zeros((instances, 1), dtype=np.int32)

k = 0
for doc_id, sentence in enumerate(data):
    for i in range(skip_window, len(sentence) - skip_window):
        context[k] = sentence[i - skip_window:i + skip_window + 1]  # Get surrounding words
        labels[k] = sentence[i]  # Get target variable
        doc[k] = doc_id
        k += 1

context = np.delete(context, skip_window, 1)  # delete the middle word

shuffle_idx = np.random.permutation(k)
labels = labels[shuffle_idx]
doc = doc[shuffle_idx]
context = context[shuffle_idx]

batch_size = 256
context_window = 2 * skip_window
embedding_size = 50  # Dimension of the embedding vector.
softmax_width = embedding_size
num_sampled = 5  # Number of negative examples to sample.
sum_ids = np.repeat(np.arange(batch_size), context_window)

len_docs = len(data)
graph = tf.Graph()

with graph.as_default():
    train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size * context_window])
    train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    segment_ids = tf.constant(sum_ids, dtype=tf.int32)

    word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size_tf, embedding_size], -1.0, 1.0))
    word_embeddings = tf.concat([word_embeddings, tf.zeros((1, embedding_size))], 0)
    doc_embeddings = tf.Variable(tf.random_uniform([len_docs, embedding_size], -1.0, 1.0))

    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size_tf, softmax_width],
                                                      stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size_tf]))

    # Look up embeddings for inputs.
    embed_words = tf.segment_mean(tf.nn.embedding_lookup(word_embeddings, train_word_dataset), segment_ids)
    embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
    embed = (embed_words + embed_docs) / 2.0  # +embed_hash+embed_users

    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(softmax_weights, softmax_biases, train_labels, embed, num_sampled, vocabulary_size_tf))
    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
    normalized_doc_embeddings = doc_embeddings / norm

# Chunk the data to be passed into the tensorflow Model
data_idx = 0


def generate_batch(batch_size):
    global data_idx

    if data_idx + batch_size < instances:
        batch_labels = labels[data_idx:data_idx + batch_size]
        batch_doc_data = doc[data_idx:data_idx + batch_size]
        batch_word_data = context[data_idx:data_idx + batch_size]
        data_idx += batch_size
    else:
        overlay = batch_size - (instances - data_idx)
        batch_labels = np.vstack([labels[data_idx:instances], labels[:overlay]])
        batch_doc_data = np.vstack([doc[data_idx:instances], doc[:overlay]])
        batch_word_data = np.vstack([context[data_idx:instances], context[:overlay]])
        data_idx = overlay
    batch_word_data = np.reshape(batch_word_data, (-1, 1))

    return batch_labels, batch_word_data, batch_doc_data


num_steps = 1000001
step_delta = int(num_steps / 20)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(f'Initialized. Time: {format(round((time() - t) / 60, 2))} min')
    average_loss = 0
    for step in range(num_steps):
        batch_labels, batch_word_data, batch_doc_data \
            = generate_batch(batch_size)
        feed_dict = {train_word_dataset: np.squeeze(batch_word_data),
                     train_doc_dataset: np.squeeze(batch_doc_data),
                     train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % step_delta == 0:
            if step > 0:
                average_loss = average_loss / step_delta
            # estimate avg loss over last 2000 batches
            print('EPOCH %d Loss: %f' % (step / 50000, average_loss), end='')
            print(f'. Time: {format(round((time() - t) / 60, 2))} min')
            average_loss = 0

    final_word_embeddings = word_embeddings.eval()
    final_word_embeddings_out = softmax_weights.eval()
    final_doc_embeddings = normalized_doc_embeddings.eval()


def print_test_update(cor, tot):
    pct = (cor * 100) / tot
    print(f'Accuracy: {round(pct, 1)}%')


def return_one_if_correct(test_doc, test_label, df, num_to_consider=1):
    ret_val = 0

    if num_to_consider == 1:
        # find closest doc
        dist = final_doc_embeddings.dot(final_doc_embeddings[int(test_doc)][:, None])
        closest_doc = np.argsort(dist, axis=0)[-4:][::-1]
        inproc_df = df.iloc[[closest_doc[0][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            ret_val = 1
    else:
        num_on_target = 0

        dist = final_doc_embeddings.dot(final_doc_embeddings[int(test_doc)][:, None])
        closest_doc = np.argsort(dist, axis=0)[-5:][::-1]

        inproc_df = df.iloc[[closest_doc[0][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            num_on_target += 1

        inproc_df = df.iloc[[closest_doc[1][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            num_on_target += 1

        inproc_df = df.iloc[[closest_doc[2][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            num_on_target += 1

        inproc_df = df.iloc[[closest_doc[3][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            num_on_target += 1

        inproc_df = df.iloc[[closest_doc[4][0]]]
        target_label = inproc_df.iloc[0][LABEL_COLUMN_NAME]
        if target_label == test_label:
            num_on_target += 1

        if num_on_target >= 2:
            ret_val = 1

    return ret_val


df2 = df.iloc[num_sentences:]
df2 = df2.reset_index()
df2 = df2.drop('index',1)
num_correct = 0
num_total = 0

for ind in df2.index:
    #print(f'Ind from df2: {ind} -- ', end='')
    #print(df2['Product'][ind])
    try:
        j = return_one_if_correct(ind, df2[LABEL_COLUMN_NAME][ind], df)
        num_total += 1
    except:
        j = 0

    num_correct += j
    if ind % 5 == 0:
        print_test_update(num_correct, tot=num_total)

print('*')
print('* 1 NEAREST ACCURACY FINAL:')
print_test_update(num_correct, tot=num_total)
print('*')


num_correct = 0
num_total = 0

for ind in df2.index:
    try:
        j = return_one_if_correct(ind, df2[LABEL_COLUMN_NAME][ind], df, 5)
        num_total += 1
    except:
        j = 0

    num_correct += j
    if ind % 5 == 0:
        print_test_update(num_correct, tot=num_total)

print('*')
print('* 5 NEAREST ACCURACY FINAL:')
print_test_update(num_correct, tot=num_total)
print('*')
