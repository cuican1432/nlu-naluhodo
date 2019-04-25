import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main(data_set, embedding_path, data_index=None, content_column='contents', label_column='labels', max_len=200, max_features=100000, embed_size=300):

    """
    Input:
    data_set: pandas DataFrame that contains text contents and labels
    embedding_path: word embedding saving path
    data_index: list of np.array that contains the training, validation, and test indeces
    content_column: Name of content column, default is 'contents'
    label_column: Name of label column(s), default is 'labels'
    max_len: the padding lenth of the text, default is 200
    max_features: Maximum number of words recognized by the model, default is 100000
    embed_size: length of embedding vector, should correspond with the embedding provided, default is 300
    """

    # Build dictionary from word embedding
    print('Loading word vectors...')
    
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    word2vec = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embedding_path)))
    
    print('Found %s word vectors.' % len(word2vec))

    # prepare text samples and their labels
    print('Loading in comments...')
    # extract the comments, fill NaN with some values
    contents = data_set[content_column].fillna("DUMMY_VALUE").values
    labels = data_set[label_column].values
    
    if data_index == None:
        X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size = 0.2, random_state=0)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state=0)
    else:
        X_train = contents[data_index[0]]
        y_train = labels[data_index[0]]
        X_valid = contents[data_index[1]]
        y_valid = labels[data_index[1]]
        X_test = contents[data_index[2]]
        y_test = labels[data_index[2]]

    # convert the sentences (strings) into integers, thus they can be used as index later on
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    #sequences = tokenizer.texts_to_sequences(sentences)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_valid_seq = tokenizer.texts_to_sequences(X_valid)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    print("max sequence length:", max(len(s) for s in X_train_seq))
    print("min sequence length:", min(len(s) for s in X_train_seq))
    s = sorted(len(s) for s in X_train_seq)
    print("median sequence length:", s[len(s) // 2])

    # get word -> integer mapping
    word2idx = tokenizer.word_index
    print('Found %s unique tokens.' % len(word2idx))

    # pad sequences so that we get a N x T matrix
    # Keras take care of the 0 only for padding purpose 
    #data = pad_sequences(sequences, maxlen=max_len)
    X_train = pad_sequences(X_train_seq, maxlen=max_len)
    X_valid = pad_sequences(X_valid_seq, maxlen=max_len)
    X_test = pad_sequences(X_test_seq, maxlen=max_len)
    print('Shape of data tensor:', X_train.shape)

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(max_features, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word2idx.items():
        if i < max_features:
            embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(
      min(max_features, embedding_matrix.shape[0]),
      embed_size,
      weights=[embedding_matrix],
      input_length=max_len,
        # don't want to make the embeddding updated during the procedure
      trainable=False
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test, embedding_layer
