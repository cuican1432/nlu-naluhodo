import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Dataset:

	def __init__(self, df_dataset, embedding_path):

		'''
	    df_dataset: pandas DataFrame that contains text contents and labels
	    embedding_path: word embedding saving path
	    self.data_index: list of np.array that contains the training, validation, and test indeces
	    content_column: Name of content column, default is 'contents'
	    label_column: Name of label column(s), default is 'labels'
	    max_len: the padding lenth of the text, default is 200
	    max_features: Maximum number of words recognized by the model, default is 100000
	    embed_size: length of embedding vector, should correspond with the embedding provided, default is 300
	   ''' 

		self.dataset = df_dataset
		self.embedding_path = embedding_path
		self.self.data_index = None
		self.content_column = 'Contents'
		self.label_column = [str(i+1) for i in range(6)]
		self.max_len = 200
		self.max_features = 100000
		self.embed_size = 300


	def load_data(self):

		'''
		return processed final datasets with embeddings:
		self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.embedding_layer
		'''

	    # Build dictionary from word embedding
	    print('Loading word vectors...')
	    # word2vec = {}
	    # with tqdm(open(embedding_path)) as f:
	    # with open(embedding_path) as f:
	    #    for line in f:
	    #        values = line.split()
	    #        word = values[0]
	    #        vec = np.asarray(values[1:], dtype='float32')
	    #        word2vec[word] = vec
	    
	    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

	    word2vec = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(self.embedding_path)))
	    
	    print('Found %s word vectors.' % len(word2vec))

	    # prepare text samples and their labels
	    print('Loading email contents...')
	    # extract the comments, fill NaN with some values
	    contents = self.dataset[self.content_column].fillna("DUMMY_VALUE").values
	    labels = self.dataset[self.label_column].values
	    
	    if self.data_index == None:
	        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(contents, labels, test_size = 0.2, random_state=0)
	        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size = 0.125, random_state=0)
	    else:
	        self.X_train = contents[self.data_index[0]]
	        self.y_train = labels[self.data_index[0]]
	        self.X_valid = contents[self.data_index[1]]
	        self.y_valid = labels[self.data_index[1]]
	        self.X_test = contents[self.data_index[2]]
	        self.y_test = labels[self.data_index[2]]

	    # convert the sentences (strings) into integers, thus they can be used as index later on
	    tokenizer = Tokenizer(num_words=self.max_features)
	    tokenizer.fit_on_texts(self.X_train)
	    #sequences = tokenizer.texts_to_sequences(sentences)
	    self.X_train_seq = tokenizer.texts_to_sequences(self.X_train)
	    self.X_valid_seq = tokenizer.texts_to_sequences(self.X_valid)
	    self.X_test_seq = tokenizer.texts_to_sequences(self.X_test)

	    print("max sequence length:", max(len(s) for s in self.X_train_seq))
	    print("min sequence length:", min(len(s) for s in self.X_train_seq))
	    s = sorted(len(s) for s in self.X_train_seq)
	    print("median sequence length:", s[len(s) // 2])

	    # get word -> integer mapping
	    word2idx = tokenizer.word_index
	    print('Found %s unique tokens.' % len(word2idx))

	    # pad sequences so that we get a N x T matrix
	    # Keras take care of the 0 only for padding purpose 
	  
	    self.X_train = pad_sequences(self.X_train_seq, maxlen=self.max_len)
	    self.X_valid = pad_sequences(self.X_valid_seq, maxlen=self.max_len)
	    self.X_test = pad_sequences(self.X_test_seq, maxlen=self.max_len)
	    print('Shape of data tensor:', self.X_train.shape)

	    # prepare embedding matrix
	    print('Filling pre-trained embeddings...')
	    num_words = min(self.max_features, len(word2idx) + 1)
	    embedding_matrix = np.zeros((num_words, self.embed_size))
	    for word, i in word2idx.items():
	        if i < self.max_features:
	            embedding_vector = word2vec.get(word)
	        if embedding_vector is not None:
	          # words not found in embedding index will be all zeros.
	            embedding_matrix[i] = embedding_vector

	    # load pre-trained word embeddings into an Embedding layer
	    # note that we set trainable = False so as to keep the embeddings fixed
	    self.embedding_layer = Embedding(
	      min(self.max_features, embedding_matrix.shape[0]),
	      self.embed_size,
	      weights=[embedding_matrix],
	      input_length=self.max_len,
	        # don't want to make the embeddding updated during the procedure
	      trainable=False
	    )

	    return self
