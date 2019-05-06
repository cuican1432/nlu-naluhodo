import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

class DataEmbeddings:

	def __init__(self, data_index, nth_sample = 0):

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

		self.nth_sample = nth_sample
		self.data_index = [data_index['train'][self.nth_sample], data_index['val'][self.nth_sample], data_index['test']]
		self.content_column = 'Contents'
		self.label_column = [str(i+1) for i in range(6)]
		self.max_len = 200
		self.max_features = 100000
		self.embed_size = 300

		print('Creating data sample ' + str(self.nth_sample+1) + ' out of 3...')
		print('max_len : ' + str(self.max_len))
		print('max_feature : ' + str(self.max_features))
		print('embed_size : ' + str(self.embed_size))

	def load_data_embeddings(self, dataset, embedding_path):

		'''
		return processed final datasets with embeddings:
		X_train, y_train, X_valid, y_valid, X_test, y_test, embedding_layer
		'''

	    # Build dictionary from word embedding
		print('Loading word vectors...');
	    
		def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

		word2vec = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embedding_path)))

		print('Found %s word vectors.' % len(word2vec))

		# prepare text samples and their labels
		print('Loading email contents...')
		# extract the comments, fill NaN with some values
		contents = dataset[self.content_column].fillna("DUMMY_VALUE").values
		labels = dataset[self.label_column].values

		if self.data_index == None:
		    X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size = 0.2, random_state=0)
		    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state=0)
		else:
		    X_train = contents[self.data_index[0]]
		    y_train = labels[self.data_index[0]]
		    X_valid = contents[self.data_index[1]]
		    y_valid = labels[self.data_index[1]]
		    X_test = contents[self.data_index[2]]
		    y_test = labels[self.data_index[2]]

		# convert the sentences (strings) into integers, thus they can be used as index later on
		tokenizer = Tokenizer(num_words=self.max_features)
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

		X_train = pad_sequences(X_train_seq, maxlen=self.max_len)
		X_valid = pad_sequences(X_valid_seq, maxlen=self.max_len)
		X_test = pad_sequences(X_test_seq, maxlen=self.max_len)
		print('Shape of data tensor:', X_train.shape)

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
		embedding_layer = Embedding(
		  min(self.max_features, embedding_matrix.shape[0]),
		  self.embed_size,
		  weights=[embedding_matrix],
		  input_length=self.max_len,
		    # don't want to make the embeddding updated during the procedure
		  trainable=False
		)

		print('Finished.')
		print('Generated X_train, y_train, X_valid, y_valid, X_test, y_test, embedding_layer as class attributes.')

		return X_train, y_train, X_valid, y_valid, X_test, y_test, embedding_layer


class BertEmbeddings:

	def __init__(self, data_index = None, nth_sample = 0):

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

		self.nth_sample = nth_sample
		self.max_len = None
		self.embed_size = None
		self.data_index = None
		if data_index == None:
			print('Creating data sample...')
		else:
			self.data_index = [data_index['train'][self.nth_sample], data_index['val'][self.nth_sample], data_index['test']]
			print('Creating data sample ' + str(self.nth_sample+1) + ' out of 3...')

	def load_bert_embeddings(self, bert_path):
		"""
		Input:
		bert_path: bert embedding saving path, pickle file only
		"""
		print('Loading Bert embeddings...')
		bert_input = pickle.load(open(bert_path, 'rb'))
            
		matrix = []
		labels = []
		for i in bert_input:
			instance = []
			for index, key in enumerate(i):
				if key != '__labels__':
					instance += [i[key]]
				else:
					labels += [i[key]]
					break
			matrix += [np.array(instance)]
		matrix = np.array(matrix)
		labels = np.array(labels)
            
		self.max_len = max(x.shape[0] for x in matrix)
        
		data_matrix = []
		for i in matrix:
			data_matrix += [np.pad(i, ((0, self.max_len - i.shape[0]), (0, 0)), 'constant')]
		data_matrix = np.array(data_matrix)

		self.embed_size = data_matrix.shape[2]
		print('max_len : ' + str(self.max_len))
		print('embed_size : ' + str(self.embed_size))
            
		if self.data_index == None:
			X_train, X_test, y_train, y_test = train_test_split(data_matrix, labels, test_size = 0.2, random_state=0)
			X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.125, random_state=0)
		else:
			X_train = data_matrix[self.data_index[0]]
			y_train = labels[self.data_index[0]]
			X_valid = data_matrix[self.data_index[1]]
			y_valid = labels[self.data_index[1]]
			X_test = data_matrix[self.data_index[2]]
			y_test = labels[self.data_index[2]]
                
		return X_train, y_train.astype('int'), X_valid, y_valid.astype('int'), X_test, y_test.astype('int')
