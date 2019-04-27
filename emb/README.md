### Embeddings Categories

We tested both using pre-trained word vectors and training word vectors on a larger email corpus [data source](https://www.cs.cmu.edu/~./enron/).

#### Pretrained Embeddings Include:

* glove.6B.100~300d.txt [data source](http://nlp.stanford.edu/data/glove.6B.zip "glove.6B.zip")
* bert-embeddings see `bert-embedding-feature-extraction.ipynb` [data source](https://github.com/cuican1432/nlu-naluhodo/blob/master/emb/bert_embedding_feature_extraction.ipynb  "bert_embedding_feature_extraction.ipynb") and [reference](https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b) 

#### Customized Embeddings (train word vectors on the new corpus):

* GloVe_ec.100~300B.txt [data source](https://drive.google.com/file/d/1KjFlxapoROMzudGXW4X7oiUtJWAHY1M_/view?usp=sharing "GloVe_ec.100~300B.txt") and [reference](https://github.com/stanfordnlp/GloVe) 



`load_embedding.py` is a class as a helper to load, transform, match embeddings for model use (data processing).
