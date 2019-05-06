from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Input, CuDNNLSTM, CuDNNGRU, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from models.utils import RocAucEvaluation


class Bert_Lstm:

    def __init__(self):
        self.arguments  = {
                    'max_len': None,
                    'batch_size': 128,
                    'epochs': 100,
                    'learning_rate': 1e-3,
                    'learning_rate_decay': 0,
                    'units': 128,
                    'drop_out_rate': 0.2,
                    'checkpoint_path': 'best_bilstm_model.hdf5',
                    'early_stop_patience': 10,
                }
        print('Building BiLstm Models ...')


    def fit(self, X_train, y_train, X_valid, y_valid):

        self.arguments['max_len'] == X_train.shape[1]
        print(self.arguments)
        file_path = self.arguments['checkpoint_path']
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                      save_best_only = True, mode = "min")
        ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = self.arguments['early_stop_patience'])

        inp = Input(shape=(self.arguments['max_len'], X_train.shape[2],))
        x1 = SpatialDropout1D(self.arguments['drop_out_rate'])(inp)
        x = Bidirectional(CuDNNGRU(self.arguments['units'], return_sequences = True))(x1)
        x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
        y = Bidirectional(CuDNNLSTM(self.arguments['units'], return_sequences = True))(x1)
        y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)
        avg_pool1 = GlobalAveragePooling1D()(x)
        max_pool1 = GlobalMaxPooling1D()(x)
        avg_pool2 = GlobalAveragePooling1D()(y)
        max_pool2 = GlobalMaxPooling1D()(y)
        x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])
        if y_train.ndim == 2:
            output = Dense(y_train.shape[1], activation='sigmoid')(x)
        else:
            output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs = inp, outputs = output)
        self.model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = self.arguments['learning_rate'], decay = self.arguments['learning_rate_decay']),\
                      metrics = ["accuracy"])
        history = self.model.fit(X_train, y_train, batch_size = self.arguments['batch_size'], epochs = self.arguments['epochs'],\
                            validation_data = (X_valid, y_valid), verbose = 1, callbacks = [ra_val, check_point, early_stop])
        self.model = load_model(file_path)

        print('Finished Building BertLstm Model as class attribute class.model')
        return self


    def predict(self, X, batch_size,  verbose):
        return self.model.predict(X, batch_size = batch_size, verbose = verbose)
