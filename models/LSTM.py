
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

class Lstm:

    def __init__(self, X_train, y_train, X_valid, y_valid, embedding_layer):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid 
        self.embedding_layer = embedding_layer
        self.arguments  = {
                    'max_len': 200,
                    'batch_size': 128,
                    'epochs': 50,
                    'learning_rate': 1e-3,
                    'learning_rate_decay': 0,
                    'units': 128,
                    'drop_out_rate': 0.2,
                    'checkpoint_path': 'best_model.hdf5',
                    'early_stop_patience': 3,
                }


    def build_model(self):

        print('Building model...')

        file_path = self.arguments['checkpoint_path']
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                      save_best_only = True, mode = "min")
        ra_val = RocAucEvaluation(validation_data=(self.X_valid, self.y_valid), interval = 1)
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = self.arguments['early_stop_patience'])
        
        inp = Input(shape=(self.arguments['max_len'],))
        x = self.embedding_layer(inp)
        x = CuDNNLSTM(self.arguments['units'], return_sequences=True, name='lstm_layer')(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(self.arguments['drop_out_rate'])(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(self.arguments['drop_out_rate'])(x)
        # using sigmoid since we are doing six binary classifications
        if self.y_train.ndim == 2:
            output = Dense(self.y_train.shape[1], activation='sigmoid')(x)
        else:
            output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs = inp, outputs = output)
        self.model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = self.arguments['learning_rate'], decay = self.arguments['learning_rate_decay']),\
                      metrics = ["accuracy"])
        history = self.model.fit(self.X_train, self.y_train, batch_size = self.arguments['batch_size'], epochs = self.arguments['epochs'],\
                            validation_data = (self.X_valid, self.y_valid), verbose = 1, callbacks = [ra_val, check_point, early_stop])
        self.model = load_model(file_path)

        print('Finished Building Lstm Model as class attribute class.model')
        return self