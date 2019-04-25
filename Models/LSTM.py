import logging
from sklearn.metrics import roc_auc_score, f1_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Input, CuDNNLSTM, CuDNNGRU, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer

arguments = {
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

def build_lstm_model(X_train, y_train, X_valid, y_valid, arguments):

    print('Building model...')

    file_path = arguments['checkpoint_path']
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    ra_val = RocAucEvaluation(validation_data=(X_valid, y_valid), interval = 1)
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = arguments['early_stop_patience'])
    
    inp = Input(shape=(arguments['max_len'],))
    x = embedding_layer(inp)
    x = CuDNNLSTM(arguments['units'], return_sequences=True, name='lstm_layer')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(arguments['drop_out_rate'])(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(arguments['drop_out_rate'])(x)
    # using sigmoid since we are doing six binary classifications
    if y_train.ndim == 2:
        output = Dense(y_train.shape[1], activation='sigmoid')(x)
    else:
        output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs = inp, outputs = output)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = arguments['learning_rate'], decay = arguments['learning_rate_decay']),\
                  metrics = ["accuracy"])
    history = model.fit(X_train, y_train, batch_size = arguments['batch_size'], epochs = arguments['epochs'],\
                        validation_data = (X_valid, y_valid), verbose = 1, callbacks = [ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model
