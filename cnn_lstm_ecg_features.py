# cnn lstm model
import numpy as np

from numpy import mean
from numpy import std

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Reshape, Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D


# fit and evaluate a model
def evaluate_model(X_train, X_train_hrv, y_train, X_test, X_test_hrv, y_test):
    # define model
    # n_steps, n_length, n_validate = 100, 50, 1000 # ptb_xl_500
    n_steps, n_length, n_validate = 50, 20, 1000 # ptb_xl_100
    # n_steps, n_length, n_validate = 500, 240, 100 # ptb
    verbose, epochs, batch_size = 2, 25, 64

    n_features = X_train.shape[1]
    n_channels = X_train.shape[2]
    n_outputs = y_train.shape[1]

    # split training data to train and valid
    X_val = X_train[:n_validate]
    y_val = y_train[:n_validate]
    X_train = X_train[n_validate:]
    y_train = y_train[n_validate:]

    # cnn features extractor branch
    X_input = Input(shape=(n_features, n_channels))
    X = Reshape((n_steps, n_length, n_channels))(X_input)
    X = Conv1D(filters=32, kernel_size=5, activation='relu',
               input_shape=(None, n_length, n_features))(X)
    X = Conv1D(filters=64, kernel_size=7, activation='relu')(X)
    X = Conv1D(filters=32, kernel_size=5, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)

    # merge the hrv features if needed
    if X_train_hrv is not None and X_test_hrv is not None:
        n_hrv_features = X_train_hrv.shape[1]
        
        # hrv features branch
        X_hrv_input = Input(shape=(n_hrv_features, n_channels))
        X_hrv = Flatten()(X_hrv_input)
        # X_hrv = Dense(100, activation='relu')(X_hrv)
        X = Concatenate()([X, X_hrv])
    
        # split hrv features in train and val
        X_val_hrv = X_train_hrv[:n_validate]
        X_train_hrv = X_train_hrv[n_validate:]

    X = Dense(100, activation='relu')(X)
    X = Dense(n_outputs, activation='sigmoid')(X)

    if X_train_hrv is not None and X_test_hrv is not None:
        model = Model(
            inputs=[X_input, X_hrv_input],
            outputs=X
        )
        
        X_fit = [X_train, X_train_hrv]
        X_validate = [X_val, X_val_hrv]
        X_evaluate = [X_test, X_test_hrv]
    else:
        model = Model(
            inputs=[X_input],
            outputs=X
        )
        
        X_fit = [X_train]
        X_validate = [X_val]
        X_evaluate = [X_test]

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )

    # fit network
    model.fit(
        X_fit,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(X_validate, y_val)
    )

    # evaluate model
    _, accuracy = model.evaluate(
        X_evaluate,
        y_test,
        batch_size=batch_size,
        verbose=0
    )

    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(npz='ecg_100_bin.npz', use_hrv_features=False, repeats=10):
    # load data
    data = np.load('./data/' + npz)
    X_train, y_train, X_test, y_test = (data['X_train'], data['y_train'],
                                        data['X_test'], data['y_test'])
    
    # load ecg features if needed
    if use_hrv_features:
        data = np.load('./data/' + npz.rsplit('_', 1)[0] + '_hrv.npz')    
        X_train_hrv = data['X_train_hrv']
        X_test_hrv = data['X_test_hrv']
    else:
        X_train_hrv = None
        X_test_hrv = None

    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, X_train_hrv, y_train,
                               X_test, X_test_hrv, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)

    # summarize results
    summarize_results(scores)

# run the experiment
run_experiment(npz='ptb_bin.npz', use_hrv_features=True, repeats=1)
# run_experiment(npz='ptb_bin.npz', use_hrv_features=True, repeats=1)
