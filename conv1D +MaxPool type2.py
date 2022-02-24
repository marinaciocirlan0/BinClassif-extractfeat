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
from tensorflow.keras.optimizers import Adam 
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

# fit and evaluate a model
def evaluate_model(X_train, X_train_hrv, y_train, X_test, X_test_hrv, y_test):
    # define model
    # n_steps, n_length, n_validate = 100, 50, 1000 # ptb_xl_500
    # n_steps, n_length, n_validate = 50, 20, 1000 # ptb_xl_100
    # n_steps, n_length, n_validate = 50, 80, 1000 # ptb_xl_
    # n_steps, n_length, n_validate = 500, 240, 100 # ptb 1000 Hz
    n_steps, n_length, n_validate = 500, 24, 100 # ptb
    verbose, epochs, batch_size = 2, 100, 64

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
    
    X = Conv1D(filters=32, kernel_size=8, activation='relu', strides=2,
               input_shape=(None, n_length, n_features))(X)
    X = Conv1D(filters=32, kernel_size=8, activation='relu',
            strides=2, padding='same')(X)
    X = LeakyReLU()(X)
    
    X = Conv1D(filters=16, kernel_size=4, activation='relu', strides=2)(X)
    X = Conv1D(filters=16, kernel_size=4, activation='relu',
            strides=2, padding='same')(X)
    X = LeakyReLU()(X)
    
    X = Conv1D(filters=4, kernel_size=1, activation='relu', strides=2)(X)
    X = Conv1D(filters=4, kernel_size=1, activation='relu',
            strides=2, padding='same')(X)
    X = LeakyReLU()(X)
    
    X = Conv1D(filters=32, kernel_size=1, activation='relu')(X)
    X = Dropout(0.7)(X)
    X = Flatten()(X)
    X = Dropout(0.7)(X)

    # merge the hrv features if needed
    if X_train_hrv is not None and X_test_hrv is not None:
        n_hrv_features = X_train_hrv.shape[1]
        
        # hrv features branch
        X_hrv_input = Input(shape=(n_hrv_features, n_channels))
        X_hrv = Flatten()(X_hrv_input)
        X_hrv = Dense(100, activation='relu')(X_hrv)
        X = Concatenate()([X, X_hrv])
    
        # split hrv features in train and val
        X_val_hrv = X_train_hrv[:n_validate]
        X_train_hrv = X_train_hrv[n_validate:]

    X = Flatten()(X)
    X = Dense(200, activation='relu')(X)
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
    
    lr=0.0001
    opt = Adam(learning_rate=lr, decay=lr/epochs)
    # lr=0.00002
    # opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, decay=lr/epochs, amsgrad=False)
    model.compile(
        loss='binary_crossentropy',
        optimizer= opt,
        metrics=['categorical_accuracy']
    )
    model.summary()
    

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
    _, _accuracy = model.evaluate(
        X_evaluate,
        y_test,
        batch_size=batch_size,
        verbose=0
    )

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    
    _f1_score = f1_score(y_test, y_pred, average='macro')
    _recall_score=recall_score(y_test, y_pred, average='macro')
    _precision_score = precision_score(y_test, y_pred, average='macro')
    _roc_auc_score = roc_auc_score(y_test, y_pred, average='macro')

    return _accuracy, _f1_score, _recall_score, _precision_score, _roc_auc_score


# summarize scores
def summarize_results(_accuracy, _f1_score, _recall_score, _precision_score, _roc_auc_score):
    print(_accuracy)
    m, s = mean(_accuracy), std(_accuracy)
    print('Accuracies: %.3f%% (+/-%.3f)' % (m, s))
    m, s = mean(_f1_score), std(_f1_score)
    print('F1 score: %.3f%% (+/-%.3f)' % (m, s))
    m, s = mean(_recall_score), std(_recall_score)
    print('Recall score: %.3f%% (+/-%.3f)' % (m, s))
    m, s = mean(_precision_score), std(_precision_score)
    print('Precision score: %.3f%% (+/-%.3f)' % (m, s))
    m, s = mean(_roc_auc_score), std(_roc_auc_score)
    print('ROC curve: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(npz='ecg_100_bin.npz', use_hrv_features=False, repeats=10):
    # load data
    data = np.load('./data/' + npz)
    X_train, y_train, X_test, y_test = (data['X_train'], data['y_train'],
                                        data['X_test'], data['y_test'])
    
    # # remove data to balance the dataset
    # indices = np.where(y_train[:,1] == 1)[0]
    # X_train = np.delete(X_train, indices[8170:], axis=0)
    # y_train = np.delete(y_train, indices[8170:], axis=0)
    
    # load ecg features if needed
    if use_hrv_features:
        data = np.load('./data/' + npz.rsplit('_', 1)[0] + '_hrv.npz')    
        X_train_hrv = data['X_train_hrv']
        X_test_hrv = data['X_test_hrv']
    else:
        X_train_hrv = None
        X_test_hrv = None
        
        
    # repeat experiment
    accuracies = list()
    f1_scores = list()
    recall_scores = list()
    precision_scores = list()
    roc_auc_curve = list()

    # repeat experiment
    for r in range(repeats):
        (_accuracy, _f1_score,
        _recall_score, _precision_score, _roc_auc_curve) = evaluate_model(
            X_train, X_train_hrv, y_train,
            X_test, X_test_hrv, y_test
        )

        _accuracy = _accuracy * 100.0
        print('>#%d: %.3f' % (r+1, _accuracy))

        accuracies.append(_accuracy)
        f1_scores.append(_f1_score)
        recall_scores.append(_recall_score)
        precision_scores.append(_precision_score)
        roc_auc_curve.append(_roc_auc_curve)
        
    # summarize results
    summarize_results(accuracies, f1_scores, recall_scores, 
                      precision_scores, roc_auc_curve)

# run the experiment
# run_experiment(npz='ptb_200_bin.npz', use_hrv_features=False, repeats=1)
# run_experiment(npz='ecg_400_bin.npz', use_hrv_features=False, repeats=1)
# run_experiment(npz='ptb_bin.npz', use_hrv_features=True, repeats=1)
run_experiment(npz='ptb_100_bin.npz', use_hrv_features=True, repeats=1) #100 Hz
# run_experiment(npz='ecg_500_bin.npz', use_hrv_features=True, repeats=3)
# run_experiment(npz='ecg_100_cat.npz', use_hrv_features=True, repeats=1)
