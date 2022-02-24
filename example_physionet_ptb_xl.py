import ast
import wfdb

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


path = './data/'
sampling_rate = 500
binary = False
categorical = True
npz_filename = './data/ecg_' + str(sampling_rate)

if binary and categorical:
    raise Exception('Only one option must be available: binary or categorical')

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# split data into train and test
test_fold = 10

# train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

# test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

if categorical:
    mlb = MultiLabelBinarizer()

    y_train = pd.DataFrame(
        mlb.fit_transform(y_train),
        columns=mlb.classes_
    ).to_numpy()

    y_test = pd.DataFrame(
        mlb.fit_transform(y_test),
        columns=mlb.classes_
    ).to_numpy()
    
    npz_filename += '_cat'

if binary:
    mlb = MultiLabelBinarizer()
    
    y_train = y_train.astype('str')
    y_train.loc[y_train != "['NORM']"] = 'O'
    y_train.loc[y_train == "['NORM']"] = 'N'
    y_train = pd.DataFrame(
        mlb.fit_transform(y_train),
        columns=mlb.classes_
    ).to_numpy()

    y_test = y_test.astype('str')
    y_test.loc[y_test != "['NORM']"] = 'O'
    y_test.loc[y_test == "['NORM']"] = 'N'
    y_test = pd.DataFrame(
        mlb.fit_transform(y_test),
        columns=mlb.classes_
    ).to_numpy()
    
    npz_filename += '_bin'


# save data to npz file
np.savez(
    npz_filename,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
