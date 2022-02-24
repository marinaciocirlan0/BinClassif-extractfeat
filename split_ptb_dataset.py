import numpy as np

from tensorflow.keras.utils import to_categorical


# load data
npz = 'ptb_bin_raw.npz'
# npz = 'ptb_300_bin.npz'
data = np.load('./data/' + npz)
X, y = data['X'], data['y']

# truncate signals to 120k
X = X[:, :120000, :]

npz = 'ptb_bin_hrv_raw.npz'
#npz = 'ptb_300_bin_hrv_raw.npz'
data = np.load('./data/' + npz)
X_hrv, y_hrv = data['X_hrv'], data['y']

y = to_categorical(y)
y_hrv = to_categorical(y_hrv)

train_test_split = 0.6
position = int(X.shape[0] * train_test_split)
rp = np.random.permutation(X.shape[0])

X_train = X[rp[:position], :, :]
X_test = X[rp[position:], :, :]
y_train = y[rp[:position]]
y_test = y[rp[position:]]

X_train_hrv = X_hrv[rp[:position], :, :]
X_test_hrv = X_hrv[rp[position:], :, :]
y_train_hrv = y_hrv[rp[:position]]
y_test_hrv = y_hrv[rp[position:]]

# save data to npz file
np.savez(
    './data/ptb_bin.npz',
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

np.savez(
    './data/ptb_bin_hrv.npz',
    X_train_hrv=X_train_hrv,
    X_test_hrv=X_test_hrv,
    y_train_hrv=y_train_hrv,
    y_test_hrv=y_test_hrv
)