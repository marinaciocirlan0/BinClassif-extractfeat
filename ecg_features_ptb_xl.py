import numpy as np
import neurokit2 as nk

from tqdm import tqdm


def extract_ecg_features(X, sampling_rate):
    try:    
        # find peaks
        peaks, info = nk.ecg_peaks(X, sampling_rate=sampling_rate)
    
        # compute hrv features        
        hrv_time = nk.hrv_time(
            peaks,
            sampling_rate=sampling_rate,
            show=False
        ).to_numpy()
    except:
        hrv_time = np.zeros((1, 20))
        pass
    
    return hrv_time


if __name__ == "__main__":
    # load data
    sampling_rate = 500
    npz = 'ecg_{}_bin.npz'.format(sampling_rate)
    data = np.load('./data/' + npz)
    X_train, y_train, X_test, y_test = (data['X_train'], data['y_train'],
                                        data['X_test'], data['y_test'])
    
    n_hrv_features = 20
    n_train_signals = X_train.shape[0]
    n_test_signals = X_test.shape[0]
    n_features = X_train.shape[1]
    n_channels = X_train.shape[2]
    
    X_train_hrv = np.zeros((n_train_signals, n_hrv_features, n_channels))
    X_test_hrv = np.zeros((n_test_signals, n_hrv_features, n_channels))
    
    for i_channel in tqdm(range(n_channels)):
        for i_signal in tqdm(range(n_train_signals)):
            data = X_train[i_signal, :, i_channel]
            X_train_hrv[i_signal, :, i_channel] = extract_ecg_features(data, sampling_rate)
            
        for i_signal in tqdm(range(n_test_signals)):
            data = X_test[i_signal, :, i_channel]
            X_test_hrv[i_signal, :, i_channel] = extract_ecg_features(data, sampling_rate)

    X_train_hrv = np.nan_to_num(X_train_hrv, nan=0, posinf=0, neginf=0)
    X_test_hrv = np.nan_to_num(X_test_hrv, nan=0, posinf=0, neginf=0)

    # save data to npz file
    np.savez(
        npz.rsplit('_', 1)[0] + '_hrv.npz',
        X_train_hrv=X_train_hrv,
        X_test_hrv=X_test_hrv
    )
