import numpy as np
import neurokit2 as nk

from tqdm import tqdm

counter = 0

def extract_ecg_features(X, sampling_rate):
    global counter

    try:
        peaks, info = nk.ecg_peaks(X, sampling_rate=sampling_rate)    
        features = nk.hrv(peaks, sampling_rate=sampling_rate, show=False).to_numpy()
    except:
        counter += 1
        features = np.zeros((1, 72))
    return features


if __name__ == "__main__":
    # load data
    npz = 'ptb_bin_raw.npz'
    # npz = 'ptb_300_bin.npz'
    data = np.load('./data/' + npz)
    # sampling_rate = 1000
    sampling_rate = 300
    X, y = data['X'], data['y']
    
    n_hrv_features = 72
    n_signals = X.shape[0]
    n_features = X.shape[1]
    n_channels = X.shape[2]
    
    X_hrv = np.zeros((n_signals, n_hrv_features, n_channels))
    
    for i_channel in tqdm(range(n_channels)):
        for i_signal in tqdm(range(n_signals)):
            data = X[i_signal, :, i_channel]
            features = extract_ecg_features(data, sampling_rate)
            X_hrv[i_signal, :, i_channel] = features[0, :n_hrv_features]

    X_hrv = np.nan_to_num(X_hrv, nan=0, posinf=0, neginf=0)

    # save data to npz file
    np.savez(
        npz.split('.')[0] + '_hrv_raw.npz',
        X_hrv=X_hrv,
        y=y
    )
