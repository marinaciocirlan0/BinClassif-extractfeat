import numpy as np
import pandas as pd

meta = './meta.csv'
npz = './data_raw.npz'
data = np.load(npz)
#ecg = data['patient098/s0389lre'][:, 0]

# get max size of signals
max_size = 120000

#for record in data.files:
#    if max_size < data[record].shape[0]:
#        max_size = data[record].shape[0]

df = pd.read_csv(meta)
df = df.dropna(axis=0, subset=['Reason_for_admission'])

index = 0
X = np.zeros((df.shape[0], max_size, 15))
y = np.zeros((df.shape[0], 1))

for _, row in df.iterrows():
    label = row.Reason_for_admission == 'Healthy control'
    signal = data[row.patient + '/' + row.record_id]
    
    X[index, :len(signal), :] = signal
    y[index] = int(label)
    index += 1
    
np.savez(
    'ptb_bin_raw.npz',
    X=X,
    y=y
)
