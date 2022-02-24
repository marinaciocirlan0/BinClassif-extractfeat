import glob

import numpy as np
import pandas as pd
import wfdb


# path to contents of archive from https://physionet.org/content/ptbdb/1.0.0/
p = './ptb-diagnostic-ecg-database-1.0.0/'

template = p + "*/*.hea"
file_list = glob.glob(template)

meta = []
data_raw = {}

for file in file_list:
    patient, record_id = file[len(p):-4].split("/")
    m = wfdb.rdsamp(file[:-4])

    # convert data
    arr = m[0]

    key = patient + "/" + record_id
    data_raw[key] = arr.astype(np.float32)

    # generate meta data
    dct = {}
    dct.update({"patient": patient, "record_id": record_id})
    dct.update(m[1])
    meta.append(dct)
    
np.savez_compressed("data_raw.npz", **data_raw)    

df = pd.DataFrame(meta)
df.sort_values(['patient', 'record_id'], ascending=[True, True], inplace=True)
df.head()

def convert_to_dct(row):
    rowp = [r.split(":") for r in row]
    rowp = [(x[0].strip().replace(" ", "_"), np.nan if x[1].strip() in ["", "n/a"] else x[1].strip()) for x in rowp]
    assert all([len(x) == 2 for x in rowp])
    return dict(rowp)

df["comments"] = df["comments"].apply(convert_to_dct)
df = pd.concat([df.drop(['comments'], axis=1), df['comments'].apply(pd.Series)], axis=1)

df.head()
df.to_csv("meta.csv", index=False)