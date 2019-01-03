from globals import DATADIR, DATAMALDIR, LOCAL_BUCKET, BENIGN_LIST, MALICIOUS_LIST, BENIGN, MALICIOUS
from storage import FsStorage
import pandas as pd


def files_df():
    flist = []
    verdicts = []
    
    with open(BENIGN_LIST) as f:
        lines = f.read().split()
    flist.extend(lines)
    verdicts.extend([BENIGN]*len(lines))
    
    with open(MALICIOUS_LIST) as f:
        lines = f.read().split()
    flist.extend(lines)
    verdicts.extend([MALICIOUS]*len(lines))
    return pd.DataFrame({'hash': flist, 'verdict': verdicts})


def get_file(name, label):
    storage = FsStorage()
    if label == BENIGN:
        return storage.get(DATADIR + name)
    elif label == MALICIOUS:
        return storage.get(DATAMALDIR + name)
    raise ValueError("Bad label")
    
    
def split_train_test(df, test_ratio):
    df = df.sample(frac=1, random_state=42)  # Shuffle
    cut_index = (int(1-test_ratio*df.shape[0]))
    df_train, df_test = df.iloc[:cut_index], df.iloc[cut_index:]
    return df_train, df_test