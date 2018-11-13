from storage import Storage
import pandas as pd
from globals import BENIGN_DF, DATADIR
import requests
import hashlib
import logging
from multiprocessing import Pool


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
storage = Storage()
downloaded = set(storage.ls())


def download_one(row):
    if (DATADIR + row['sha1_of_attached_file']) in downloaded:
        return row['sha1_of_attached_file']
    try:
        cnt = requests.get(row['file_url_src']).content
        sha1 = hashlib.sha1(cnt).hexdigest()
        if sha1 != row['sha1_of_attached_file']:
            return
        storage.put(DATADIR + sha1, cnt)
        logger.info('Stored %s' % sha1)
        return sha1
    except Exception:
        return


def run():
    df = pd.read_csv(BENIGN_DF)
    rows = (r for _, r in df.iterrows())
    with Pool(20) as p:
        shas = p.map(download_one, rows)
    with open('data/benign_downloaded.csv', 'w') as f:
        f.write(",".join([s for s in shas if s is not None]))

if __name__ == '__main__':
    run()