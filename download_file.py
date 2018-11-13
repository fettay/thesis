from storage import Storage
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
    if (DATADIR + row[5]) in downloaded:
        return row[5]
    try:
        cnt = requests.get(row[2]).content
        sha1 = hashlib.sha1(cnt).hexdigest()
        if sha1 != row[5]:
            return
        storage.put(DATADIR + sha1, cnt)
        logger.info('Stored %s' % sha1)
        return sha1
    except Exception:
        return


def loop_csv(fname):
    with open(fname, 'rb') as f:
        data = f.read().decode('latin-1').split("\n")
    for d in data:
        yield d.split(',')


def run():
    rows = loop_csv(BENIGN_DF)
    next(rows)
    with Pool(20) as p:
        shas = p.map(download_one, rows)
    with open('data/benign_downloaded.csv', 'w') as f:
        f.write(",".join([s for s in shas if s is not None]))

if __name__ == '__main__':
    run()