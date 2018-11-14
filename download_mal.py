import os
from vt import VTConnection
from storage import Storage
import logging
from globals import DATAMALDIR


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
vt_key = os.environ['VT_API_KEY']

conn = VTConnection(vt_key)

query = 'type:pdf positives:10+'
outfile = 'data/malicious.csv'


def search_files():
    conn.search(query, outfile=outfile)


def download_files():
    storage = Storage()
    with open(outfile) as f:
        all_mals = f.read().split()
    for mal in all_mals:
        content = conn.download_file(mal, store=False)
        storage.put(DATAMALDIR + mal, content)
        logger.info('Downloaded %s' % mal)

if __name__ == "__main__":
    download_files()


