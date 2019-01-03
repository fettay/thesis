import os
from vt import VTConnection
from storage import Storage
import logging
from globals import DATAMALDIR
import schedule
import time


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
    all_mals = [m.replace('.pdf', '') for m in all_mals]
    for mal in all_mals:
        try:
            downloaded = storage.get(DATAMALDIR + mal)
            if len(downloaded) > 0:
                logger.info("Already downloaded %s" % mal)
                continue
        except Exception: # FileNotFound in bucket
            pass
        content = conn.download_file(mal, store=False)
        if len(content) == 0:
            logger.error('Consumed all VT')
            return
        storage.put(DATAMALDIR + mal, content)
        logger.info('Downloaded %s' % mal)


def main():
    schedule.every().day.at("12:15").do(download_files)
    while(True):
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()


