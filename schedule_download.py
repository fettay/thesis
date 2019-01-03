from storage import Storage, FsStorage
import requests
import hashlib
import logging
from multiprocessing import Pool
import schedule
import time


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_all():
    logger.info('Start downloading the files')
    bucket_storage = Storage()
    local_storage = FsStorage()
    all_files = list(set(bucket_storage.ls()))
    for i, fname in enumerate(all_files):
        content = bucket_storage.get(fname)
        local_storage.put(fname, content)
        if i % 100 == 0:
            logger.info('%d files downloaded so far' % i)
    logger.info('Downloaded all files')
    

def main():
    schedule.every().sunday.at("7:15").do(download_all)
    while(True):
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
